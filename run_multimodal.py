import torch
import whisper
import os
from openai import OpenAI
from encoder import SLIPVideoEncoder 
from models import HybridTemporalModel
import sys # sys.stderr 출력을 위해 import
import numpy as np # 👈 numpy.ndarray 타입을 사용했으므로 추가합니다.

# ⚠️ API 키 파일을 읽는 함수 정의
def load_openai_api_key(filepath="openai.txt"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 파일의 첫 번째 줄을 키로 사용 (줄바꿈 제거)
            key = f.readline().strip()
            if not key:
                raise ValueError("API 키 파일이 비어 있습니다.")
            return key
    except FileNotFoundError:
        print(f"❌ 오류: API 키 파일 '{filepath}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"❌ 오류: API 키 파일을 읽는 중 오류 발생: {e}", file=sys.stderr)
        raise
        
def print_error(message):
    """오류 메시지를 sys.stderr로 출력하여 Electron 콘솔에 표시합니다."""
    print(message, file=sys.stderr)


class MultimodalAgent:
    def __init__(self, model_path, proto_path, device="cuda"):
        self.device = device
        
        # 1. 수어 모델 로드
        self.encoder = SLIPVideoEncoder(pretrained=False, embed_dim=512).to(device)
        self.temporal = HybridTemporalModel(input_dim=512, hidden_dim=512).to(device)
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.temporal.load_state_dict(checkpoint['temporal'])
        except FileNotFoundError:
            print_error(f"❌ 오류: 모델 파일 '{model_path}'을(를) 찾을 수 없습니다.")
            raise
        self.encoder.eval()
        self.temporal.eval()

        # 2. 프로토타입(기준점) 로드
        print_error("📂 수어 기준점(Prototype) 로딩 중...")
        try:
            # self.prototypes는 로드된 딕셔너리 전체
            data = torch.load(proto_path, map_location=device) 
        except FileNotFoundError:
            print_error(f"❌ 오류: 프로토타입 파일 '{proto_path}'을(를) 찾을 수 없습니다. make_prototypes.py를 먼저 실행하세요.")
            raise
        
        # 🚨🚨 수정된 부분: 로드된 딕셔너리 구조에 따라 클래스 이름과 프로토타입 추출 🚨🚨
        
        # 1) 만약 파일이 {'classes': [이름들], 'prototypes': Tensor} 구조라면:
        if isinstance(data, dict) and 'classes' in data and 'prototypes' in data:
            self.class_names = data['classes']
            self.proto_matrix = data['prototypes'].to(device)
            
            if not isinstance(self.proto_matrix, torch.Tensor):
                raise TypeError("로딩된 'prototypes' 키의 값이 Tensor가 아닙니다.")
        
        # 2) 만약 파일이 {클래스이름: 텐서, 클래스이름2: 텐서, ...} 형태라면:
        else:
            self.prototypes = data # 딕셔너리 {이름: 텐서}
            self.class_names = []
            proto_tensors = []
            
            for key, value in self.prototypes.items():
                # 텐서나 NumPy 배열만 필터링하고 문자열(str) 같은 것은 무시
                if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                    self.class_names.append(key)
                    # 텐서가 아니면 (NumPy 배열이면) 강제 변환
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, dtype=torch.float32)
                    proto_tensors.append(value)
                else:
                    # 'str' 같은 불순물은 경고만 출력하고 건너뜁니다.
                    print_error(f"⚠️ 경고: 프로토타입 딕셔너리에서 예상치 못한 값({key}: {type(value)})이 발견되어 무시합니다.")

            if not proto_tensors:
                 raise ValueError("프로토타입 딕셔너리에서 유효한 텐서를 찾을 수 없습니다.")
                 
            self.proto_matrix = torch.stack(proto_tensors).to(device)
            
        # -----------------------------------------------------------------------

        # 3. Whisper 로드
        print_error("🎧 Whisper 모델 로드 중...")
        try:
            self.whisper = whisper.load_model("base").to(device)
        except Exception as e:
            print_error(f"❌ 오류: Whisper 모델 로드 실패. 에러: {e}")
            raise

        # 4. LLM 클라이언트 (OpenAI 예시)
        # 🌟🌟🌟 수정된 부분: 파일에서 API 키 로드 🌟🌟🌟
        try:
            api_key = load_openai_api_key()
            self.client = OpenAI(api_key=api_key) 
        except Exception as e:
            print_error(f"❌ 오류: OpenAI 클라이언트 초기화 실패. 'openai.txt' 파일을 확인하세요. 에러: {e}")
            raise

    def predict_sign(self, video_tensor):
        """저장된 프로토타입과 비교하여 가장 가까운 수어 단어 찾기"""
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            features = self.encoder(video_tensor)
            query_emb = self.temporal(features) # (1, 512)

            # 유클리드 거리 계산 (Euclidean Distance)
            dists = torch.cdist(query_emb, self.proto_matrix) # (1, Class_Num)
            
            # 가장 거리가 짧은 인덱스 찾기
            min_dist_idx = torch.argmin(dists, dim=1).item()
            predicted_word = self.class_names[min_dist_idx]
            
            return predicted_word

    def generate_response(self, video_tensor, audio_path):
        # 1. 인식 수행
        sign_word = self.predict_sign(video_tensor)
        
        try:
            audio_result = self.whisper.transcribe(audio_path)['text']
        except Exception as e:
            print_error(f"❌ 오류: Whisper 음성 인식 실패. 에러: {e}")
            audio_result = "[음성 인식 실패]"
            
        print_error(f"\n👀 수어 인식: {sign_word}")
        print_error(f"👂 음성 인식: {audio_result}")

        # 2. LLM 프롬프트 (Prompt Engineering)
        system_prompt = "당신은 청각 장애인과 비장애인의 소통을 돕는 고급 통역 AI입니다.당신의 목표는 단순 번역이 아니라, 수어와 음성이라는 불완전한 멀티모달 입력을 통합 추론하여 사용자의 실제 의도를 재구성하는 것입니다. 당신은 반드시 다음 역할을 수행해야 합니다: 1. 수어 단어와 음성 텍스트 각각의 의미를 분리 해석 2. 두 정보 간의 보완 관계 또는 충돌 여부 판단 3. 사용자의 의도(목적, 대상, 요청, 방향성)를 추론 4. 인간 통역사가 말하듯 자연스러운 한국어 문장으로 재구성"
        
        user_prompt = f"""
        [입력 정보]
        - 수어 단어: {sign_word}
        - 음성 텍스트: {audio_result}

        [추론 지침]
        아래 단계를 순서대로 내부적으로 수행한 뒤, 최종 해석만 출력하세요.

        1️⃣ 수어 단어 분석
        - 수어 단어가 나타내는 핵심 개념, 행동, 장소, 감정은 무엇인가?

        2️⃣ 음성 텍스트 분석
        - 음성 텍스트에 포함된 요청, 상태, 대상, 방향 표현은 무엇인가?
        - 문장이 불완전하다면 어떤 정보가 생략되었는가?

        3️⃣ 통합 추론
        - 수어와 음성이 서로 보완하는가, 아니면 하나가 핵심인가?
        - 사용자의 최종 목적은 무엇인가?
        - 사용자가 **무엇을 원하고**, **누구에게**, **어떤 행동을 요청**하는가?

        4️⃣ 방향 및 환경 추론
        - 손짓, 지시 수어, 음성 맥락을 바탕으로 방향(왼쪽/오른쪽/앞/뒤/위/아래)을 추론
        - 가능하다면 주변 환경(사물, 장소, 상황)을 자연스럽게 포함

        5️⃣ 최종 해석 생성
        아래 형식을 반드시 따르세요:

        "사용자는 (수어 인식 결과)를 표현하며 (음성 인식 결과)라고 말하고 있습니다.
        또한 (방향/환경에 대한 추론)을 고려할 때,
        사용자의 의도는 (통합된 의도 문장)입니다."

        """

        # 3. LLM 호출
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # 또는 다른 모델 사용 가능
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print_error(f"❌ 오류: LLM API 호출 실패. API 키 또는 인터넷 연결 확인 필요. 에러: {e}")
            return "❌ LLM 호출 실패 (API 오류)"


def main_run(model_path, proto_path, video_file, audio_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print_error(f"🚀 멀티모달 에이전트 초기화 중... (Device: {device})")
    try:
        agent = MultimodalAgent(model_path, proto_path, device)
    except Exception as e:
        # 🚨 수정: 구체적인 오류 메시지(e)를 출력하도록 변경
        print_error(f"❌ 에이전트 초기화 실패로 종료합니다. 상세 오류: {e}")
        return

    # 1. 비디오 텐서 로드
    print_error(f"📂 비디오 텐서 로딩 중: {video_file}")
    try:
        video_tensor = torch.load(video_file, map_location=device).float()
    except Exception as e:
        print_error(f"❌ 비디오 텐서 로드 실패. 파일이 손상되었을 수 있습니다. 에러: {e}")
        return
    
    # 🚨🚨 수정: [T, C, H, W] 형태를 [1, C, T, H, W] 형태로 변환 (C와 T 위치 변경)
    if video_tensor.dim() == 4:
        # 현재 형태가 (T, C, H, W)라면 -> (C, T, H, W)로 permute
        # 캡처된 텐서의 차원이 (89, 3, 224, 224)라고 가정
        if video_tensor.size(1) == 3:
             # 형태가 이미 (T, C, H, W)라면, C와 T를 바꿔 (C, T, H, W)로 만듭니다.
             video_tensor = video_tensor.permute(1, 0, 2, 3) 
        
        # 최종적으로 Batch 차원 (1)을 추가하여 (1, C, T, H, W)로 만듭니다.
        video_tensor = video_tensor.unsqueeze(0) 

    # 2. 멀티모달 추론 및 응답 생성
    print_error("🧠 LLM 응답 생성 시작...")
    
    llm_response = agent.generate_response(video_tensor, audio_file)
    
    if "❌ LLM 호출 실패" not in llm_response:
        # 성공 시: Electron이 구독하는 stdout으로 결과 전달
        print(f"LLM_RESPONSE::{llm_response.strip()}")
        sys.stdout.flush()
    else:
        # 실패 시: stderr로 알림
        print_error("❌ 최종 출력 실패.")
        
    print_error("="*50 + "\n")
    return llm_response


if __name__ == '__main__':
    # ⚠️ 캡처 파일 경로가 motionCapture.py와 일치하는지 확인
    MODEL_PATH = "slip_protonet_final.pth" 
    PROTO_PATH = "prototypes.pt"
    VIDEO_FILE = "captured_video.pt" 
    AUDIO_FILE = "captured_audio.wav"
    
    if os.path.exists(VIDEO_FILE) and os.path.exists(AUDIO_FILE):
        main_run(MODEL_PATH, PROTO_PATH, VIDEO_FILE, AUDIO_FILE)
    else:
        print_error(f"⚠️ {VIDEO_FILE} 또는 {AUDIO_FILE} 파일이 존재하지 않습니다.")
        print_error("motionCapture.py를 먼저 실행하여 수어 동작과 음성을 캡처해주세요.")
