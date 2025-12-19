import cv2
import numpy as np
import torch
import time
import os
import sounddevice as sd
from scipy.io.wavfile import write
import sys 
import json

# run_multimodal에서 main_run 함수를 가져옵니다.
# 주의: 이 파일과 run_multimodal.py는 같은 디렉토리에 있어야 합니다.
from run_multimodal import main_run 

# --- 설정값 ---
FPS = 30
CAPTURE_DURATION = 3.0  # 3초 동안 수어 동작 캡처
OUTPUT_VIDEO_PATH = "captured_video.pt"  # PyTorch 텐서 파일
OUTPUT_AUDIO_PATH = "captured_audio.wav" # 오디오 파일

# ⚠️ 사용자의 최종 학습된 모델 및 프로토타입 경로로 변경하세요.
MODEL_PATH = "../model/slip_protonet_final.pth" 
PROTO_PATH = "prototypes.pt"

def print_status(message):
    """
    모든 상태 메시지(진행 상황, 오류)를 sys.stderr로 출력하여 
    stdout(최종 통역 결과)과 분리하고 Electron의 제스처 오작동을 방지합니다.
    """
    print(message, file=sys.stderr)


def record_audio(filename, duration, samplerate=44100):
    """음성을 녹음하여 WAV 파일로 저장"""
    print_status(f"\n🎤 {duration}초 동안 음성 녹음 시작...")
    # 원하는 입력 장치가 있다면 MIC_DEVICE 환경변수에 index(숫자)나 이름을 넣어 사용
    mic_device = os.environ.get("MIC_DEVICE")
    if mic_device:
        try:
            mic_device = int(mic_device)
        except ValueError:
            pass  # 문자열 이름 그대로 사용
        print_status(f"🎙️ 입력 장치 지정: {mic_device}")

    try:
        # 녹음 시작
        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            device=mic_device  # None이면 기본 입력 장치 사용
        )
        sd.wait()  # 녹음이 끝날 때까지 대기

        # 입력 신호가 없는 경우(모두 0) 바로 알려줌
        mean_amp = float(np.abs(recording).mean())
        if mean_amp < 1.0:
            print_status("⚠️ 녹음된 오디오가 비어있습니다. 마이크 권한/입력 장치 설정을 확인하세요.")
            return False

        write(filename, samplerate, recording)
        print_status(f"✅ 음성 녹음 완료: {filename}")
        return True
    except Exception as e:
        print_status(f"❌ 음성 녹음 실패 (마이크 설정 및 'sounddevice' 권한 확인 필요): {e}")
        return False


def main_capture():
    
    # 1. 오디오 녹음을 시작 (비디오 캡처와 순차적으로 진행)
    # 오디오 실패 시 바로 종료하여 비디오 자원 낭비 방지
    audio_success = record_audio(OUTPUT_AUDIO_PATH, CAPTURE_DURATION)
    if not audio_success:
        return None, None 

    # 2. 카메라 초기화 및 비디오 캡처
    print_status("🎥 카메라 초기화 중...")
    cap = cv2.VideoCapture(0) # 0번 카메라 장치 시도
    
    if not cap.isOpened():
        print_status("❌ 카메라를 열 수 없습니다. 장치 연결 및 권한 확인 필요.")
        return None, None
        
    TARGET_SIZE = (224, 224) 
    frames = []
    start_time = time.time()
    
    print_status(f"🎬 {CAPTURE_DURATION}초 동안 수어 동작 캡처 시작...")

    while time.time() - start_time < CAPTURE_DURATION:
        ret, frame = cap.read()
        
        if not ret: 
            # 프레임을 읽지 못하면 루프를 중단하고 오류로 간주
            print_status("⚠️ 경고: 프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
            break
        
        # 캡처된 프레임 처리 (GUI 코드 제거)
        processed_frame = cv2.flip(frame, 1)  # 좌우 반전
        processed_frame = cv2.resize(processed_frame, TARGET_SIZE)
        # BGR -> RGB 및 정규화 (0-255 -> 0-1). dtype은 이미 float32입니다.
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(processed_frame)

    cap.release()
    cv2.destroyAllWindows()
    
    if not frames:
        print_status("❌ 치명적 오류: 캡처된 프레임이 0개입니다.")
        return None, None
        
    print_status(f"✅ 비디오 캡처 완료. 총 {len(frames)} 프레임 저장됨.")


    # 3. PyTorch 텐서로 변환 및 저장
    video_np = np.stack(frames) # (T, H, W, 3)
    
    # 🚨🚨 최종 수정: dtype을 명시하고 contiguous array로 변환한 후 torch.from_numpy 호출 🚨🚨
    # 이렇게 하면 거의 모든 환경에서 호환성 문제가 해결됩니다.
    video_np_final = np.ascontiguousarray(video_np, dtype=np.float32)
    video_tensor = torch.from_numpy(video_np_final).permute(0, 3, 1, 2) # (T, 3, H, W)
    
    torch.save(video_tensor, OUTPUT_VIDEO_PATH)
    print_status(f"✅ 비디오 텐서 저장 완료: {OUTPUT_VIDEO_PATH}")

    return OUTPUT_VIDEO_PATH, OUTPUT_AUDIO_PATH

if __name__ == '__main__':
    video_file, audio_file = main_capture()
    
    # -------------------------------------------------------------
    # 🌟🌟🌟 통합 실행: 캡처 완료 후 run_multimodal의 main_run 호출 🌟🌟🌟
    # -------------------------------------------------------------
    if video_file and audio_file:
        
        # motionCapture.py가 run_multimodal.py의 main_run 함수를 직접 호출합니다.
        llm_response = main_run(
            MODEL_PATH, PROTO_PATH, video_file, audio_file
        )
        print(json.dumps({
            "type": "LLM_RESPONSE",
            "data": llm_response
        }))
    else:
        print_status("❌ 캡처 오류로 인해 추론을 시작할 수 없습니다.")
