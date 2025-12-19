import React, { useRef, useEffect, useState } from "react";
import { Hands , HAND_CONNECTIONS } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";


// landmarks: array of {x, y, z} (landmarks.landmark에 해당)

function getFingerStatus(landmarks) {
  // landmarks는 array임 (landmarks.landmark와 동일)
  // 손가락 구조:
  // thumb: tip=4, pip=2 (x축 비교)
  // others: tip, pip y축 비교
  const fingers = [
    { name: "thumb", tip: 4, pip: 2 },
    { name: "index", tip: 8, pip: 6 },
    { name: "middle", tip: 12, pip: 10 },
    { name: "ring", tip: 16, pip: 14 },
    { name: "pinky", tip: 20, pip: 18 },
  ];

  const fingerStatus = [];

  for (const finger of fingers) {
    if (finger.name === "thumb") {
      const tipX = landmarks[finger.tip].x;
      const pipX = landmarks[finger.pip].x;
      fingerStatus.push(tipX < pipX);
    } else {
      const tipY = landmarks[finger.tip].y;
      const pipY = landmarks[finger.pip].y;
      fingerStatus.push(tipY < pipY);
    }
  }

  return fingerStatus; // [thumb, index, middle, ring, pinky] boolean 배열
}

function recognizeGesture(fingerStatus) {
  if (!fingerStatus || fingerStatus.length !== 5) return null;
  const [thumb, index, middle, ring, pinky] = fingerStatus;

  if (index && middle && !ring && !pinky) {
    return "scissors";
  } else if (!index && !middle && !ring && !pinky) {
    return "rock";
  } else if (index && middle && ring && pinky) {
    return "paper";
  } else {
    return null;
  }
}

export default function HandGesture() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [gesture, setGesture] = useState("None");

  useEffect(() => {
    const hands = new Hands({
        locateFile: (file) => {
            return `/mediapipe/hands/${file}`;
        },
    });

    console.log("Initializing Mediapipe Hands...");

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });

    hands.onResults((results) => {
        const canvasCtx = canvasRef.current.getContext("2d");
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        if (results.image) {
            canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);
        }

        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, Hands.HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 5,
            });
            drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });

            const fingerStatus = getFingerStatus(landmarks);
            const recognizedGesture = recognizeGesture(fingerStatus);
            setGesture(recognizedGesture);
            }
        } else {
            setGesture("None");
        }

        canvasCtx.restore();
    });


    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 800,
      height: 600,
    });
    console.log("videoRef.current:", videoRef.current);

    camera.start();

    return () => {
      camera.stop();
    };
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "30px", color: "#0ff" }}>
      <div className="glass">
        <video
          ref={videoRef}
          style={{
            position: "absolute",
            width: "640px",
            height: "480px",
            zIndex: 0,
        }}
        autoPlay
        muted
        />
        <canvas
          ref={canvasRef}
          width="800"
          height="600"
          style={{ borderRadius: "20px" }}
        />
        <div
          style={{
            marginTop: "1rem",
            fontSize: "2rem",
            fontWeight: "bold",
            color: "#ffffffff",
            textShadow: "0 0 10px #aeb7ffff",
            fontFamily: "'Orbitron', sans-serif",
          }}
        >
          Gesture: {gesture}
        </div>
      </div>
    </div>
  );
}
