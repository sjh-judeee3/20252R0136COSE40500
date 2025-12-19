import React, { useEffect, useState , useMemo } from "react";

//import customGestures from '../../electron/motionCapture/data/customData.json';

export default function GesturePanel() {
  const [customGestures, setCustomGestures] = useState([]);
  const [defaultGestures, setDefaultGestures] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [gestureName, setGestureName] = useState("");
  const [actionType, setActionType] = useState("shortcut"); // 'shortcut' or 'url'
  const [shortcutKeys, setShortcutKeys] = useState([]); // ['alt', 'tab']
  const [isListening, setIsListening] = useState(false);
  const [url, setUrl] = useState("");


  useEffect(() => {
    Promise.all([
      window.electronAPI.loadDefaultGestures(),
      window.electronAPI.loadCustomGestures()
    ]).then(([defaultGs, customGs]) => {
      // 각각 상태에 저장하거나 합쳐서 UI 렌더
      setDefaultGestures(defaultGs);
      setCustomGestures(customGs);
    });
  }, []);


  // 기본 + 커스텀 합친 리스트 (렌더링 최적화)
  const gestures = useMemo(() => {
    return [
      ...defaultGestures.map(g => ({ ...g, isCustom: false })),
      ...customGestures.map(g => ({ ...g, isCustom: true }))
    ];
  }, [customGestures]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isListening) return;

      e.preventDefault(); // 창 이동 방지

      const key = e.key.toLowerCase();
      if (!shortcutKeys.includes(key)) {
        setShortcutKeys((prev) => [...prev, key]);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isListening, shortcutKeys]);

  const handleAddGesture = () => {
    setShowModal(true); // 팝업 열기
  };

  const handleSaveGesture = () => {
    window.electronAPI.startWebcam();
    const mappedShortcut = actionType === "shortcut" ? shortcutKeys : url;
    const newGesture = {
        name: gestureName,
        type: actionType,
        value: mappedShortcut,
        isCustom: true,
    };

    const updatedCustom = [...customGestures, newGesture];
    setCustomGestures(updatedCustom);
    window.electronAPI.saveCustomGestures(updatedCustom); // 커스텀만 저장

    // Python에 새 제스처 전송
    window.electronAPI.sendCustomGestureData(newGesture);

    // 초기화
    setGestureName("");
    setShortcutKeys([]);
    setUrl("");
    setShowModal(false);
    setIsListening(false);
    setActionType("shortcut");
  };

  const handleCancelGesture = () => {
    setGestureName("");
    setShortcutKeys([]);
    setUrl("");
    setShowModal(false);
    setIsListening(false);
    setActionType("shortcut");
  };

  const handleDeleteGesture = (index) => {
    // gestures 배열 내 인덱스가 아닌 커스텀 제스처 배열 인덱스를 받아야 함
    const updated = customGestures.filter((_, i) => i !== index);
    setCustomGestures(updated);
    window.electronAPI.saveCustomGestures(updated);
  };

  const startShortcutCapture = () => {
    setShortcutKeys([]);
    setIsListening(true);
  };

  const stopShortcutCapture = () => {
    setIsListening(false);
  };

return (
  <div className="glass" style={{ width: 400, padding: 10 }}>
    <h3>My Gestures</h3>

    {/* 리스트 영역: 최대 5개 정도 보이도록 고정 높이 & 스크롤 */}
    <div
      style={{
        maxHeight: 'calc(5 * 80px)', // 아이템 5개 * 대략 80px 높이 (조절 가능)
        overflowY: 'auto',
        marginBottom: 10, // 버튼과 간격
      }}
    >
      {gestures.map((g, i) => (
        <div
          key={i}
          className="glass"
          style={{ position: 'relative', padding: 10, marginTop: 10 ,width: "80%"}}
        >
          {g.isCustom && (
            <button
              onClick={() => {
                const customIndex = customGestures.findIndex(
                  cg => cg.name === g.name && cg.type === g.type
                );
                if (customIndex >= 0) handleDeleteGesture(customIndex);
              }}
              style={{
                position: 'absolute',
                top: 5,
                right: 5,
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: '#fff',
              }}
              title="Delete gesture"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}

          <p>
            Gesture: <strong>{g.name}</strong>
          </p>
          <p>
            Activate:{' '}
            {g.type === 'shortcut'
              ? g.value.map(k => k.toUpperCase()).join(' + ')
              : g.value}
          </p>
        </div>
      ))}
    </div>

    {/* Add Gesture 버튼은 항상 보여야 하므로 리스트 바깥에 위치 */}
    <div
      style={{
        display: "flex",
        justifyContent: "space-between", // 버튼 사이 간격 확보
        gap: 10,                         // 버튼 사이 여백
        marginTop: 20,
      }}
    >
      <button
        onClick={handleAddGesture}
        style={{
          flex: 1,
          borderRadius: 10,
          background: "rgba(255,255,255,0.2)",
          border: "none",
          color: "white",
          padding: 10,
        }}
      >
        Add Gesture
      </button>
      <button
        onClick={() => window.electronAPI.trainLSTM()} // <- 필요시 이 핸들러도 정의되어 있어야 함
        style={{
          flex: 1,
          borderRadius: 10,
          background: "rgba(99, 142, 169, 0.2)",
          border: "none",
          color: "white",
          padding: 10,
        }}
      >
        Train Model
      </button>
    </div>

    

      {/* 모달 */}
      {showModal && (
        <>
          {/* 배경 오버레이 */}
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              background: "rgba(0,0,0,0.4)",
              zIndex: 998,
            }}
          />

          {/* 모달 본문 */}
            <div
              className="glass"
              style={{
                position: "fixed",
                top: "80%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "320px",
                padding: "20px",
                borderRadius: "12px",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                zIndex: 999,
                cursor: "move", // 드래그 가능한 커서
              }}
            >
              <h3>Add New Gesture</h3>
              <input
                type="text"
                placeholder="Gesture name"
                value={gestureName}
                onChange={(e) => setGestureName(e.target.value)}
                style={{ padding: "8px", borderRadius: "6px" }}
              />

              <label>
                Choose Action Type:{" "}
                <select
                  value={actionType}
                  onChange={(e) => setActionType(e.target.value)}
                  style={{ marginLeft: 10 }}
                >
                  <option value="shortcut">Shortcut</option>
                  <option value="url">URL</option>
                </select>
              </label>

              {/* 조건부 렌더링 */}
              {actionType === "shortcut" && (
                <div style={{ marginTop: 10 }}>
                  <button
                    onClick={startShortcutCapture}
                    disabled={isListening}
                    style={{ padding: "8px", borderRadius: "6px" }}
                  >
                    {isListening ? "Listening..." : "Capture Shortcut"}
                  </button>

                  {isListening && (
                    <button
                      onClick={stopShortcutCapture}
                      style={{
                        marginLeft: "10px",
                        padding: "8px",
                        borderRadius: "6px",
                      }}
                    >
                      Done
                    </button>
                  )}

                  <p style={{ marginTop: 10 }}>
                    Captured:{" "}
                    {shortcutKeys.length > 0
                      ? shortcutKeys.map((k) => k.toUpperCase()).join(" + ")
                      : "None"}
                  </p>
                </div>
              )}

              {actionType === "url" && (
                <div style={{ marginTop: 10 }}>
                  <input
                    type="text"
                    placeholder="Enter URL"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    style={{ padding: "8px", borderRadius: "6px", width: "100%" }}
                  />
                </div>
              )}

              <h3>Instructions: How to input data</h3>
              <ul>
                <li>The camera records you every 4 seconds.</li>
                <li>
                  Please follow the on-screen instructions carefully during each
                  recording session.
                </li>
                <li>(slowly / far away / closer / quickly)</li>
                <li>Complete all steps to ensure accurate gesture capture.</li>
              </ul>

              <div style={{ display: "flex", gap: "10px" }}>
                <button
                  onClick={handleSaveGesture}
                  style={{
                    background: "#c5e8c5ff",
                    color: "#fff",
                    padding: "10px",
                    border: "none",
                    borderRadius: "6px",
                    flex: 1,
                  }}
                >
                  Save
                </button>
                <button
                  onClick={handleCancelGesture}
                  style={{
                    background: "#f1aaa5ff",
                    color: "#fff",
                    padding: "10px",
                    border: "none",
                    borderRadius: "6px",
                    flex: 1,
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
        </>
      )}
    </div>
  );
}
