const { contextBridge, ipcRenderer } = require('electron');

// Preload 로드 여부 확인용 로그
console.log('[Preload] loaded');

// 모든 llm-response를 렌더러로 window.postMessage로도 전달 (fallback)
ipcRenderer.on('llm-response', (_e, msg) => {
  window.postMessage({ type: 'llm-response', payload: msg }, '*');
});

contextBridge.exposeInMainWorld('electronAPI', {
  startPython: () => ipcRenderer.send('start-python'),
  getWeather: () => ipcRenderer.invoke("get-weather"),
  loadDefaultGestures: () => ipcRenderer.invoke('load-default-gestures'),
  loadCustomGestures: () => ipcRenderer.invoke('load-custom-gestures'),
  saveCustomGestures: (gestures) => ipcRenderer.send('save-custom-gestures', gestures),
  startWebcam: () => ipcRenderer.send('start-webcam-script'),
  trainLSTM: () => ipcRenderer.send('train-LSTM-script'),
  sendCustomGestureData: (data) => ipcRenderer.send("custom-gesture-data", data),
  onLLMResponse: (cb) => {
    const handler = (_e, msg) => {
      console.log('[Preload] llm-response event:', msg);
      cb(msg);
    };
    ipcRenderer.on('llm-response', handler);
    return () => ipcRenderer.removeListener('llm-response', handler);
  }
});
