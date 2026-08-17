/* -------------------------------------------------------------
 * ISL AI Studio — Real-Time Streaming & UI Controller
 * ------------------------------------------------------------- */

(function () {
  // DOM Elements
  const videoEl = document.getElementById("webcamVideo");
  const canvasEl = document.getElementById("skeletonCanvas");
  const ctx = canvasEl.getContext("2d");
  const toggleCameraBtn = document.getElementById("toggleCameraBtn");
  const toggleSkeletonBtn = document.getElementById("toggleSkeletonBtn");
  const viewportOverlay = document.getElementById("viewportOverlay");

  const systemStatusPill = document.getElementById("systemStatusPill");
  const systemStatusText = document.getElementById("systemStatusText");
  const fpsDisplay = document.getElementById("fpsDisplay");
  const streamStateBadge = document.getElementById("streamStateBadge");
  const bufferFillBar = document.getElementById("bufferFillBar");
  const bufferProgressText = document.getElementById("bufferProgressText");

  const predictedSignText = document.getElementById("predictedSignText");
  const signStatusSub = document.getElementById("signStatusSub");
  const confidenceTag = document.getElementById("confidenceTag");
  const candidateBars = document.getElementById("candidateBars");

  const languageSelect = document.getElementById("languageSelect");
  const translationOutputText = document.getElementById("translationOutputText");
  const speakButton = document.getElementById("speakButton");
  const agentList = document.getElementById("agentList");

  // State
  let isCameraActive = false;
  let isSkeletonVisible = true;
  let websocket = null;
  let holistic = null;
  let camera = null;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 30;
  let lastPredictedSign = "";

  // 1. Initialize WebSocket Connection
  function initWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream`;
    
    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      systemStatusText.textContent = "Backend Connected";
      systemStatusPill.querySelector(".pulse-dot").style.backgroundColor = "var(--accent-emerald)";
    };

    websocket.onmessage = (event) => {
      const packet = JSON.parse(event.data);
      handleServerPacket(packet);
    };

    websocket.onclose = () => {
      systemStatusText.textContent = "Reconnecting...";
      systemStatusPill.querySelector(".pulse-dot").style.backgroundColor = "var(--accent-rose)";
      setTimeout(initWebSocket, 2000);
    };

    websocket.onerror = () => {
      systemStatusText.textContent = "Connection Error";
      systemStatusPill.querySelector(".pulse-dot").style.backgroundColor = "var(--accent-rose)";
    };
  }

  // 2. Handle Server Inference Response
  function handleServerPacket(packet) {
    const state = packet.state || "IDLE";
    streamStateBadge.textContent = state;
    streamStateBadge.className = `state-badge ${state.toLowerCase()}`;

    // Buffer Meter
    const fillRatio = packet.buffer_fill_ratio || 0.0;
    const currentFrames = packet.current_frames || 0;
    const targetFrames = packet.target_frames || 45;
    bufferFillBar.style.width = `${Math.min(100, Math.round(fillRatio * 100))}%`;
    bufferProgressText.textContent = `${currentFrames} / ${targetFrames} frames (${Math.round(fillRatio * 100)}%)`;

    // Telemetry
    const inferLatency = packet.inference_latency_ms || 0.0;
    fpsDisplay.textContent = `FPS: ${currentFps.toFixed(0)} | ${inferLatency.toFixed(1)} ms`;

    // Prediction
    const pred = packet.prediction;
    if (pred && pred.class_name && (state === "PREDICTED" || fillRatio > 0.8)) {
      const sign = pred.class_name;
      const conf = (pred.confidence * 100).toFixed(1);
      
      predictedSignText.textContent = sign;
      confidenceTag.textContent = `${conf}%`;
      signStatusSub.textContent = `Recognized with ${conf}% confidence`;

      // Trigger automatic TTS if new sign
      if (sign !== lastPredictedSign && state === "PREDICTED") {
        lastPredictedSign = sign;
        if (packet.translation && packet.translation.translated_text) {
          translationOutputText.textContent = packet.translation.translated_text;
          speakText(packet.translation.translated_text, languageSelect.value);
        }
      }
    } else if (state === "IDLE") {
      predictedSignText.textContent = "WAITING FOR SIGN";
      confidenceTag.textContent = "-- %";
      signStatusSub.textContent = "Position hands in camera view";
    }

    if (packet.translation && packet.translation.translated_text) {
      translationOutputText.textContent = packet.translation.translated_text;
    }
  }

  // 3. Web Speech API (TTS)
  function speakText(text, langCode) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const langMap = {
      "hin_Deva": "hi-IN",
      "tam_Taml": "ta-IN",
      "tel_Telu": "te-IN",
      "ben_Beng": "bn-IN",
      "mar_Mrai": "mr-IN",
      "eng_Latn": "en-IN",
    };
    utterance.lang = langMap[langCode] || "hi-IN";
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
  }

  speakButton.addEventListener("click", () => {
    speakText(translationOutputText.textContent, languageSelect.value);
  });

  // 4. MediaPipe Holistic Ingestion & Skeleton Rendering
  function onHolisticResults(results) {
    // Update FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      currentFps = frameCount * 1000 / (now - lastFrameTime);
      frameCount = 0;
      lastFrameTime = now;
    }

    canvasEl.width = videoEl.videoWidth || 1280;
    canvasEl.height = videoEl.videoHeight || 720;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    // Extract 76 Keypoint Landmarks:
    // Left Hand (21), Right Hand (21), Upper Pose (11), Facial NMMs (23) = 76 points
    const landmarks76 = new Array(76).fill(null).map(() => [0, 0, 0]);

    let hasHands = false;

    // Left Hand
    if (results.leftHandLandmarks) {
      hasHands = true;
      results.leftHandLandmarks.forEach((pt, i) => {
        if (i < 21) landmarks76[i] = [pt.x, pt.y, pt.z || 0];
      });
      if (isSkeletonVisible) drawHandLandmarks(results.leftHandLandmarks, "#00e5ff");
    }

    // Right Hand
    if (results.rightHandLandmarks) {
      hasHands = true;
      results.rightHandLandmarks.forEach((pt, i) => {
        if (i < 21) landmarks76[21 + i] = [pt.x, pt.y, pt.z || 0];
      });
      if (isSkeletonVisible) drawHandLandmarks(results.rightHandLandmarks, "#00e5ff");
    }

    // Upper Pose
    if (results.poseLandmarks) {
      const poseIndices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 1, 4];
      poseIndices.forEach((srcIdx, dstIdx) => {
        const pt = results.poseLandmarks[srcIdx];
        if (pt) landmarks76[42 + dstIdx] = [pt.x, pt.y, pt.z || 0];
      });
      if (isSkeletonVisible) drawPoseLandmarks(results.poseLandmarks, "#ffd600");
    }

    // Facial Non-Manual Markers (Eyebrows, Eyes, Lips)
    if (results.faceLandmarks) {
      const faceIndices = [
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        33, 133, 362, 263,
        61, 185, 40, 39, 37, 0, 267, 269, 270
      ];
      faceIndices.forEach((srcIdx, dstIdx) => {
        const pt = results.faceLandmarks[srcIdx];
        if (pt) landmarks76[53 + dstIdx] = [pt.x, pt.y, pt.z || 0];
      });
      if (isSkeletonVisible) drawFaceMarkers(results.faceLandmarks, "#00e676");
    }

    // Stream Landmark Array to Python Backend via WebSocket
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({
        landmarks: hasHands ? landmarks76 : null,
        target_lang: languageSelect.value,
      }));
    }
  }

  // Helper Drawing Functions
  function drawHandLandmarks(landmarks, color) {
    ctx.fillStyle = color;
    landmarks.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x * canvasEl.width, pt.y * canvasEl.height, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  function drawPoseLandmarks(landmarks, color) {
    ctx.fillStyle = color;
    [11, 12, 13, 14, 15, 16].forEach(idx => {
      const pt = landmarks[idx];
      if (pt) {
        ctx.beginPath();
        ctx.arc(pt.x * canvasEl.width, pt.y * canvasEl.height, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }

  function drawFaceMarkers(landmarks, color) {
    ctx.fillStyle = color;
    [70, 63, 105, 336, 296, 33, 133, 362, 61, 0, 291].forEach(idx => {
      const pt = landmarks[idx];
      if (pt) {
        ctx.beginPath();
        ctx.arc(pt.x * canvasEl.width, pt.y * canvasEl.height, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }

  // 5. Camera Activation & Controls
  async function startCamera() {
    try {
      holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        refineFaceLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      holistic.onResults(onHolisticResults);

      camera = new Camera(videoEl, {
        onFrame: async () => {
          if (isCameraActive && holistic) {
            await holistic.send({ image: videoEl });
          }
        },
        width: 1280,
        height: 720,
      });

      await camera.start();
      isCameraActive = true;
      viewportOverlay.classList.add("hidden");
      toggleCameraBtn.textContent = "Stop Camera";
      toggleCameraBtn.className = "btn btn-secondary";
    } catch (err) {
      alert(`Camera Access Error: ${err.message}`);
    }
  }

  function stopCamera() {
    if (camera) {
      camera.stop();
      camera = null;
    }
    isCameraActive = false;
    viewportOverlay.classList.remove("hidden");
    toggleCameraBtn.textContent = "Start Camera";
    toggleCameraBtn.className = "btn btn-primary";
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }

  toggleCameraBtn.addEventListener("click", () => {
    if (isCameraActive) stopCamera();
    else startCamera();
  });

  toggleSkeletonBtn.addEventListener("click", () => {
    isSkeletonVisible = !isSkeletonVisible;
    toggleSkeletonBtn.textContent = `Skeleton: ${isSkeletonVisible ? "ON" : "OFF"}`;
    if (!isSkeletonVisible) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  });

  // 6. Fetch 9-Agent Telemetry List
  async function loadAgentTelemetry() {
    try {
      const res = await fetch("/api/agents");
      const data = await res.json();
      agentList.innerHTML = "";
      data.agents.forEach(agent => {
        const card = document.createElement("div");
        card.className = "agent-card";
        card.innerHTML = `
          <div class="agent-info">
            <span class="agent-role">${agent.id}. ${agent.role}</span>
            <span class="agent-model">${agent.model} • ${agent.task}</span>
          </div>
          <span class="agent-status-badge">${agent.status}</span>
        `;
        agentList.appendChild(card);
      });
    } catch (err) {
      console.warn("Telemetry fetch error:", err);
    }
  }

  // Init
  initWebSocket();
  loadAgentTelemetry();
})();
