document.addEventListener("DOMContentLoaded", () => {
  const splitContainer = document.getElementById("splitContainer");
  const splitBefore = document.getElementById("splitBefore");
  const sliderDivider = document.getElementById("sliderDivider");
  const sliderHandle = document.getElementById("sliderHandle");
  const fileInput = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const uploadPrompt = document.getElementById("uploadPrompt");
  const runBtn = document.getElementById("runBtn");
  const cloudSlider = document.getElementById("cloudSlider");
  const cloudLabel = document.getElementById("cloudLabel");
  const demCheck = document.getElementById("demCheck");
  const rightTag = document.getElementById("rightTag");

  const imgLeft = document.getElementById("imgLeft");
  const imgRight = document.getElementById("imgRight");

  const valPsnr = document.getElementById("valPsnr");
  const valSsim = document.getElementById("valSsim");
  const valSam = document.getElementById("valSam");
  const valUnc = document.getElementById("valUnc");
  const valLat = document.getElementById("valLat");

  let currentAoi = "user_lake";
  let selectedFile = null;
  let currentResults = null;
  let activeTab = "sr";

  // 1. Pixel-Perfect Slider Drag Logic via CSS Clip-Path Inset
  let isDragging = false;
  function updateSlider(clientX) {
    const rect = splitContainer.getBoundingClientRect();
    let pos = (clientX - rect.left) / rect.width;
    pos = Math.max(0.01, Math.min(0.99, pos));
    const pct = pos * 100;
    
    // Lock layers in identical spatial coordinate space
    splitBefore.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    sliderDivider.style.left = `${pct}%`;
    sliderHandle.style.left = `${pct}%`;
  }
  
  splitContainer.addEventListener("mousedown", (e) => { isDragging = true; updateSlider(e.clientX); });
  window.addEventListener("mousemove", (e) => { if (isDragging) updateSlider(e.clientX); });
  window.addEventListener("mouseup", () => { isDragging = false; });

  splitContainer.addEventListener("touchstart", (e) => { isDragging = true; updateSlider(e.touches[0].clientX); });
  window.addEventListener("touchmove", (e) => { if (isDragging) updateSlider(e.touches[0].clientX); });
  window.addEventListener("touchend", () => { isDragging = false; });

  cloudSlider.addEventListener("input", (e) => { cloudLabel.textContent = `${e.target.value}%`; });

  // 2. Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentAoi = btn.dataset.id;
      selectedFile = null;
      uploadPrompt.innerHTML = `<strong>Click to Browse Image</strong> or Drag & Drop`;
      triggerInference();
    });
  });

  // 3. File Selection & Drag-and-Drop
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedFile = e.target.files[0];
      onFileSelected(selectedFile);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#38bdf8";
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "#0284c7";
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#0284c7";
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      selectedFile = e.dataTransfer.files[0];
      onFileSelected(selectedFile);
    }
  });

  function onFileSelected(file) {
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    uploadPrompt.innerHTML = `Loaded: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
    triggerInference();
  }

  // 4. Tab Switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.view;
      renderCurrentView();
    });
  });

  const lulcLegend = document.getElementById("lulcLegend");

  function renderCurrentView() {
    if (!currentResults) return;

    if (lulcLegend) {
      lulcLegend.style.display = (activeTab === "lulc") ? "flex" : "none";
    }

    if (activeTab === "sr") {
      imgRight.src = currentResults.sr_image_b64;
      rightTag.textContent = "2.5m Super-Resolved (BharatSRM)";
      rightTag.style.color = "#38bdf8";
    } else if (activeTab === "uncertainty") {
      imgRight.src = currentResults.uncertainty_b64;
      rightTag.textContent = "Calibrated Uncertainty Heatmap (Turbo)";
      rightTag.style.color = "#fb923c";
    } else if (activeTab === "roads") {
      imgRight.src = currentResults.road_overlay_b64;
      rightTag.textContent = "PMGSY Rural Road Extraction Overlay";
      rightTag.style.color = "#ef4444";
    } else if (activeTab === "lulc") {
      imgRight.src = currentResults.lulc_map_b64;
      rightTag.textContent = "ISRO 5-Class LULC Disaggregation";
      rightTag.style.color = "#4ade80";
    }
  }

  // 5. Run Super-Resolution
  async function triggerInference() {
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Full-Res 4x SR...';
    runBtn.disabled = true;

    try {
      const fd = new FormData();
      fd.append("aoi_id", currentAoi);
      fd.append("cloud_threshold", (cloudSlider.value / 100).toFixed(2));
      fd.append("enable_context_dem", demCheck.checked);
      if (selectedFile) {
        fd.append("file", selectedFile);
      }

      const res = await fetch("/api/super_resolve", { method: "POST", body: fd });
      const data = await res.json();

      if (data.status === "success") {
        currentResults = data;
        imgLeft.src = data.lr_image_b64;
        renderCurrentView();

        valPsnr.textContent = `${data.metrics.PSNR_dB.toFixed(2)} dB`;
        valSsim.textContent = `${data.metrics.SSIM.toFixed(3)}`;
        valSam.textContent = `${data.metrics.SAM_deg.toFixed(2)}°`;
        valUnc.textContent = `${data.mean_uncertainty.toFixed(4)}`;
        valLat.textContent = `${data.latency_ms} ms`;
      } else {
        alert("Inference Error: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      console.error("Super-Resolution Error:", err);
    } finally {
      runBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Super-Resolution (4&times;)';
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener("click", triggerInference);

  // 6. Download
  document.getElementById("downloadBtn").addEventListener("click", () => {
    if (!currentResults) return;
    const a = document.createElement("a");
    a.href = imgRight.src;
    a.download = `bharatsrm_v4_${activeTab}_output.png`;
    a.click();
  });

  // Run on load
  triggerInference();
});
