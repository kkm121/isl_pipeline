document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("comparisonSlider");
  const afterWrapper = document.getElementById("afterWrapper");
  const sliderHandle = document.getElementById("sliderHandle");
  const cloudSlider = document.getElementById("cloudSlider");
  const cloudVal = document.getElementById("cloudVal");
  const aoiSelector = document.getElementById("aoiSelector");
  const aoiDescription = document.getElementById("aoiDescription");
  const layerUncertainty = document.getElementById("layerUncertainty");
  const uncertaintyOverlay = document.getElementById("uncertaintyOverlay");
  const runSrBtn = document.getElementById("runSrBtn");

  const aoiDescriptions = {
    indo_gangetic: "Smallholder agriculture, Kharif/Rabi rotation",
    western_ghats: "Steep canopy, persistent monsoon cloud cover",
    peri_urban: "Rapid built-up expansion, informal settlements",
    rajasthan: "Unpaved rural tracks, high bare-soil reflectance",
  };

  // Slider dragging logic
  let isDragging = false;

  function setSliderPosition(x) {
    const rect = container.getBoundingClientRect();
    let pos = (x - rect.left) / rect.width;
    pos = Math.max(0.05, Math.min(0.95, pos));
    const pct = pos * 100;
    afterWrapper.style.width = `${pct}%`;
    sliderHandle.style.left = `${pct}%`;
  }

  container.addEventListener("mousedown", (e) => {
    isDragging = true;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Touch support
  container.addEventListener("touchstart", (e) => {
    isDragging = true;
    setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isDragging = false;
  });

  // Cloud slider value update
  cloudSlider.addEventListener("input", (e) => {
    cloudVal.textContent = `${e.target.value}%`;
  });

  // AOI selector update
  aoiSelector.addEventListener("change", (e) => {
    aoiDescription.textContent = aoiDescriptions[e.target.value] || "";
  });

  // Uncertainty layer toggle
  layerUncertainty.addEventListener("change", (e) => {
    if (e.target.checked) {
      uncertaintyOverlay.classList.add("active");
    } else {
      uncertaintyOverlay.classList.remove("active");
    }
  });

  // Run Super-Resolution action
  runSrBtn.addEventListener("click", async () => {
    runSrBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing BharatSRM-Net...';
    runSrBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("aoi_id", aoiSelector.value);
      formData.append("cloud_threshold", (cloudSlider.value / 100).toFixed(2));
      formData.append("enable_context_dem", document.getElementById("contextDemToggle").checked);

      const res = await fetch("/api/super_resolve", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.status === "success") {
        document.getElementById("metricPsnr").textContent = `${data.metrics.PSNR_dB} dB`;
        document.getElementById("metricSsim").textContent = `${data.metrics.SSIM}`;
        document.getElementById("metricSam").textContent = `${data.metrics.SAM_deg}°`;
        document.getElementById("metricErgas").textContent = `${data.metrics.ERGAS}`;
        document.getElementById("metricDegrade").textContent = `${data.metrics.L_degrade}`;
      }
    } catch (err) {
      console.error("Inference Error:", err);
    } finally {
      runSrBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run BharatSRM-Net v4';
      runSrBtn.disabled = false;
    }
  });
});
