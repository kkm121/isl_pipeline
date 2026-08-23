document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("comparisonSlider");
  const afterWrapper = document.getElementById("afterWrapper");
  const sliderHandle = document.getElementById("sliderHandle");
  const cloudSlider = document.getElementById("cloudSlider");
  const cloudVal = document.getElementById("cloudVal");
  const aoiSelector = document.getElementById("aoiSelector");
  const layerUncertainty = document.getElementById("layerUncertainty");
  const uncertaintyOverlay = document.getElementById("uncertaintyOverlay");
  const runSrBtn = document.getElementById("runSrBtn");
  const imageUpload = document.getElementById("imageUpload");
  const lrImage = document.getElementById("lrImage");
  const srImage = document.getElementById("srImage");

  // Initialize Leaflet map
  const map = L.map('map').setView([20.5937, 78.9629], 5);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
  }).addTo(map);

  const markers = {};
  fetch('/api/aois').then(r => r.json()).then(aois => {
      for (const [id, aoi] of Object.entries(aois)) {
          const marker = L.marker([aoi.lat, aoi.lon]).addTo(map);
          marker.bindPopup(`<b>${aoi.name}</b>`);
          markers[id] = marker;
      }
  });

  aoiSelector.addEventListener("change", (e) => {
      const selected = e.target.value;
      if (markers[selected]) {
          const m = markers[selected].getLatLng();
          map.flyTo(m, 13);
          markers[selected].openPopup();
      }
  });

  let isDragging = false;
  function setSliderPosition(x) {
    const rect = container.getBoundingClientRect();
    let pos = (x - rect.left) / rect.width;
    pos = Math.max(0.0, Math.min(1.0, pos));
    const pct = pos * 100;
    afterWrapper.style.width = `${pct}%`;
    sliderHandle.style.left = `${pct}%`;
  }
  container.addEventListener("mousedown", (e) => { isDragging = true; setSliderPosition(e.clientX); });
  window.addEventListener("mousemove", (e) => { if (isDragging) setSliderPosition(e.clientX); });
  window.addEventListener("mouseup", () => { isDragging = false; });

  cloudSlider.addEventListener("input", (e) => { cloudVal.textContent = `${e.target.value}%`; });

  layerUncertainty.addEventListener("change", (e) => {
    uncertaintyOverlay.style.opacity = e.target.checked ? "0.6" : "0";
  });

  let base64Image = null;
  imageUpload.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
          const reader = new FileReader();
          reader.onload = (ev) => {
              base64Image = ev.target.result;
              lrImage.src = base64Image;
              lrImage.style.display = "block";
          };
          reader.readAsDataURL(file);
      }
  });

  runSrBtn.addEventListener("click", async () => {
    runSrBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    runSrBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("aoi_id", aoiSelector.value);
      formData.append("cloud_threshold", (cloudSlider.value / 100).toFixed(2));
      formData.append("enable_context_dem", document.getElementById("contextDemToggle").checked);
      if (base64Image) formData.append("image_base64", base64Image);

      const res = await fetch("/api/super_resolve", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.status === "success") {
        document.getElementById("metricPsnr").textContent = `${data.metrics.PSNR_dB.toFixed(2)} dB`;
        document.getElementById("metricSsim").textContent = `${data.metrics.SSIM.toFixed(3)}`;
        document.getElementById("metricSam").textContent = `${data.metrics.SAM_deg.toFixed(2)}°`;
        document.getElementById("metricErgas").textContent = `${data.metrics.ERGAS.toFixed(2)}`;

        srImage.src = data.sr_image_b64;
        srImage.style.display = "block";
        uncertaintyOverlay.src = data.uncertainty_b64;
      }
    } catch (err) {
      console.error("Inference Error:", err);
    } finally {
      runSrBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run BharatSRM-Net v4';
      runSrBtn.disabled = false;
    }
  });
});
