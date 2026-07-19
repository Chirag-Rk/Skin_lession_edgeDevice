// =============================================================================
// DermOS Edge Dashboard Javascript Logic
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
    
    // Elements - Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Elements - Capture & Inference
    const captureBtn = document.getElementById("capture-btn");
    const ledToggleBtn = document.getElementById("led-toggle-btn");
    const ledTestBtn = document.getElementById("led-test-btn");
    const captureOverlay = document.getElementById("capture-overlay");
    const liveStream = document.getElementById("live-stream");

    // Elements - Prediction Panel
    const topClassLabel = document.getElementById("top-class-label");
    const topClassConf = document.getElementById("top-class-conf");
    const probsContainer = document.getElementById("probs-container");
    const latencyVal = document.getElementById("latency-val");
    const capturedImg = document.getElementById("captured-img");
    const heatmapImg = document.getElementById("heatmap-img");
    const overlayImg = document.getElementById("overlay-img");
    const deviceTypeBadge = document.getElementById("device-type-badge");

    // Elements - System Stats
    const cpuStat = document.getElementById("cpu-stat");
    const cpuBar = document.getElementById("cpu-bar");
    const gpuStat = document.getElementById("gpu-stat");
    const gpuBar = document.getElementById("gpu-bar");
    const ramStat = document.getElementById("ram-stat");
    const ramBar = document.getElementById("ram-bar");
    const tempStat = document.getElementById("temp-stat");
    const tempBar = document.getElementById("temp-bar");

    // Elements - Federated Learning
    const flClientId = document.getElementById("fl-client-id");
    const flStartBtn = document.getElementById("fl-start-btn");
    const flStopBtn = document.getElementById("fl-stop-btn");
    const flStatusIndicator = document.getElementById("fl-status-indicator");
    const terminalOutput = document.getElementById("terminal-output");

    // Elements - Settings
    const cameraModeSelect = document.getElementById("camera-mode-select");
    const cameraIndexInput = document.getElementById("camera-index-input");
    const ledPinInput = document.getElementById("led-pin-input");

    // State Variables
    let flPollInterval = null;
    let ledState = false;

    // -------------------------------------------------------------------------
    // 1. TABS NAVIGATION
    // -------------------------------------------------------------------------
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            // Add active state to clicked button and target tab pane
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // -------------------------------------------------------------------------
    // 2. SETTINGS INITIALIZATION & INPUTS
    // -------------------------------------------------------------------------
    async function loadSettings() {
        try {
            const res = await fetch("/api/settings");
            const data = await res.json();
            
            cameraModeSelect.value = data.camera_mode;
            cameraIndexInput.value = data.camera_index;
            ledPinInput.value = data.led_pin;
            
            const dev = data.device.toUpperCase();
            deviceTypeBadge.innerHTML = `<i class="fa-solid fa-microchip"></i> DEVICE: ${dev}`;
            if (dev === "CUDA") {
                deviceTypeBadge.style.background = "rgba(16, 185, 129, 0.15)";
                deviceTypeBadge.style.color = "var(--accent-emerald)";
                deviceTypeBadge.style.borderColor = "var(--accent-emerald)";
            }
        } catch (e) {
            console.error("Failed to load settings:", e);
        }
    }

    async function saveSettings() {
        const payload = {
            camera_mode: cameraModeSelect.value,
            camera_index: parseInt(cameraIndexInput.value) || 0,
            led_pin: parseInt(ledPinInput.value) || 12
        };
        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                console.log("Settings updated successfully.");
                // Reload stream if mode changes
                liveStream.src = "/video_feed?t=" + new Date().getTime();
            }
        } catch (e) {
            console.error("Failed to save settings:", e);
        }
    }

    cameraModeSelect.addEventListener("change", saveSettings);
    cameraIndexInput.addEventListener("change", saveSettings);
    ledPinInput.addEventListener("change", saveSettings);

    // -------------------------------------------------------------------------
    // 3. CAPTURE & DIAGNOSIS FLOW
    // -------------------------------------------------------------------------
    captureBtn.addEventListener("click", async () => {
        // Flash overlay animation
        captureOverlay.classList.add("flash");
        setTimeout(() => captureOverlay.classList.remove("flash"), 120);

        // UI states (Loading)
        captureBtn.disabled = true;
        captureBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Neural Diagnostic...`;
        
        try {
            const res = await fetch("/api/capture", { method: "POST" });
            const data = await res.json();
            
            if (data.success) {
                // Update Prediction Output
                topClassLabel.textContent = data.label_name;
                topClassConf.textContent = `${(data.confidence * 100).toFixed(1)}%`;
                latencyVal.textContent = data.latency_ms.toFixed(2);

                // Render Probability Bars
                renderProbabilityBars(data.probabilities, data.label_name);

                // Reload Images with Cache Buster
                const t = new Date().getTime();
                capturedImg.src = `${data.captured_url}&t=${t}`;
                heatmapImg.src = `${data.heatmap_url}&t=${t}`;
                overlayImg.src = `${data.overlay_url}&t=${t}`;

                // Auto switch tabs to show overlay
                tabBtns.forEach(b => b.classList.remove("active"));
                tabPanes.forEach(p => p.classList.remove("active"));
                tabBtns[0].classList.add("active");
                tabPanes[0].classList.add("active");
            } else {
                alert("Neural prediction error: " + data.error);
            }
        } catch (e) {
            console.error("Capture request error:", e);
            alert("Connection error executing diagnostic.");
        } finally {
            // Restore button state
            captureBtn.disabled = false;
            captureBtn.innerHTML = `<i class="fa-solid fa-circle-radiation"></i> Capture & Diagnose`;
        }
    });

    function renderProbabilityBars(probs, topLabel) {
        probsContainer.innerHTML = "";
        
        // Sort classes by probability descending
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        
        sorted.forEach(([labelName, val]) => {
            const pct = (val * 100).toFixed(1);
            const isTop = labelName === topLabel;
            
            const row = document.createElement("div");
            row.className = `prob-row ${isTop ? 'top-pred' : ''}`;
            row.innerHTML = `
                <div class="prob-meta">
                    <span class="class-title">${labelName}</span>
                    <span class="class-pct">${pct}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
            `;
            probsContainer.appendChild(row);
            
            // Micro-animation for bar fill
            setTimeout(() => {
                row.querySelector(".prob-bar-fill").style.width = `${pct}%`;
            }, 50);
        });
    }

    // -------------------------------------------------------------------------
    // 4. LED HARDWARE CONTROLS
    // -------------------------------------------------------------------------
    ledToggleBtn.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/toggle_led", { method: "POST" });
            const data = await res.json();
            
            ledState = data.led_enabled;
            if (ledState) {
                ledToggleBtn.classList.add("active-led");
                ledToggleBtn.innerHTML = `<i class="fa-solid fa-lightbulb"></i> LED: On`;
            } else {
                ledToggleBtn.classList.remove("active-led");
                ledToggleBtn.innerHTML = `<i class="fa-solid fa-lightbulb"></i> LED: Off`;
            }
        } catch (e) {
            console.error("LED trigger error:", e);
        }
    });

    ledTestBtn.addEventListener("click", async () => {
        try {
            await fetch("/api/test_led", { method: "POST" });
        } catch (e) {
            console.error("Test LED request error:", e);
        }
    });

    // -------------------------------------------------------------------------
    // 5. SYSTEM HEALTH STATS MONITOR
    // -------------------------------------------------------------------------
    async function updateSystemStats() {
        try {
            const res = await fetch("/api/system_stats");
            const data = await res.json();
            
            // CPU
            cpuStat.textContent = `${data.cpu_usage.toFixed(0)}%`;
            cpuBar.style.width = `${data.cpu_usage}%`;
            
            // GPU
            gpuStat.textContent = `${data.gpu_usage.toFixed(0)}%`;
            gpuBar.style.width = `${data.gpu_usage}%`;
            
            // RAM
            ramStat.textContent = `${data.ram_usage.toFixed(0)}%`;
            ramBar.style.width = `${data.ram_usage}%`;
            
            // Temperature
            tempStat.textContent = `${data.temperature.toFixed(1)}°C`;
            // Normalize temp bar: 30C to 80C
            const tempPct = Math.min(Math.max((data.temperature - 30) * 2, 0), 100);
            tempBar.style.width = `${tempPct}%`;
            
            // Color temp warning if too high
            if (data.temperature > 70) {
                tempBar.style.background = "var(--accent-rose)";
            } else if (data.temperature > 55) {
                tempBar.style.background = "var(--accent-amber)";
            } else {
                tempBar.style.background = "var(--accent-teal)";
            }
        } catch (e) {
            console.error("Telemetry query failed:", e);
        }
    }

    // Start Telemetry loop
    setInterval(updateSystemStats, 2000);
    updateSystemStats();

    // -------------------------------------------------------------------------
    // 6. FEDERATED LEARNING ACTIONS
    // -------------------------------------------------------------------------
    async function pollFLStatus() {
        try {
            const res = await fetch("/api/fl/status");
            const data = await res.json();
            
            terminalOutput.textContent = data.logs;
            // Auto-scroll log terminal
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
            
            if (!data.active) {
                clearInterval(flPollInterval);
                flPollInterval = null;
                
                flStartBtn.classList.remove("hidden");
                flStopBtn.classList.add("hidden");
                
                flStatusIndicator.textContent = "IDLE";
                flStatusIndicator.className = "status-indicator idle";
                
                // Trigger model weights reload (via predict instantiation next capture)
                console.log("Federated round finished. Global model updated.");
            }
        } catch (e) {
            console.error("FL Status check failed:", e);
        }
    }

    flStartBtn.addEventListener("click", async () => {
        const clientId = parseInt(flClientId.value) || 0;
        
        flStartBtn.classList.add("hidden");
        flStopBtn.classList.remove("hidden");
        flStatusIndicator.textContent = "RUNNING";
        flStatusIndicator.className = "status-indicator running";
        terminalOutput.textContent = "Connecting to Flower server...\n";

        try {
            const res = await fetch("/api/fl/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ client_id: clientId })
            });
            const data = await res.json();
            
            if (data.success) {
                // Start log polling
                flPollInterval = setInterval(pollFLStatus, 1000);
            } else {
                alert(data.message);
                flStartBtn.classList.remove("hidden");
                flStopBtn.classList.add("hidden");
                flStatusIndicator.textContent = "IDLE";
                flStatusIndicator.className = "status-indicator idle";
            }
        } catch (e) {
            console.error("FL Trigger failed:", e);
            flStartBtn.classList.remove("hidden");
            flStopBtn.classList.add("hidden");
            flStatusIndicator.textContent = "IDLE";
            flStatusIndicator.className = "status-indicator idle";
        }
    });

    flStopBtn.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/fl/stop", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                terminalOutput.textContent += "\n[System] FL Execution aborted by User.\n";
            }
        } catch (e) {
            console.error("FL abort error:", e);
        }
    });

    // -------------------------------------------------------------------------
    // INIT RUN
    // -------------------------------------------------------------------------
    loadSettings();
});
