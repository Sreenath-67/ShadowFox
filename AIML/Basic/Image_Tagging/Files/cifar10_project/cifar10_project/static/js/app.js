/* ═══════════════════════════════════════════════════════════
   CIFAR-10 Vision Lab – Client JS
   ═══════════════════════════════════════════════════════════ */

"use strict";

// ── Socket.IO ────────────────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });

socket.on("connect",          ()  => console.log("[WS] connected"));
socket.on("user_count",       (d) => updateUserCount(d.count));
socket.on("new_prediction",   (d) => prependFeedCard(d));
socket.on("server_info",      (d) => {
  updateUserCount(d.user_count);
  if (d.recent_feed?.length) {
    d.recent_feed.slice().reverse().forEach(prependFeedCard);
  }
});

function updateUserCount(n) {
  const el = document.getElementById("userCount");
  if (el) el.textContent = `${n} USER${n !== 1 ? "S" : ""}`;
}

// ── DOM refs ─────────────────────────────────────────────────
const dropZone     = document.getElementById("dropZone");
const fileInput    = document.getElementById("fileInput");
const dropInner    = document.getElementById("dropInner");
const previewImg   = document.getElementById("previewImg");
const classifyBtn  = document.getElementById("classifyBtn");
const resultCard   = document.getElementById("resultCard");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const loadingOverlay    = document.getElementById("loadingOverlay");
const probList     = document.getElementById("probList");
const feedStrip    = document.getElementById("feedStrip");
const toast        = document.getElementById("toast");

let selectedFile = null;

// ── Drag & drop ──────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave",  () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    dropInner.classList.add("hidden");
    classifyBtn.disabled = false;
  };
  reader.readAsDataURL(file);
  // Reset results
  resultCard.classList.add("hidden");
  resultPlaceholder.classList.remove("hidden");
  clearProbList();
  highlightChip(null);
}

// ── Classify ─────────────────────────────────────────────────
classifyBtn.addEventListener("click", classify);

async function classify() {
  if (!selectedFile) return;

  setLoading(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showToast(data.error || "Classification failed", "error");
      setLoading(false);
      return;
    }

    displayResult(data);
  } catch (err) {
    showToast("Network error: " + err.message, "error");
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  if (on) {
    loadingOverlay.classList.remove("hidden");
    resultCard.classList.add("hidden");
    resultPlaceholder.classList.add("hidden");
    classifyBtn.disabled = true;
    classifyBtn.querySelector(".btn-text").textContent = "CLASSIFYING…";
  } else {
    loadingOverlay.classList.add("hidden");
    classifyBtn.disabled = false;
    classifyBtn.querySelector(".btn-text").textContent = "CLASSIFY IMAGE";
  }
}

// ── Display result ───────────────────────────────────────────
function displayResult(data) {
  // Result card
  document.getElementById("resultEmoji").textContent   = data.emoji;
  document.getElementById("resultLabel").textContent   = data.prediction.toUpperCase();
  document.getElementById("resultCategory").textContent = data.category.toUpperCase();
  document.getElementById("confNumber").textContent    = (data.confidence * 100).toFixed(1);
  document.getElementById("inferenceTime").textContent = data.inference_ms;

  resultCard.classList.remove("hidden");
  resultPlaceholder.classList.add("hidden");

  // Probability bars
  renderProbBars(data.all_classes);

  // Highlight chip
  highlightChip(data.prediction);

  // Confetti pop for high confidence
  if (data.confidence > 0.85) confettiPop();
}

// ── Prob bars ────────────────────────────────────────────────
function renderProbBars(classes) {
  probList.innerHTML = "";
  classes.forEach((c, idx) => {
    const pct    = (c.confidence * 100).toFixed(1);
    const isTop  = idx === 0;

    const row    = document.createElement("div");
    row.className = "prob-row";

    const emoji  = document.createElement("div");
    emoji.className = "prob-emoji";
    emoji.textContent = c.emoji;

    const barWrap = document.createElement("div");
    barWrap.className = "prob-bar-wrap";

    const fill = document.createElement("div");
    fill.className  = `prob-bar-fill${isTop ? " top-bar" : ""}`;
    fill.setAttribute("data-label", c.label);
    barWrap.appendChild(fill);

    const pctEl = document.createElement("div");
    pctEl.className = `prob-pct${isTop ? " top-pct" : ""}`;
    pctEl.textContent = pct + "%";

    row.appendChild(emoji);
    row.appendChild(barWrap);
    row.appendChild(pctEl);
    probList.appendChild(row);

    // Animate bar width after a tick
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.style.width = Math.max(c.confidence * 100, 2) + "%";
      });
    });
  });
}

function clearProbList() {
  probList.innerHTML = `<div class="prob-placeholder">
    <p>Run a classification to see<br/>probability distribution</p>
  </div>`;
}

// ── Class chip highlight ─────────────────────────────────────
function highlightChip(label) {
  document.querySelectorAll(".class-chips span").forEach((s) => {
    s.classList.toggle("highlight", label && s.textContent.includes(label));
  });
}

// ── Live feed ────────────────────────────────────────────────
function prependFeedCard(entry) {
  // Remove empty placeholder
  const empty = feedStrip.querySelector(".feed-empty");
  if (empty) empty.remove();

  const card = document.createElement("div");
  card.className = "feed-card";
  card.innerHTML = `
    <img src="${entry.thumbnail}" alt="${entry.label}" loading="lazy"/>
    <div class="feed-info">
      <div class="feed-label">${entry.emoji} ${entry.label}</div>
      <div class="feed-conf">${entry.confidence}%</div>
      <div class="feed-time">${entry.ts}</div>
    </div>`;

  feedStrip.prepend(card);

  // Keep max 20 cards
  while (feedStrip.children.length > 20) {
    feedStrip.removeChild(feedStrip.lastChild);
  }
}

// ── Toast ────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = "info") {
  toast.textContent = msg;
  toast.className = `toast${type === "error" ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.classList.add("hidden"); }, 4000);
}

// ── Confetti pop ─────────────────────────────────────────────
function confettiPop() {
  const colors = ["#00f5a0", "#7b5ea7", "#4fc3f7", "#ff6b6b"];
  for (let i = 0; i < 28; i++) {
    const dot = document.createElement("div");
    const size = Math.random() * 8 + 4;
    dot.style.cssText = `
      position:fixed;
      left:${Math.random() * 100}vw;
      top:${Math.random() * 40 + 20}vh;
      width:${size}px; height:${size}px;
      border-radius:${Math.random() > 0.5 ? "50%" : "2px"};
      background:${colors[Math.floor(Math.random() * colors.length)]};
      pointer-events:none; z-index:998;
      animation: confettiFall ${Math.random() * 1.5 + 0.8}s ease forwards;
    `;
    document.body.appendChild(dot);
    setTimeout(() => dot.remove(), 2500);
  }
}

// Inject confetti keyframe
const confettiStyle = document.createElement("style");
confettiStyle.textContent = `
  @keyframes confettiFall {
    0%   { transform: translateY(-20px) rotate(0deg); opacity:1; }
    100% { transform: translateY(120px) rotate(360deg); opacity:0; }
  }
`;
document.head.appendChild(confettiStyle);

// ── Init: check feed ─────────────────────────────────────────
(async () => {
  try {
    const r = await fetch("/api/feed");
    const d = await r.json();
    if (d.feed?.length === 0) {
      feedStrip.innerHTML = `<span class="feed-empty">No predictions yet — upload an image to get started</span>`;
    }
  } catch {}
})();
