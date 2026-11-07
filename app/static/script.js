"use strict";

const HEBREW = { Apple: "תפוח", Avocado: "אבוקדו", Banana: "בננה", Eggplant: "חציל", Tomato: "עגבנייה" };
const EMOJI = { Apple: "🍎", Avocado: "🥑", Banana: "🍌", Eggplant: "🍆", Tomato: "🍅", unknown: "❓" };
const MAX_SIDE = 1024; // downscale big phone photos before uploading

const $ = (id) => document.getElementById(id);
const els = {
  camera: $("camera-input"), gallery: $("gallery-input"),
  preview: $("preview"), img: $("preview-img"),
  actionRow: $("action-row"), identify: $("identify-btn"), save: $("save-btn"),
  loader: $("loader"), error: $("error"), result: $("result"),
  emoji: $("result-emoji"), label: $("result-label"), conf: $("result-conf"),
  note: $("unknown-note"), bars: $("bars"),
};

let currentBlob = null; // the (downscaled) image that will be sent and saved

function showError(msg) { els.error.textContent = msg; els.error.hidden = false; }
function clearError() { els.error.hidden = true; }

// Load the picked file, fix its orientation, shrink it and turn it into a JPEG blob.
async function fileToBlob(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; // PNGs with transparency would otherwise turn black in JPEG
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("canvas"))), "image/jpeg", 0.9));
}

async function onPick(event) {
  const file = event.target.files[0];
  event.target.value = ""; // allow picking the same file again later
  if (!file) return;
  clearError();
  els.result.hidden = true;
  if (!/^image\//.test(file.type)) { showError("הקובץ שנבחר אינו תמונה."); return; }
  try {
    currentBlob = await fileToBlob(file);
  } catch {
    showError("לא הצלחנו לקרוא את התמונה. נסו קובץ JPG או PNG אחר.");
    return;
  }
  els.img.src = URL.createObjectURL(currentBlob);
  els.preview.hidden = false;
  els.actionRow.hidden = false;
}

function renderResult(r) {
  const key = r.is_unknown ? "unknown" : r.prediction;
  els.emoji.textContent = EMOJI[key] || "🍽️";
  els.label.textContent = r.is_unknown ? "לא ידוע" : (HEBREW[r.prediction] || r.prediction);
  els.label.classList.toggle("unknown", r.is_unknown);
  els.conf.innerHTML = `רמת ביטחון: <strong>${(r.confidence * 100).toFixed(1)}%</strong>`;
  els.note.hidden = !r.is_unknown;

  els.bars.innerHTML = "";
  const entries = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);
  entries.forEach(([name, p], i) => {
    const li = document.createElement("li");
    li.className = "bar-row" + (i === 0 ? " top" : "");
    li.innerHTML = `<span>${HEBREW[name] || name}</span>
      <div class="bar-track"><div class="bar-fill"></div></div>
      <span class="bar-pct">${(p * 100).toFixed(1)}%</span>`;
    els.bars.appendChild(li);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      li.querySelector(".bar-fill").style.width = `${p * 100}%`;
    }));
  });
  els.result.hidden = false;
  els.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function identify() {
  if (!currentBlob) return;
  clearError();
  els.result.hidden = true;
  els.loader.hidden = false;
  els.identify.disabled = true;
  try {
    const form = new FormData();
    form.append("file", currentBlob, "photo.jpg");
    const res = await fetch("/predict", { method: "POST", body: form });
    if (!res.ok) {
      let detail = "אירעה שגיאה בשרת.";
      try { detail = (await res.json()).detail || detail; } catch { /* keep default */ }
      throw new Error(detail);
    }
    renderResult(await res.json());
  } catch (e) {
    showError(e instanceof TypeError ? "אין חיבור לשרת. בדקו שהוא רץ ושאתם מחוברים לרשת." : e.message);
  } finally {
    els.loader.hidden = true;
    els.identify.disabled = false;
  }
}

function saveImage() {
  if (!currentBlob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(currentBlob);
  a.download = "fruit-photo.jpg";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

els.camera.addEventListener("change", onPick);
els.gallery.addEventListener("change", onPick);
els.identify.addEventListener("click", identify);
els.save.addEventListener("click", saveImage);
