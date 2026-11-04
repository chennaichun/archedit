(() => {
const $ = id => document.getElementById(id);
const state = { mode: "edit", models: {}, source: null, style: null, busy: false, timer: 0, seq: 0 };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `Error ${res.status}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res;
}
const setStatus = (msg = "", warn = false) => { $("status").textContent = msg; $("status").classList.toggle("warn", warn); };
const show = (imgId, emptyId, url) => { const i = $(imgId); i.src = url; i.hidden = false; $(emptyId).hidden = true; };
const hide = (imgId, emptyId) => { $(imgId).hidden = true; $(emptyId).hidden = false; };

/* ---------- boot ---------- */
async function boot() {
  try {
    const [st, cat] = await Promise.all([api("/api/status").then(r => r.json()), api("/api/catalog").then(r => r.json())]);
    const pill = $("statusPill");
    pill.textContent = st.cloudflare_ready ? "Cloudflare connected" : "Cloudflare not set up";
    pill.className = "pill " + (st.cloudflare_ready ? "ok" : "bad");
    for (const s of cat.sizes) $("size").add(new Option(s.replace("x", " × "), s));
    for (const m of cat.models) {
      state.models[m.key] = m;
      const o = new Option(`${m.label} (free ${m.free_per_day}/day)`, m.key);
      o.title = m.description;
      $("model").add(o);
    }
    if (state.models[st.default_model]) $("model").value = st.default_model;
    for (const m of cat.modes) {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.k = m.key; b.textContent = m.label; b.title = m.description;
      b.onclick = () => selectMode(m.key);
      $("modes").appendChild(b);
    }
    selectMode(state.mode);
    onModel();
    loadExample();
  } catch (e) {
    $("statusPill").textContent = "Server not responding"; $("statusPill").className = "pill bad";
    setStatus(e.message, true);
  }
  loadHistory();
}

function selectMode(key) {
  state.mode = key;
  for (const b of $("modes").children) b.setAttribute("aria-pressed", b.dataset.k === key);
  $("cannyBox").hidden = key !== "edges";
  $("brushBox").hidden = key !== "local";
  showSourceView();
  if (key === "text" || key === "local") {
    hide("mapImg", "mapEmpty"); $("mapInfo").textContent = "";
    if (key === "local" && state.source) setStatus("Brush the area you want to change on the input image");
  } else schedulePreview(0);
}

/* ---------- local edit: brush mask ---------- */
const canvas = $("srcCanvas"), ctx = canvas.getContext("2d");
function showSourceView() {
  const local = state.mode === "local" && state.img;
  canvas.hidden = !local;
  $("srcImg").hidden = local || !state.source;
  if (local) drawCanvas();
}
function resetMask() {
  if (!state.img) return;
  state.mask = document.createElement("canvas");
  state.mask.width = state.img.naturalWidth; state.mask.height = state.img.naturalHeight;
  state.maskEmpty = true;
  canvas.width = state.img.naturalWidth; canvas.height = state.img.naturalHeight;
}
function drawCanvas() {
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(state.img, 0, 0);
  const tint = document.createElement("canvas");
  tint.width = canvas.width; tint.height = canvas.height;
  const t = tint.getContext("2d");
  t.drawImage(state.mask, 0, 0);
  t.globalCompositeOperation = "source-in";
  t.fillStyle = "rgba(255,70,50,0.5)";
  t.fillRect(0, 0, tint.width, tint.height);
  ctx.drawImage(tint, 0, 0);
}
function toImageXY(e) {
  const r = canvas.getBoundingClientRect();
  const s = Math.min(r.width / canvas.width, r.height / canvas.height);
  const ox = (r.width - canvas.width * s) / 2, oy = (r.height - canvas.height * s) / 2;
  return { x: (e.clientX - r.left - ox) / s, y: (e.clientY - r.top - oy) / s, s };
}
let painting = false, last = null;
function paint(e) {
  const p = toImageXY(e);
  const radius = (+$("brush").value / 2) / p.s;
  const m = state.mask.getContext("2d");
  m.fillStyle = "#fff"; m.strokeStyle = "#fff"; m.lineCap = "round"; m.lineWidth = radius * 2;
  if (last) { m.beginPath(); m.moveTo(last.x, last.y); m.lineTo(p.x, p.y); m.stroke(); }
  m.beginPath(); m.arc(p.x, p.y, radius, 0, Math.PI * 2); m.fill();
  last = p; state.maskEmpty = false;
  drawCanvas();
}
canvas.addEventListener("pointerdown", e => { e.stopPropagation(); painting = true; last = null; canvas.setPointerCapture(e.pointerId); paint(e); setStatus(); });
canvas.addEventListener("pointermove", e => { if (painting) paint(e); });
canvas.addEventListener("pointerup", () => { painting = false; last = null; });
canvas.addEventListener("click", e => e.stopPropagation());
$("brush").oninput = () => { $("brushOut").textContent = $("brush").value; };
$("clearMask").onclick = () => { resetMask(); drawCanvas(); };
function maskBlob() {
  const out = document.createElement("canvas");
  out.width = state.mask.width; out.height = state.mask.height;
  const o = out.getContext("2d");
  o.fillStyle = "#000"; o.fillRect(0, 0, out.width, out.height);
  o.drawImage(state.mask, 0, 0);
  return new Promise(res => out.toBlob(res, "image/png"));
}
function onModel() { $("stepsBox").hidden = !state.models[$("model").value]?.adjustable_steps; }
$("model").onchange = onModel;

/* ---------- source image ---------- */
function setSource(file) {
  if (!file || !file.type.startsWith("image/")) { setStatus("Choose an image file", true); return; }
  state.source = file;
  const url = URL.createObjectURL(file);
  show("srcImg", "srcEmpty", url);
  $("srcInfo").textContent = file.name || "";
  const img = new Image();
  img.onload = () => {
    state.img = img; resetMask();
    if (state.mode === "text") selectMode("edit");
    else if (state.mode === "local") selectMode("local");
    else schedulePreview(0);
  };
  img.src = url;
}
// Start with the bedroom photo from the original StableDiffusionWithGradio repo.
async function loadExample() {
  if (state.source) return;
  try {
    const blob = await (await api("/examples/sample_input.png")).blob();
    if (!state.source) setSource(new File([blob], "sample_input.png", { type: "image/png" }));
  } catch {}
}
$("upload").onclick = () => $("file").click();
$("drop").onclick = () => { if (!(state.mode === "local" && state.img)) $("file").click(); };
$("drop").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
$("file").onchange = e => setSource(e.target.files[0]);
const drop = $("drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); setSource(e.dataTransfer.files[0]); };
window.addEventListener("paste", e => {
  const it = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
  if (it) setSource(it.getAsFile());
});

/* ---------- style reference ---------- */
$("styleBtn").onclick = () => {
  if (state.style) { state.style = null; $("styleBtn").textContent = "Style image"; return; }
  $("styleFile").click();
};
$("styleFile").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  state.style = f;
  $("styleBtn").innerHTML = "Remove style image";
  const t = document.createElement("img"); t.className = "styleThumb"; t.alt = ""; t.src = URL.createObjectURL(f);
  $("styleBtn").appendChild(t);
  e.target.value = "";
};

/* ---------- guide preview ---------- */
function schedulePreview(delay = 250) {
  clearTimeout(state.timer);
  if (!state.source || state.mode === "text") return;
  state.timer = setTimeout(preview, delay);
}
async function preview() {
  const seq = ++state.seq;
  const fd = new FormData();
  fd.append("image", state.source);
  fd.append("mode", state.mode);
  fd.append("size", $("size").value);
  fd.append("low", $("low").value);
  fd.append("high", String(Math.min(255, +$("low").value * 2)));
  try {
    const blob = await (await api("/api/preview", { method: "POST", body: fd })).blob();
    if (seq !== state.seq) return;
    show("mapImg", "mapEmpty", URL.createObjectURL(blob));
    $("mapInfo").textContent = $("size").value;
  } catch (e) {
    if (seq === state.seq) setStatus(e.message, true);
  }
}
$("size").onchange = () => schedulePreview(0);
$("low").oninput = () => { $("lowOut").textContent = $("low").value; schedulePreview(); };
$("steps").oninput = () => { $("stepsOut").textContent = $("steps").value; };

/* ---------- Claude ---------- */
$("askClaude").onclick = async () => {
  const idea = $("idea").value.trim();
  if (!idea) return;
  const btn = $("askClaude"); btn.disabled = true; btn.textContent = "Writing…";
  setStatus();
  try {
    const s = await (await api("/api/prompt", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idea, mode: state.mode, use_image: !!state.source }),
    })).json();
    $("prompt").value = s.prompt;
    if (s.mode !== state.mode && !(s.mode !== "text" && !state.source)) selectMode(s.mode);
    setStatus(s.notes);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "Write prompt with Claude";
  }
};

/* ---------- generate ---------- */
$("generate").onclick = async () => {
  if (state.busy) return;
  if (state.mode !== "text" && !state.source) { setStatus("Upload an image first", true); return; }
  if (state.mode === "local" && state.maskEmpty) { setStatus("Brush the area you want to change first", true); return; }
  state.busy = true; $("generate").disabled = true;
  const fd = new FormData();
  fd.append("prompt", $("prompt").value);
  fd.append("mode", state.mode);
  fd.append("model", $("model").value);
  fd.append("size", $("size").value);
  fd.append("seed", $("seed").value || "-1");
  fd.append("low", $("low").value);
  fd.append("high", String(Math.min(255, +$("low").value * 2)));
  if (!$("stepsBox").hidden) fd.append("steps", $("steps").value);
  if (state.source && state.mode !== "text") fd.append("image", state.source);
  if (state.style && state.mode !== "text") fd.append("style_image", state.style);
  if (state.mode === "local") fd.append("mask", await maskBlob(), "mask.png");

  const t0 = Date.now();
  $("progress").hidden = false; $("bar").style.width = "8%";
  const tick = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    $("bar").style.width = Math.min(92, 8 + s * 6) + "%";
    $("outInfo").textContent = `${s.toFixed(0)} s`;
  }, 300);
  setStatus();
  try {
    const r = await (await api("/api/generate", { method: "POST", body: fd })).json();
    show("outImg", "outEmpty", r.image_url);
    $("openOut").href = r.image_url; $("openOut").hidden = false;
    $("useResult").hidden = false;
    $("outInfo").textContent = `seed ${r.seed} · ${r.elapsed}s`;
    if (r.guide_url) { show("mapImg", "mapEmpty", r.guide_url); $("mapInfo").textContent = state.mode === "local" ? "Crop sent" : r.size; }
    loadHistory();
  } catch (e) {
    setStatus(e.message, true);
    $("outInfo").textContent = "";
  } finally {
    clearInterval(tick);
    $("progress").hidden = true;
    state.busy = false; $("generate").disabled = false;
  }
};

$("useResult").onclick = async () => {
  try {
    const blob = await (await api($("outImg").src)).blob();
    setSource(new File([blob], "result.png", { type: "image/png" }));
  } catch (e) { setStatus(e.message, true); }
};

/* ---------- history ---------- */
async function loadHistory() {
  try {
    const { items } = await (await api("/api/history")).json();
    if (!items.length) return;
    $("historyPanel").hidden = false;
    const box = $("history"); box.replaceChildren();
    for (const it of items) {
      const b = document.createElement("button"); b.type = "button"; b.title = it.prompt;
      const img = document.createElement("img"); img.src = it.image_url; img.alt = ""; img.loading = "lazy";
      b.appendChild(img);
      b.onclick = () => {
        show("outImg", "outEmpty", it.image_url);
        $("openOut").href = it.image_url; $("openOut").hidden = false;
        $("outInfo").textContent = `seed ${it.seed}`;
        $("useResult").hidden = false;
        $("prompt").value = it.prompt;
        $("seed").value = it.seed;
        if (state.models[it.model]) { $("model").value = it.model; onModel(); }
        if ([...$("size").options].some(o => o.value === it.size)) $("size").value = it.size;
        if (it.guide_url) show("mapImg", "mapEmpty", it.guide_url);
      };
      box.appendChild(b);
    }
  } catch {}
}

boot();
})();
