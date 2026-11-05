(() => {
const $ = id => document.getElementById(id);
const state = { rounds: [], round: -1, mode: "sketch", tool: null, tools: {}, models: {}, source: null, sourceIsExample: false, style: null, busy: false, timer: 0, seq: 0 };

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

function showRemaining(n) {
  const el = $("remaining");
  el.hidden = n == null;
  if (n != null) el.textContent = `${n} edit${n === 1 ? "" : "s"} left today`;
}

/* ---------- boot ---------- */
async function boot() {
  try {
    const [st, cat] = await Promise.all([api("/api/status").then(r => r.json()), api("/api/catalog").then(r => r.json())]);
    const pill = $("statusPill");
    pill.textContent = st.cloudflare_ready ? "Cloudflare connected" : "Cloudflare not set up";
    pill.className = "pill " + (st.cloudflare_ready ? "ok" : "bad");
    showRemaining(st.remaining_today);
    if (!st.claude_ready) $("askClaude").hidden = true;
    if (!st.cloudflare_ready) $("fillBox").hidden = true;
    if (st.agent_brain) {
      $("agentPanel").hidden = false;
      $("agentBrain").textContent = st.agent_brain === "claude" ? "Claude" : st.agent_brain === "cloudflare" ? "Free Cloudflare model" : st.agent_brain;
    }
    for (const s of cat.sizes) $("size").add(new Option(s === "auto" ? "Match input" : s.replace("x", " × "), s));
    for (const m of cat.models) {
      state.models[m.key] = m;
      const o = new Option(`${m.label} (free ${m.free_per_day}/day)`, m.key);
      o.title = m.description;
      $("model").add(o);
    }
    if (state.models[st.default_model]) $("model").value = st.default_model;
    initMotions(cat.motions || []);
    for (const t of cat.tools) {
      state.tools[t.key] = t;
      const b = document.createElement("button");
      b.type = "button"; b.dataset.k = t.key; b.textContent = t.label; b.setAttribute("role", "tab");
      b.onclick = () => selectTool(t.key);
      $("tools").appendChild(b);
    }
    onModel();
    selectTool(cat.tools[0].key);
  } catch (e) {
    $("statusPill").textContent = "Server not responding"; $("statusPill").className = "pill bad";
    setStatus(e.message, true);
  }
  loadHistory();
}

/* ---------- tools and presets ---------- */
function selectTool(key) {
  const tool = state.tools[key];
  state.tool = tool;
  for (const b of $("tools").children) b.setAttribute("aria-selected", b.dataset.k === key);
  $("toolHint").textContent = tool.hint;
  $("traceBox").hidden = key !== "sketch";
  $("scopeBox").hidden = !tool.brushable;
  state.area = false;
  $("scopeWhole").setAttribute("aria-pressed", true);
  $("scopeArea").setAttribute("aria-pressed", false);
  $("targetBox").hidden = !tool.targets.length;
  $("targetList").replaceChildren(...tool.targets.map(t => new Option(t)));
  $("target").value = tool.targets[0] || "";
  $("targetLbl").textContent = tool.key === "materials" ? "Surface to change" : "Object (for Remove and Repaint)";
  const box = $("presets"); box.replaceChildren();
  tool.presets.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = p.label;
    b.onclick = () => applyPreset(i);
    box.appendChild(b);
  });
  applyPreset(0);
  selectMode(currentMode());
  // Swap in this tool's example unless the user uploaded their own image.
  if (!state.source || state.sourceIsExample) loadExample(tool.example);
}
function currentMode() {
  const tool = state.tool;
  if (tool.key === "sketch" && $("trace").checked) return "edges";
  if (tool.brushable && state.area) return "local";
  return tool.mode;
}
function applyPreset(i) {
  state.preset = i;
  const preset = state.tool.presets[i];
  $("prompt").value = preset.prompt.replaceAll("{target}", $("target").value.trim() || "object");
  [...$("presets").children].forEach((b, j) => b.setAttribute("aria-pressed", i === j));
  if (state.tool.brushable && preset.area !== !!state.area) setScope(preset.area);
}
$("target").oninput = () => { if (state.preset != null) applyPreset(state.preset); };
$("trace").onchange = () => selectMode(currentMode());
function setScope(area) {
  state.area = area;
  $("scopeWhole").setAttribute("aria-pressed", !area);
  $("scopeArea").setAttribute("aria-pressed", area);
  selectMode(currentMode());
}
$("scopeWhole").onclick = () => setScope(false);
$("scopeArea").onclick = () => setScope(true);
$("prompt").addEventListener("input", () => { state.preset = null; [...$("presets").children].forEach(b => b.setAttribute("aria-pressed", false)); });

function selectMode(key) {
  state.mode = key;
  syncAgentScope();
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
function setSource(file, isExample = false, keepRounds = false) {
  if (!file || !file.type.startsWith("image/")) { setStatus("Choose an image file", true); return; }
  state.source = file;
  state.sourceIsExample = isExample;
  const url = URL.createObjectURL(file);
  if (!keepRounds) {  // a new image starts a new editing session
    state.rounds = [{ file, url, label: "Original", prompt: "" }];
    state.round = 0;
    renderRounds();
    newChat();
  }
  show("srcImg", "srcEmpty", url);
  $("srcInfo").textContent = file.name || "";
  $("videoPanel").hidden = false;
  $("videoFrom").textContent = state.round > 0 ? `From round ${state.round}` : "From the original";
  const img = new Image();
  img.onload = () => {
    const old = state.keepMask && !state.maskEmpty ? state.mask : null;
    state.keepMask = false;
    state.img = img; resetMask();
    if (old) {  // the agent edits the same brushed area again in the next round
      state.mask.getContext("2d").drawImage(old, 0, 0, state.mask.width, state.mask.height);
      state.maskEmpty = false;
    }
    if (state.mode === "text") selectMode("edit");
    else if (state.mode === "local") selectMode("local");
    else schedulePreview(0);
  };
  img.src = url;
}
// Examples from the original StableDiffusionWithGradio repo: an interior line drawing and a bedroom photo.
async function loadExample(name) {
  if (state.source && state.source.name === name) return;
  try {
    const blob = await (await api(`/examples/${name}`)).blob();
    if (!state.source || state.sourceIsExample) setSource(new File([blob], name, { type: "image/png" }), true);
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
    $("mapInfo").textContent = "";
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
    const fd = new FormData();
    fd.append("idea", idea);
    fd.append("mode", state.mode);
    if (state.source) fd.append("image", state.source);
    const s = await (await api("/api/prompt", { method: "POST", body: fd })).json();
    $("prompt").value = s.prompt;
    [...$("presets").children].forEach(b => b.setAttribute("aria-pressed", false));
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
  if ($("varToggle").checked) fd.append("variations", "3");

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
    $("outInfo").textContent = `seed ${r.seed} · ${r.cached ? "cached" : r.elapsed + "s"}` + driftNote(r.drift);
    await addRound(r, fd.get("prompt"));
    showVariants(r.variants || [], r.failed || 0);
    showRemaining(r.remaining_today);
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

/* ---------- variations ---------- */
// Several seeds of the same edit. The first one becomes the round right away;
// picking another swaps it in place of that round.
function showVariants(variants, failed) {
  const box = $("variants"); box.replaceChildren();
  $("variantsPanel").hidden = variants.length < 2;
  if (failed) setStatus(`${failed} variation${failed > 1 ? "s" : ""} failed; showing the rest.`, true);
  variants.forEach((v, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("aria-pressed", i === 0);
    const img = document.createElement("img"); img.src = v.image_url; img.alt = `Variation ${i + 1}`;
    const cap = document.createElement("span"); cap.textContent = `seed ${v.seed}`;
    b.append(img, cap);
    b.onclick = () => pickVariant(v, b);
    box.appendChild(b);
  });
}
async function pickVariant(v, button) {
  [...$("variants").children].forEach(b => b.setAttribute("aria-pressed", b === button));
  const blob = await (await api(v.image_url)).blob();
  const cur = state.rounds[state.round];
  Object.assign(cur, { file: new File([blob], `round-${state.round}.png`, { type: "image/png" }), url: v.image_url });
  state.rounds = state.rounds.slice(0, state.round + 1);  // later rounds were built on the old pick
  setSource(cur.file, false, true);
  show("outImg", "outEmpty", v.image_url);
  $("openOut").href = v.image_url;
  $("outInfo").textContent = `seed ${v.seed}`;
  renderRounds();
}

/* ---------- multi-round editing ---------- */
// Every result becomes the input of the next round. Going back to an earlier
// round and editing again branches: the later rounds are replaced.
async function addRound(r, prompt) {
  const blob = await (await api(r.image_url)).blob();
  const n = state.round + 1;
  const file = new File([blob], `round-${n}.png`, { type: "image/png" });
  state.rounds = state.rounds.slice(0, state.round + 1);
  state.rounds.push({ file, url: r.image_url, label: state.tool.label, prompt, guide: r.guide_url });
  state.round = n;
  setSource(file, false, true);
  // A render is a photo now, so the next round edits it rather than re-rendering lines.
  if (state.tool.key === "sketch") selectTool("edit");
  renderRounds();
}
function goToRound(i) {
  if (i < 0 || i >= state.rounds.length || i === state.round) return;
  $("variantsPanel").hidden = true;
  state.round = i;
  const r = state.rounds[i];
  setSource(r.file, false, true);
  if (i > 0) {
    show("outImg", "outEmpty", r.url);
    $("openOut").href = r.url; $("openOut").hidden = false;
    $("outInfo").textContent = `round ${i}`;
  } else {
    hide("outImg", "outEmpty"); $("openOut").hidden = true; $("outInfo").textContent = "";
  }
  renderRounds();
}
function renderRounds() {
  const list = $("rounds"); list.replaceChildren();
  $("roundsPanel").hidden = state.rounds.length < 2;
  state.rounds.forEach((r, i) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button"; b.title = r.prompt || r.label;
    if (i === state.round) b.setAttribute("aria-current", "step");
    if (i > state.round) b.classList.add("future");
    const img = document.createElement("img"); img.src = r.url; img.alt = "";
    const cap = document.createElement("span"); cap.className = "cap";
    const num = document.createElement("span"); num.className = "num"; num.textContent = i;
    const lbl = document.createElement("span"); lbl.textContent = r.label;
    cap.append(num, lbl); b.append(img, cap);
    b.onclick = () => goToRound(i);
    li.appendChild(b); list.appendChild(li);
  });
  $("undoRound").disabled = state.round <= 0;
  $("redoRound").disabled = state.round >= state.rounds.length - 1;
  $("resetRounds").disabled = state.round === 0;
  list.querySelector('[aria-current="step"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
  renderCompare();
}
/* ---------- before / after ---------- */
state.cmpPrev = false;
function renderCompare() {
  const show = state.round >= 1;
  $("comparePanel").hidden = !show;
  if (!show) return;
  const beforeIdx = state.cmpPrev ? state.round - 1 : 0;
  const before = state.rounds[beforeIdx], after = state.rounds[state.round];
  $("cmpBefore").src = before.url;
  $("cmpAfter").src = after.url;
  $("cmpBeforeTag").textContent = beforeIdx === 0 ? "Original" : `Round ${beforeIdx}`;
  $("cmpAfterTag").textContent = `Round ${state.round}`;
  $("cmpOrig").setAttribute("aria-pressed", !state.cmpPrev);
  $("cmpPrev").setAttribute("aria-pressed", state.cmpPrev);
  const img = new Image();
  img.onload = () => { $("compare").style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`; };
  img.src = after.url;
}
$("cmpOrig").onclick = () => { state.cmpPrev = false; renderCompare(); };
$("cmpPrev").onclick = () => { state.cmpPrev = true; renderCompare(); };
{
  const cmp = $("compare");
  const set = pct => {
    pct = Math.max(0, Math.min(100, pct));
    $("cmpAfter").style.clipPath = `inset(0 0 0 ${pct}%)`;
    $("handle").style.left = pct + "%";
    cmp.setAttribute("aria-valuenow", Math.round(pct));
  };
  const fromEvent = e => { const r = cmp.getBoundingClientRect(); set((e.clientX - r.left) / r.width * 100); };
  let drag = false;
  cmp.addEventListener("pointerdown", e => { drag = true; cmp.setPointerCapture(e.pointerId); fromEvent(e); });
  cmp.addEventListener("pointermove", e => { if (drag) fromEvent(e); });
  cmp.addEventListener("pointerup", () => { drag = false; });
  cmp.addEventListener("keydown", e => {
    const now = +cmp.getAttribute("aria-valuenow");
    if (e.key === "ArrowLeft") { set(now - 5); e.preventDefault(); }
    if (e.key === "ArrowRight") { set(now + 5); e.preventDefault(); }
  });
}

$("undoRound").onclick = () => goToRound(state.round - 1);
$("redoRound").onclick = () => goToRound(state.round + 1);
$("resetRounds").onclick = () => goToRound(0);

/* ---------- chat editing agent ---------- */
// Each message runs the LangGraph agent on the server: plan -> edit -> check
// (retry once) -> next step. Accepted steps become rounds; the chat thread
// remembers the conversation, and every message starts from the current round.
const newThreadId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).replace(/[^A-Za-z0-9_-]/g, "");
state.thread = newThreadId();
function newChat() {
  state.thread = newThreadId();
  const log = $("chatLog");
  [...log.children].slice(1).forEach(el => el.remove());  // keep the intro note
}
$("newChat").onclick = () => { if (!state.busy) newChat(); };
function chatAdd(cls, text) {
  const el = document.createElement("div");
  el.className = cls; el.textContent = text;
  $("chatLog").appendChild(el);
  el.scrollIntoView({ block: "nearest" });
  return el;
}
function stepLine(text) {
  const el = document.createElement("div"); el.className = "step";
  const img = document.createElement("img"); img.alt = ""; img.hidden = true;
  const t = document.createElement("span");
  el.append(img, t);
  $("chatLog").appendChild(el);
  el.scrollIntoView({ block: "nearest" });
  const set = (html) => { t.innerHTML = html; el.scrollIntoView({ block: "nearest" }); };
  set(text);
  return { el, img, set };
}
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function* readSSE(res) {
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

async function addAgentRound(r) {
  const blob = await (await api(r.image_url)).blob();
  const n = state.round + 1;
  const file = new File([blob], `round-${n}.png`, { type: "image/png" });
  state.rounds = state.rounds.slice(0, state.round + 1);
  state.rounds.push({ file, url: r.image_url, label: "Agent · " + r.title, prompt: r.prompt });
  state.round = n;
  state.keepMask = state.mode === "local";
  setSource(file, false, true);
  if (state.tool.key === "sketch") selectTool("edit");
  show("outImg", "outEmpty", r.image_url);
  $("openOut").href = r.image_url; $("openOut").hidden = false;
  $("outInfo").textContent = `round ${n} · seed ${r.seed}`;
  if (r.guide_url) { show("mapImg", "mapEmpty", r.guide_url); $("mapInfo").textContent = "Crop sent"; }
  renderRounds();
}

// The chat follows the same "Apply to" choice as the tools: in Brushed area mode
// the input turns into the brush canvas and every step changes only that area.
function syncAgentScope() {
  const area = state.mode === "local";
  $("agentWhole").setAttribute("aria-pressed", !area);
  $("agentArea").setAttribute("aria-pressed", area);
  $("agentHint").textContent = area
    ? "Brush the area on the input image. Every step changes only that area; the rest stays pixel-identical. The area is kept for follow-up messages. Lighting and style need Whole image."
    : "Each step edits the whole image. Switch to Brushed area to change only a part you paint. Each step uses one edit from the daily allowance (two if it is retried).";
}
function setAgentScope(area) {
  if (state.busy || area === (state.mode === "local")) return;
  if (state.tool.brushable) setScope(area);
  else if (area && state.tool.mode === "local") selectMode("local");
  else { selectTool("edit"); if (area) setScope(true); }
}
$("agentWhole").onclick = () => setAgentScope(false);
$("agentArea").onclick = () => setAgentScope(true);

$("agentMsg").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("agentForm").requestSubmit(); }
});
$("agentForm").onsubmit = async e => {
  e.preventDefault();
  const message = $("agentMsg").value.trim();
  if (!message || state.busy) return;
  if (!state.source) { setStatus("Upload an image first", true); return; }
  const area = state.mode === "local";
  if (area && state.maskEmpty) { setStatus("Brush the area you want to change on the input image first", true); return; }
  state.busy = true; $("agentSend").disabled = true; $("generate").disabled = true;
  $("variantsPanel").hidden = true;
  chatAdd("msg user", message);
  $("agentMsg").value = "";
  const fd = new FormData();
  fd.append("message", message);
  fd.append("thread", state.thread);
  fd.append("model", $("model").value);
  fd.append("image", state.source);
  if (area) fd.append("mask", await maskBlob(), "mask.png");
  let steps = [], idx = 0, line = stepLine('<span class="spin"></span>Planning…');
  const editing = () => `<span class="spin"></span>Step ${idx + 1}/${steps.length} · <b>${esc(steps[idx] || "")}</b> — editing…`;
  try {
    const res = await api("/api/agent", { method: "POST", body: fd });
    for await (const { event, data } of readSSE(res)) {
      if (event === "plan") {
        line.el.remove(); line = null;
        steps = data.steps;
        if (steps.length) {
          const m = chatAdd("msg agent", data.reply || "Plan:");
          const ol = document.createElement("ol");
          steps.forEach(t => { const li = document.createElement("li"); li.textContent = t; ol.appendChild(li); });
          m.appendChild(ol);
          line = stepLine(editing());
        }
      } else if (event === "attempt") {
        line.img.src = data.image_url; line.img.hidden = false;
        line.set(`<span class="spin"></span>Step ${idx + 1}/${steps.length} · <b>${esc(steps[idx])}</b> — checking…`);
      } else if (event === "check") {
        if (data.retry) line.set(`<span class="spin"></span>Step ${idx + 1}/${steps.length} · <b>${esc(steps[idx])}</b> — <span class="bad">${esc(data.reason || "missed")}</span>, retrying…`);
      } else if (event === "round") {
        const mark = data.ok ? `<span class="ok">✓</span>` : `<span class="bad">⚠</span>`;
        line.img.src = data.image_url; line.img.hidden = false;
        line.set(`${mark} Step ${idx + 1}/${steps.length} · <b>${esc(data.title)}</b>${data.check ? " — " + esc(data.check) : ""}${data.attempts > 1 ? " (2nd try)" : ""}${esc(driftNote(data.drift))}`);
        await addAgentRound(data);
        idx += 1;
        line = idx < steps.length ? stepLine(editing()) : null;
      } else if (event === "done") {
        if (line) { line.el.remove(); line = null; }
        // With a plan, its reply is already shown: add only warnings and stop reasons.
        const text = steps.length ? data.notes.join("\n") : data.reply;
        if (text) chatAdd("msg agent" + (steps.length ? " err" : ""), text);
        showRemaining(data.remaining_today);
      } else if (event === "error") {
        if (line) { line.el.remove(); line = null; }
        chatAdd("msg agent err", data.detail);
      }
    }
  } catch (err) {
    if (line) line.el.remove();
    chatAdd("msg agent err", err.message);
  } finally {
    state.busy = false; $("agentSend").disabled = false; $("generate").disabled = false;
    loadHistory();
  }
};

// Drift guard: whole-image rounds are registered back onto their input on the server.
function driftNote(d) {
  if (!d || !d.fixed) return "";
  const pct = Math.abs(d.scale - 1) * 100;
  return ` · drift fixed ${Math.round(d.max_offset)}px` + (pct >= 0.5 ? ` (${pct.toFixed(1)}% zoom)` : "");
}

/* ---------- camera-move video ---------- */
// Depth from the current round, then a small virtual camera move rendered on the server.
function initMotions(motions) {
  const box = $("motions"); box.replaceChildren();
  motions.forEach((m, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = m.label; b.title = m.description; b.dataset.k = m.key;
    b.onclick = () => pickMotion(m);
    box.appendChild(b);
    if (i === 0) pickMotion(m);
  });
}
function pickMotion(m) {
  state.motion = m.key;
  [...$("motions").children].forEach(b => b.setAttribute("aria-pressed", b.dataset.k === m.key));
  $("motionHint").textContent = m.description;
}
$("makeVideo").onclick = async () => {
  if (!state.source || state.videoBusy) return;
  state.videoBusy = true;
  const btn = $("makeVideo"); btn.disabled = true; btn.textContent = "Rendering…";
  const fd = new FormData();
  fd.append("image", state.source);
  fd.append("motion", state.motion || "push");
  fd.append("seconds", $("videoLen").value);
  const fill = $("videoFill").checked;
  if (fill) { fd.append("fill", "true"); fd.append("model", $("model").value); btn.textContent = "Rendering + filling…"; }
  try {
    const r = await (await api("/api/video", { method: "POST", body: fd })).json();
    $("videoOut").hidden = false;
    const v = $("videoPlayer");
    v.src = r.video_url; v.play().catch(() => {});
    $("depthImg").src = r.depth_url;
    $("videoDownload").href = r.video_url;
    setStatus(`Video ready: ${r.frames} frames in ${r.elapsed}s` + (fill ? ` · ${r.fills} area${r.fills === 1 ? "" : "s"} filled by FLUX` : ""));
    if (fill) showRemaining(r.remaining_today);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    state.videoBusy = false; btn.disabled = false; btn.textContent = "Make video";
  }
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
