const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { provider: "ollama", model: null, kind: "Local", mode: "native", running: false, columns: [] };

/* ---------------- models ---------------- */
async function loadModels() {
  const status = $("#status");
  try {
    const r = await (await fetch("/api/models")).json();
    const menu = $("#modelMenu");
    const item = (m, kind) => {
      const meta = kind === "Local" ? [m.params, m.size_gb ? `${m.size_gb} GB` : ""].filter(Boolean).join(" · ")
        : (m.available ? "free tier" : `set ${m.needs}`);
      return `<button class="menu-item" role="option" data-provider="${esc(m.provider)}" data-model="${esc(m.model)}" data-kind="${kind}" ${m.available ? "" : "disabled"}>
        <span><span class="mi-name">${esc(m.model)}</span><span class="mi-provider">${esc(kind === "Local" ? "Ollama · on this machine" : m.provider)}</span></span>
        <span class="mi-meta">${esc(meta)}</span></button>`;
    };
    menu.innerHTML =
      `<div class="menu-group mono-label">Local <span class="menu-group-sub">private, no limits</span></div>` +
      (r.local.length ? r.local.map((m) => item(m, "Local")).join("") : `<div class="menu-item" style="cursor:default"><span class="mi-provider">Ollama isn't running</span></div>`) +
      `<div class="menu-group mono-label">Free tier <span class="menu-group-sub">bigger models, rate limits</span></div>` +
      r.free_tier.map((m) => item(m, "Free tier")).join("");

    const preferred = r.local.find((m) => m.model === "qwen3.5:4b") || r.local[0] || r.free_tier.find((m) => m.available);
    if (preferred) selectModel(preferred.provider, preferred.model, r.local.includes(preferred) ? "Local" : "Free tier");
    status.className = "status " + (r.local.length ? "is-ok" : "is-err");
    $("#statusText").textContent = r.local.length ? "ollama connected" : "ollama offline";
  } catch (e) {
    status.className = "status is-err";
    $("#statusText").textContent = "server offline";
  }
}

function selectModel(provider, model, kind) {
  Object.assign(state, { provider, model, kind });
  $("#modelName").textContent = model;
  const k = $("#modelKind");
  k.textContent = kind;
  k.classList.toggle("is-free", kind !== "Local");
  $("#hintModel").textContent = `${model} · ${kind === "Local" ? "local" : provider}`;
  document.querySelectorAll(".menu-item").forEach((b) => b.classList.toggle("is-selected", b.dataset.model === model && b.dataset.provider === provider));
}

function toggleMenu(open) {
  const menu = $("#modelMenu"), btn = $("#modelButton");
  const show = open ?? menu.hidden;
  menu.hidden = !show;
  btn.setAttribute("aria-expanded", String(show));
}
$("#modelButton").addEventListener("click", (e) => { e.stopPropagation(); toggleMenu(); });
$("#modelMenu").addEventListener("click", (e) => {
  const b = e.target.closest(".menu-item[data-model]");
  if (!b || b.disabled) return;
  selectModel(b.dataset.provider, b.dataset.model, b.dataset.kind);
  toggleMenu(false);
});
document.addEventListener("click", () => toggleMenu(false));
document.querySelectorAll(".seg").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".seg").forEach((x) => { x.classList.toggle("is-active", x === b); x.setAttribute("aria-checked", String(x === b)); });
  state.mode = b.dataset.mode;
}));

/* ---------------- data panel ---------------- */
async function loadData() {
  const d = await (await fetch("/api/data")).json();
  state.columns = d.columns.map((c) => c.name);
  $("#dataFile").textContent = d.file;
  $("#statRows").textContent = d.rows.toLocaleString("en-US");
  $("#statCols").textContent = d.columns.length;
  $("#statSize").textContent = d.size_mb;
  $("#columns").innerHTML = d.columns.map((c) => `<li class="col" data-col="${esc(c.name)}">${esc(c.name)}<span class="dt">${esc(shortType(c.dtype))}</span></li>`).join("");
  const numeric = d.columns.map((c) => /int|float/.test(c.dtype));
  $("#preview").innerHTML =
    `<thead><tr>${d.columns.map((c) => `<th data-col="${esc(c.name)}">${esc(c.name)}</th>`).join("")}</tr></thead>` +
    `<tbody>${d.preview.map((row) => `<tr>${row.map((v, i) => v === null ? `<td class="null">null</td>` : `<td class="${numeric[i] ? "num" : ""}">${esc(fmtCell(v))}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("#files").innerHTML = d.other_files.map((f) => `<li><span>${esc(f.file)}</span><span>${f.size_kb} KB</span></li>`).join("");
}
const shortType = (t) => ({ int64: "int", float64: "float", object: "str", bool: "bool", str: "str", string: "str" }[t] || t);
const fmtCell = (v) => (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : v);

function markColumnsUsed(code) {
  const used = state.columns.filter((c) => new RegExp(`['"\\.]${c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`).test(code));
  used.forEach((c) => document.querySelectorAll(`[data-col="${CSS.escape(c)}"]`).forEach((el) => el.classList.add("is-used")));
  const n = document.querySelectorAll(".col.is-used").length;
  $("#colHint").textContent = n ? `${n} touched by the agent` : "";
}
const clearColumnsUsed = () => { document.querySelectorAll(".is-used").forEach((el) => el.classList.remove("is-used")); $("#colHint").textContent = ""; };

/* ---------------- python highlighter ---------------- */
const KW = new Set("and as assert break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(" "));
const BI = new Set("print len sum sorted round int float str list dict set range enumerate zip min max abs open isinstance type".split(" "));
function highlight(code) {
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rfb]?"(?:\\.|[^"\\\n])*"|[rfb]?'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const [t] = m;
    if (m[1]) out += `<span class="tk-com">${esc(t)}</span>`;
    else if (m[2]) out += `<span class="tk-str">${esc(t)}</span>`;
    else if (m[3]) out += `<span class="tk-num">${esc(t)}</span>`;
    else if (m[4]) out += `<span class="${BI.has(t) ? "tk-bi" : KW.has(t) ? "tk-kw" : "tk-fn"}">${esc(t)}</span>`;
    else out += KW.has(t) ? `<span class="tk-kw">${esc(t)}</span>` : esc(t);
    last = re.lastIndex;
  }
  out += esc(code.slice(last));
  return out.split("\n").map((l) => `<span class="ln">${l || " "}</span>`).join("");
}

/* ---------------- run rendering ---------------- */
const ICON_CODE = `<svg viewBox="0 0 16 16"><path d="M5.5 4L2 8l3.5 4M10.5 4L14 8l-3.5 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ICON_FILE = `<svg viewBox="0 0 16 16"><path d="M4 2h5l3 3v9H4z M9 2v3h3" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
const CARET = `<svg class="caret" viewBox="0 0 12 12"><path d="M4 2.5L7.5 6 4 9.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;

function parseArgs(a) {
  if (a && typeof a === "object") return a;
  try { return JSON.parse(a); } catch { return { raw: String(a ?? "") }; }
}
function firstJson(text) {
  const s = text.indexOf("{");
  if (s < 0) return null;
  for (let e = text.lastIndexOf("}"); e > s; e = text.lastIndexOf("}", e - 1)) {
    try { return JSON.parse(text.slice(s, e + 1)); } catch { /* keep shrinking */ }
  }
  return null;
}

function typewrite(el, text, onDone) {
  let i = 0;
  const step = Math.max(3, Math.ceil(text.length / 90));
  const tick = () => {
    i = Math.min(text.length, i + step);
    el.textContent = text.slice(0, i);
    if (i < text.length) requestAnimationFrame(tick); else onDone?.();
  };
  tick();
}

function newRun(task) {
  $("#empty")?.remove();
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="user-msg"><div class="user-bubble">${esc(task)}</div></div>
    <div class="run-meta"><span class="rm-model">${esc(state.model)}</span><span class="sep">/</span><span>${state.mode === "native" ? "native tools" : "prompt-based"}</span><span class="sep">/</span><span class="rm-steps">step 0</span><span class="timer">0.0s</span></div>
    <div class="steps"></div>`;
  $("#thread").appendChild(turn);
  return { turn, steps: $(".steps", turn), pendingTools: [], lastThink: null, llmCalls: 0, t0: performance.now() };
}

function scrollDown() { const t = $("#thread"); t.scrollTop = t.scrollHeight; }

function addThinking(run) {
  if (run.lastThink) collapse(run.lastThink);
  const el = document.createElement("div");
  el.className = "step is-think think is-live";
  el.innerHTML = `<button class="think-head" type="button">${CARET}<span class="think-label shimmer">Thinking</span><span class="think-preview"></span><span class="think-dur"></span></button><div class="think-body"></div>`;
  $(".think-head", el).addEventListener("click", () => el.classList.toggle("is-open"));
  run.steps.appendChild(el);
  run.lastThink = el;
  scrollDown();
  return el;
}
const collapse = (el) => el.classList.remove("is-open");

function fillThinking(run, el, ev) {
  let text = (ev.reasoning || "").trim();
  if (state.mode !== "native") {
    const obj = firstJson(ev.content || "");
    const thought = obj && typeof obj.thought === "string" ? obj.thought : (!obj ? ev.content : "");
    text = [text, thought].filter(Boolean).join("\n\n").trim();
  } else if (ev.has_tool_calls && ev.content?.trim()) {
    text = [text, ev.content.trim()].filter(Boolean).join("\n\n");
  }
  el.classList.remove("is-live");
  const label = $(".think-label", el);
  label.classList.remove("shimmer");
  label.textContent = "Thought";
  $(".think-dur", el).textContent = `${ev.seconds.toFixed(1)}s`;
  if (!text) { $(".think-head", el).disabled = true; $(".caret", el).style.visibility = "hidden"; label.textContent = "Decided"; return; }
  $(".think-preview", el).textContent = text.split("\n").find((l) => l.trim()) || "";
  el.classList.add("is-open");
  typewrite($(".think-body", el), text, () => { const b = $(".think-body", el); b.scrollTop = b.scrollHeight; });
}

function addToolCard(run, name, args) {
  const a = parseArgs(args);
  const el = document.createElement("div");
  el.className = "step is-tool";
  const isCode = name === "run_python" && a.code;
  const argLabel = isCode ? "" : Object.entries(a).filter(([k]) => k !== "content").map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  el.innerHTML = `<div class="tool-card">
      <div class="tool-head"><span class="tool-icon">${isCode ? ICON_CODE : ICON_FILE}</span><span class="tool-name">${esc(name)}</span><span class="tool-arg">${esc(argLabel)}</span><span class="tool-state run shimmer">running</span></div>
      ${isCode ? `<pre class="code">${highlight(a.code)}</pre>` : a.content ? `<pre class="code">${esc(a.content)}</pre>` : ""}
    </div>`;
  if (isCode) markColumnsUsed(a.code);
  run.steps.appendChild(el);
  scrollDown();
  return el;
}

function resolveToolCard(el, result) {
  const isErr = /^Error/.test(result) || /^User denied/.test(result);
  const st = $(".tool-state", el);
  st.className = `tool-state ${isErr ? "err" : "ok"}`;
  st.textContent = isErr ? "error" : "done";
  el.classList.toggle("is-error", isErr);
  const lines = result.split("\n").length;
  const obs = document.createElement("div");
  obs.className = `obs${isErr ? " is-err" : ""}`;
  obs.innerHTML = `<div class="obs-head">Output <span class="muted">${lines} line${lines === 1 ? "" : "s"}</span>${lines > 8 ? `<button class="obs-more" type="button">expand</button>` : ""}</div><pre class="obs-body">${esc(result)}</pre>`;
  $(".obs-more", obs)?.addEventListener("click", (e) => { obs.classList.toggle("is-open"); e.target.textContent = obs.classList.contains("is-open") ? "collapse" : "expand"; });
  $(".tool-card", el).appendChild(obs);
  scrollDown();
}

function addNote(run, cls, html) {
  const el = document.createElement("div");
  el.className = `step ${cls === "guard" ? "is-guard" : cls === "err" ? "is-error" : ""}`;
  el.innerHTML = `<div class="note ${cls}">${html}</div>`;
  run.steps.appendChild(el);
  scrollDown();
}

const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

function renderAnswer(text) {
  const lines = String(text).split("\n");
  const out = [];
  for (let i = 0; i < lines.length; ) {
    const line = lines[i];
    if (/^\s*\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const body = rows.filter((r) => !/^\s*\|?[\s:|-]+\|?\s*$/.test(r));
      const [head, ...rest] = body;
      const num = (c) => /^[-+]?[\d,.]+%?$/.test(c) ? ' class="num"' : "";
      out.push(`<table class="md-table"><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rest.map((r) => `<tr>${cells(r).map((c) => `<td${num(c)}>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, ""));
      out.push(`<${ordered ? "ol" : "ul"} class="md-list">${items.map((it) => `<li>${inline(it)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
    } else if (/^#{1,4}\s/.test(line)) {
      out.push(`<p class="md-h">${inline(line.replace(/^#+\s/, ""))}</p>`); i++;
    } else if (!line.trim()) {
      i++;
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*(\||[-*]\s|\d+\.\s|#)/.test(lines[i])) para.push(lines[i++]);
      out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    }
  }
  return out.join("");
}

function handle(run, ev) {
  switch (ev.kind) {
    case "llm_start":
      run.llmCalls += 1;
      $(".rm-steps", run.turn).textContent = `step ${run.llmCalls}`;
      run.pendingThink = addThinking(run);
      break;
    case "llm_end": {
      fillThinking(run, run.pendingThink, ev);
      let calls = ev.tool_calls || [];
      if (state.mode !== "native") {
        const obj = firstJson(ev.content || "");
        calls = obj && obj.tool && obj.tool !== "final_answer" ? [{ name: obj.tool, arguments: obj.args || {} }] : [];
      }
      calls.forEach((c) => run.pendingTools.push(addToolCard(run, c.name, c.arguments)));
      break;
    }
    case "rescued_tool_call_from_text":
      addNote(run, "rescue", "The model wrote its tool call as plain text instead of using <code>tool_calls</code>. The fallback parser recovered it.");
      break;
    case "action": {
      const card = run.pendingTools.shift() || addToolCard(run, ev.name, ev.args);
      resolveToolCard(card, String(ev.result));
      break;
    }
    case "repeat_guard": {
      const card = run.pendingTools.shift();
      if (card) resolveToolCard(card, "Error: you already called this exact tool with these exact arguments. Try something else.");
      addNote(run, "guard", `Loop guard: refused to run <b>${esc(ev.name)}</b> again with the exact same arguments.`);
      break;
    }
    case "final": {
      if (run.lastThink) collapse(run.lastThink);
      const stopped = ev.outcome !== "final";
      const el = document.createElement("div");
      el.className = `answer frame-corners${stopped ? " is-stopped" : ""}`;
      const secs = ((performance.now() - run.t0) / 1000).toFixed(1);
      el.innerHTML = `<span class="fc tl"></span><span class="fc tr"></span><span class="fc bl"></span><span class="fc br"></span>
        <p class="eyebrow">${stopped ? "Stopped" : "Answer"}</p>
        <div class="answer-text">${renderAnswer(ev.answer)}</div>
        <div class="answer-foot"><span><b>${run.llmCalls}</b> model calls</span><span><b>${run.steps.querySelectorAll(".is-tool").length}</b> tool calls</span><span><b>${secs}s</b></span><span>outcome <b>${esc(ev.outcome)}</b></span></div>`;
      run.turn.appendChild(el);
      scrollDown();
      break;
    }
    case "error":
      if (run.pendingThink) { const l = $(".think-label", run.pendingThink); l.classList.remove("shimmer"); l.textContent = "Failed"; }
      addNote(run, "err", esc(ev.message));
      break;
  }
}

async function run(task) {
  if (state.running || !task.trim() || !state.model) return;
  state.running = true;
  $("#send").disabled = true;
  $("#status").className = "status is-busy";
  $("#statusText").textContent = "agent running";
  clearColumnsUsed();
  const r = newRun(task);
  const timer = setInterval(() => { $(".timer", r.turn).textContent = `${((performance.now() - r.t0) / 1000).toFixed(1)}s`; }, 100);
  try {
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, provider: state.provider, model: state.model, mode: state.mode }) });
    if (!res.ok) { handle(r, { kind: "error", message: (await res.json()).error }); return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        if (chunk.startsWith("data: ")) handle(r, JSON.parse(chunk.slice(6)));
      }
    }
  } catch (e) {
    handle(r, { kind: "error", message: String(e) });
  } finally {
    clearInterval(timer);
    $(".timer", r.turn).textContent = `${((performance.now() - r.t0) / 1000).toFixed(1)}s`;
    state.running = false;
    $("#send").disabled = false;
    $("#status").className = "status is-ok";
    $("#statusText").textContent = "ollama connected";
  }
}

/* ---------------- composer ---------------- */
const input = $("#input");
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 160)}px`; });
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = input.value; input.value = ""; input.style.height = "auto"; run(t); });
$("#suggestions").addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) run(c.textContent); });

window.agentUI = { run };  // lets a screen-capture script drive a run without typing
loadModels();
loadData();
