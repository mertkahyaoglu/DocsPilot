const log = document.getElementById("log");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const history = [];
const keyBar = document.getElementById("keybar");
const keyInput = document.getElementById("key");
document.getElementById("savekey").addEventListener("click", () => {
  setApiKey(keyInput.value.trim()); keyInput.value = ""; keyBar.hidden = true;
});
const MAX_TURNS = 20;           // keep in sync with DOCSPILOT_MAX_HISTORY_MESSAGES

function bubble(role, text = "") {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;           // textContent: model output is never interpreted as HTML
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

// Parse an SSE stream from fetch (EventSource can't POST).
async function* sse(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    // Normalise on the accumulated buffer: "\r\n" may be split across chunks.
    buf = (buf + value).replace(/\r\n/g, "\n");
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

function errorText(body, fallback) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg).join("; ");
  return fallback;
}

function windowed() {
  let h = history.slice(-MAX_TURNS);
  while (h.length && h[0].role !== "user") h = h.slice(1);   // must start with a user turn
  return h;
}

// API key lives in sessionStorage (per tab, gone on close). Never in localStorage.
function apiKey() { try { return sessionStorage.getItem("docspilot_key") || ""; } catch { return ""; } }
function setApiKey(v) { try { sessionStorage.setItem("docspilot_key", v); } catch { /* ignore */ } }

async function post(url, payload) {
  const headers = { "content-type": "application/json" };
  const key = apiKey();
  if (key) headers.authorization = `Bearer ${key}`;
  const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload) });
  if (res.status === 401) {
    keyBar.hidden = false;
    throw new Error("Please enter your API key above.");
  }
  if (res.status === 429) {
    const wait = res.headers.get("retry-after");
    throw new Error(`You're sending requests too quickly or have reached your usage limit.${wait ? ` Try again in ~${wait}s.` : ""}`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(errorText(body, res.statusText));
  }
  return res;
}

function formatCost(usd) {
  if (usd === null || usd === undefined) return "cost n/a";        // unpriced model
  if (usd === 0) return "$0 (cached)";
  return usd < 0.0001 ? "<$0.0001" : "$" + usd.toFixed(usd < 0.01 ? 5 : 4);
}

function feedbackButtons(requestId) {
  const wrap = document.createElement("span");
  for (const [label, rating] of [["👍", "up"], ["👎", "down"]]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.className = "thumb";
    b.setAttribute("aria-label", rating === "up" ? "Helpful" : "Not helpful");
    b.addEventListener("click", async () => {
      wrap.textContent = "Thanks for the feedback.";
      try { await post("/v1/feedback", { request_id: requestId, rating }); }
      catch { /* best effort */ }
    });
    wrap.appendChild(b);
  }
  return wrap;
}

// Render one streamed turn into `out`. Returns {answer, ok, pending}.
async function consume(res, out) {
  const state = { answer: "", ok: false, pending: null, failed: false, stepStart: 0, guarded: false };
  const status = document.createElement("div");
  status.className = "meta";
  for await (const { event, data } of sse(res)) {
    if (event === "delta") { state.answer += data.text; out.textContent = state.answer; }
    else if (event === "replace") {   // output guard withheld the streamed answer
      state.answer = ""; state.guarded = true;
      out.classList.add("error"); out.textContent = data.text;
    }
    else if (event === "reset") {   // refusal fallback: drop only the current step's text
      state.answer = state.answer.slice(0, state.stepStart); out.textContent = state.answer;
    }
    else if (event === "tool_start") { status.textContent = `Using ${data.name}…`; out.after(status); }
    else if (event === "tool_end") {
      status.remove();
      if (state.answer && !state.answer.endsWith("\n")) state.answer += "\n\n";
      state.stepStart = state.answer.length;
      out.textContent = state.answer;
    }
    else if (event === "confirmation_required") { state.pending = data; }
    else if (event === "citations") {
      const box = document.createElement("div");
      box.className = "meta sources";
      box.append("Sources: ");
      data.sources.forEach((src, i) => {
        if (i) box.append(" · ");
        const s = document.createElement("span");
        s.title = src.section;                 // textContent/title only: never innerHTML
        s.textContent = src.title;
        box.appendChild(s);
      });
      out.appendChild(box);
    }
    else if (event === "error") {
      state.answer = ""; state.failed = true;
      out.classList.add("error"); out.textContent = data.message;
    } else if (event === "done") {
      state.ok = true;
      if (data.usage) {
        const m = document.createElement("div"); m.className = "meta";
        m.textContent = `${data.model} · ${data.usage.input_tokens} in / ${data.usage.output_tokens} out · ${formatCost(data.cost_usd)} `;
        if (data.request_id) m.appendChild(feedbackButtons(data.request_id));
        out.appendChild(m);
      }
    }
    log.scrollTop = log.scrollHeight;
  }
  status.remove();
  if (!state.ok && !state.failed && !state.pending) {   // stream ended without done/error
    state.failed = true;
    out.classList.add("error"); out.textContent = "Connection lost. Please retry.";
  }
  return state;
}

// Show the exact action the server will run and let the human decide.
function confirmCard(pending, onDecision) {
  const card = document.createElement("div");
  card.className = "msg confirm";
  const title = document.createElement("strong");
  title.textContent = "Approve this action?";
  card.appendChild(title);
  for (const item of pending.items) {
    const p = document.createElement("div");
    p.textContent = item.summary;
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(item.arguments, null, 2);
    card.append(p, pre);
  }
  const row = document.createElement("div");
  row.className = "row";
  for (const [label, approve] of [["Approve", true], ["Reject", false]]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label;
    if (!approve) b.className = "secondary";
    b.addEventListener("click", () => { card.remove(); onDecision(approve); });
    row.appendChild(b);
  }
  card.appendChild(row);
  log.appendChild(card);
  log.scrollTop = log.scrollHeight;
}

let busy = false;
function setBusy(v) { busy = v; send.disabled = v; if (!v) input.focus(); }

// One user-visible turn: stream, then maybe pause for confirmation and continue.
async function runTurn(requestFn) {
  const texts = [];
  let current = bubble("assistant");
  let approved = false;
  try {
    let state = await consume(await requestFn(), current);
    texts.push(state.answer);
    while (state.pending) {
      const decision = await new Promise((resolve) => confirmCard(state.pending, resolve));
      approved = approved || decision;
      current = bubble("assistant");           // errors land here, never over earlier answers
      state = await consume(
        await post("/v1/chat/confirm",
          { pending_id: state.pending.pending_id, approve: decision }), current);
      texts.push(state.answer);
    }
    return { ok: state.ok, approved, text: texts.filter(Boolean).join("\n\n") };
  } catch (err) {
    current.classList.add("error"); current.textContent = String(err.message ?? err);
    return { ok: false, approved, text: texts.filter(Boolean).join("\n\n") };
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;
  setBusy(true); input.value = "";
  history.push({ role: "user", content: text });
  bubble("user", text);
  const { ok, approved, text: answer } =
    await runTurn(() => post("/v1/chat", { messages: windowed() }));
  // Only a completed turn becomes history; otherwise drop the user turn so roles alternate.
  // Exception: an approved action already ran, so the model must know it happened.
  if (ok && answer) history.push({ role: "assistant", content: answer });
  else if (approved) {
    history.push({ role: "assistant",
      content: answer || "(The action you approved was carried out, but my reply was interrupted.)" });
  } else history.pop();
  setBusy(false);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
