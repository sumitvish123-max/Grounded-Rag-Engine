/* Veritas AI - frontend logic (vanilla JS, no build step).
   All model/user text is inserted with textContent, never innerHTML, so
   document content or LLM output can't inject markup (XSS-safe by construction). */

// Same-origin when served by FastAPI on :8000; otherwise talk to the API directly.
const API_BASE = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
const els = {
  fileInput: $("file-input"),
  uploadBtn: $("upload-btn"),
  status: $("status"),
  statusText: $("status-text"),
  docList: $("doc-list"),
  feed: $("feed"),
  empty: $("empty"),
  question: $("question"),
  sendBtn: $("send-btn"),
};

let busy = false;
let docCount = 0;

/* ---------- Helpers ---------- */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(state, text) {
  els.status.dataset.state = state;
  els.statusText.textContent = text;
}

async function api(path, options) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch {
    throw new Error("Can't reach the server. Is the backend running on port 8000?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status}).`);
  return data;
}

function scrollToBottom() {
  els.feed.scrollTop = els.feed.scrollHeight;
}

function refreshComposer() {
  const ready = docCount > 0 && !busy;
  els.question.disabled = docCount === 0 || busy;
  els.sendBtn.disabled = !ready;
  els.question.placeholder = docCount > 0 ? "Ask a question about your documents" : "Upload a PDF to begin";
}

/* ---------- Documents ---------- */
function renderDocs(docs) {
  docCount = docs.length;
  els.docList.replaceChildren(
    ...docs.map((d) => {
      const li = el("li", "doc-item");
      const meta = el("div", "doc-meta");
      meta.append(el("div", "doc-name", d.filename), el("div", "doc-sub", `${d.pages_indexed} pages · ${d.chunks} passages`));
      meta.firstChild.title = d.filename;

      const rm = el("button", "doc-remove", "×");
      rm.type = "button";
      rm.setAttribute("aria-label", `Remove ${d.filename}`);
      rm.addEventListener("click", () => removeDoc(d.filename));

      li.append(meta, rm);
      return li;
    })
  );
  if (docs.length === 0) setStatus("idle", "Waiting for a document");
  refreshComposer();
}

async function loadDocs() {
  try {
    const { documents } = await api("/documents");
    renderDocs(documents);
    if (documents.length) setStatus("ok", `${documents.length} document${documents.length > 1 ? "s" : ""} ready`);
  } catch (err) {
    setStatus("error", err.message);
  }
}

async function removeDoc(filename) {
  try {
    await api(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    await loadDocs();
  } catch (err) {
    setStatus("error", err.message);
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("error", "Choose a PDF file.");
    return;
  }
  els.uploadBtn.disabled = true;
  setStatus("working", `Reading and indexing ${file.name}…`);

  const form = new FormData();
  form.append("file", file);
  try {
    const result = await api("/upload", { method: "POST", body: form });
    await loadDocs();
    setStatus("ok", `Indexed ${result.filename}: ${result.pages} pages, ${result.chunks} passages`);
    els.question.focus();
  } catch (err) {
    setStatus("error", err.message);
  } finally {
    els.uploadBtn.disabled = false;
    els.fileInput.value = "";
  }
}

/* ---------- Chat rendering ---------- */
function addUserMessage(text) {
  els.empty?.remove();
  const row = el("div", "msg-user");
  row.append(el("div", "bubble", text));
  els.feed.append(row);
  scrollToBottom();
}

function addTyping() {
  const row = el("div", "msg-bot");
  const dots = el("div", "typing");
  dots.setAttribute("aria-label", "Veritas is searching your documents");
  dots.append(el("span"), el("span"), el("span"));
  row.append(dots);
  els.feed.append(row);
  scrollToBottom();
  return row;
}

function buildCitations(sources) {
  const wrap = el("div", "citations");
  wrap.append(el("div", "citations-title", `Verified against ${sources.length} passage${sources.length > 1 ? "s" : ""}`));

  for (const s of sources) {
    const card = el("div", "cite");
    const head = el("div", "cite-head");
    // Distance is cosine distance (lower = closer); show it as a similarity percentage.
    const similarity = Math.max(0, Math.round((1 - s.distance) * 100));
    head.append(
      el("span", "cite-file", s.filename),
      el("span", "cite-page", `Page ${s.page}`),
      el("span", "cite-match", `${similarity}% match`)
    );
    card.append(head, el("div", "cite-snippet", `“${s.snippet}”`));
    wrap.append(card);
  }
  return wrap;
}

function renderBotMessage(row, data) {
  row.replaceChildren(el("div", "answer", data.answer));
  if (data.grounded && data.sources.length) {
    row.append(buildCitations(data.sources));
  } else {
    row.append(el("div", "ungrounded", "No supporting passage found. Nothing was guessed."));
  }
  scrollToBottom();
}

function renderBotError(row, message) {
  row.classList.add("error");
  row.replaceChildren(el("div", "answer", message));
  scrollToBottom();
}

async function sendQuestion() {
  const question = els.question.value.trim();
  if (!question || busy) return;

  busy = true;
  refreshComposer();
  addUserMessage(question);
  els.question.value = "";
  autoGrow();
  const row = addTyping();

  try {
    const data = await api("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderBotMessage(row, data);
  } catch (err) {
    renderBotError(row, err.message);
  } finally {
    busy = false;
    refreshComposer();
    els.question.focus();
  }
}

/* ---------- Events ---------- */
function autoGrow() {
  els.question.style.height = "auto";
  els.question.style.height = Math.min(els.question.scrollHeight, 160) + "px";
}

els.uploadBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => uploadFile(els.fileInput.files[0]));
els.sendBtn.addEventListener("click", sendQuestion);
els.question.addEventListener("input", autoGrow);
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

loadDocs();
