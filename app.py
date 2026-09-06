import os
from dotenv import load_dotenv
# Load .env FIRST, before any langchain/langsmith imports, so tracing env vars exist at import time
load_dotenv()
# Explicitly surface tracing/endpoint config into os.environ (langchain reads these at import time)
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))
if os.getenv("LANGSMITH_API_KEY"): os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_PROJECT"): os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
if os.getenv("OPENAI_API_KEY"): os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_BASE_URL"): os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

import hashlib, json, re, base64
from datetime import datetime
from typing import Optional, List, Mapping, Any
import streamlit as st
import streamlit.components.v1 as components
import time
import tempfile, httpx, bcrypt, pandas as pd, numpy as np
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic.v1 import PrivateAttr
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
import rate_limit_fallback as rlf
import timeout_handling as th
import async_pipelines as ap
import streaming_tokens_retry as str_mod
from langsmith import Client as LangSmithClient

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 4
SEMANTIC_CACHE_THRESHOLD = 0.9
REAL_MODEL_TIMEOUT = 15.0  # seconds before a real (non-simulated) model call is abandoned -> TimeoutError -> Retry button
LLM_TEMPERATURE = 0.0
MAX_CONTEXT_CHARS = 12000
MAX_CONTEXT_TOKENS = 3000
MAX_STORED_CHUNK_CHARS = 500
BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
CONVERSATIONS_DIR = os.path.join(BASE_DIR, "conversations")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

class HttpxChatLLM(BaseLLM):
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    _client: Any = PrivateAttr()
    def __init__(self, api_key, base_url, model="gpt-4o-mini", temperature=0.0, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, temperature=temperature, **kwargs)
        self._client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=httpx.Timeout(120.0, connect=30.0))
    @property
    def _llm_type(self): return "httpx-openai-compatible"
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        generations = []; usage = {}
        for prompt in prompts:
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if stop: payload["stop"] = stop
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            generations.append([Generation(text=content, generation_info={"finish_reason": data["choices"][0].get("finish_reason"), "usage": usage})])
        llm_output = {"token_usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)}, "model_name": self.model}
        return LLMResult(generations=generations, llm_output=llm_output)
    def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature, "stream": True, "stream_options": {"include_usage": True}}
        if stop: payload["stop"] = stop
        with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line: continue
                if isinstance(line, bytes): line = line.decode("utf-8", errors="ignore")
                if not line.startswith("data: "): continue
                data = line[6:]
                if data.strip() == "[DONE]": break
                try: obj = json.loads(data)
                except Exception: continue
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    yield GenerationChunk(text=delta)
    @property
    def _identifying_params(self): return {"model": self.model, "base_url": self.base_url}

llm_model = HttpxChatLLM(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

def show_langsmith_trace():
    """Fetch and display the latest LangSmith trace after an LLM call."""
    try:
        ls_key = os.getenv("LANGSMITH_API_KEY", "")
        if not ls_key: return
        client = LangSmithClient(api_key=ls_key)
        runs = list(client.list_runs(project_name=os.getenv("LANGSMITH_PROJECT", "docchat"), limit=1))
        if runs:
            run = runs[0]
            with st.expander("🔍 LangSmith Trace"):
                url = run.url if hasattr(run, "url") else f"https://smith.langchain.com/runs/{run.id}"
                st.markdown(f"**Trace URL:** [{url}]({url})")
                st.markdown(f"**Run ID:** `{run.id}`")
                st.markdown(f"**Model:** {LLM_MODEL}")
                if run.start_time and run.end_time:
                    latency = (run.end_time - run.start_time).total_seconds()
                    st.markdown(f"**Latency:** {latency:.2f}s")
                if hasattr(run, "inputs") and run.inputs:
                    st.markdown("**Input:**")
                    st.text(str(run.inputs)[:500])
                if hasattr(run, "outputs") and run.outputs:
                    st.markdown("**Output:**")
                    st.text(str(run.outputs)[:500])
    except Exception as e:
        st.caption(f"Trace fetch failed: {e}")

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #fce4ec 0%, #f3e5f5 30%, #e1bee7 60%, #ce93d8 100%); }
    .main { background: transparent; }
    .stChatMessage { background: rgba(255,255,255,0.85); border-radius: 16px; padding: 12px; margin: 6px 0; box-shadow: 0 2px 12px rgba(156,39,176,0.1); border-left: 3px solid #e91e9c; }
    .upload-box { background: rgba(255,255,255,0.9); border-radius: 20px; padding: 28px; box-shadow: 0 4px 20px rgba(156,39,176,0.15); border: 1px solid rgba(156,39,176,0.1); }
    h1 { background: linear-gradient(90deg, #e91e63, #9c27b0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton > button { background: linear-gradient(90deg, #f06292, #ba68c8); color: white; border: none; border-radius: 20px; padding: 6px 18px; }
    .stButton > button:hover { background: linear-gradient(90deg, #ec407a, #ab47bc); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(156,39,176,0.3); }
    .stSidebar { background: linear-gradient(180deg, #fce4ec 0%, #f3e5f5 100%); }
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 { color: #880e4f; }
    .stSidebar .stButton > button { background: linear-gradient(90deg, #f8bbd0, #ce93d8); color: #4a148c; width: 100%; }
    .stTextInput > div > input { border-radius: 12px; border: 2px solid #e1bee7; }
    .stFileUploader { background: rgba(255,255,255,0.7); border-radius: 12px; border: 2px dashed #ce93d8; }
    .stTextArea > div > textarea { border-radius: 12px; border: 2px solid #e1bee7; }
    .stCaption { color: #6a1b9a; }
    .stExpander { background: rgba(255,255,255,0.7); border-radius: 12px; border: 1px solid rgba(156,39,176,0.1); }
</style>
""", unsafe_allow_html=True)

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}
def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f: json.dump(users, f, ensure_ascii=False, indent=2)
def is_valid_email(email): return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def show_auth_screen():
    st.title("📄 DocChat"); st.markdown("---")
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    with tab1:
        st.subheader("Login")
        le = st.text_input("Email", key="login_email")
        lp = st.text_input("Password", key="login_password", type="password")
        if st.button("Login", key="login_btn"):
            if not le or not lp: st.error("Enter email and password.")
            else:
                users = load_users(); user = users.get(le)
                if user and bcrypt.checkpw(lp.encode(), user["password"].encode()):
                    st.session_state.user_email = le; st.session_state.user_name = user.get("name", le); st.rerun()
                else: st.error("Invalid email or password.")
    with tab2:
        st.subheader("Create an Account")
        rn = st.text_input("Your name", key="reg_name")
        re_ = st.text_input("Email", key="reg_email")
        rp = st.text_input("Password", key="reg_password", type="password")
        rp2 = st.text_input("Confirm Password", key="reg_password2", type="password")
        if st.button("Register", key="register_btn"):
            if not rn or not re_ or not rp: st.error("All fields required.")
            elif not is_valid_email(re_): st.error("Invalid email.")
            elif rp != rp2: st.error("Passwords don't match.")
            elif len(rp) < 4: st.error("Password too short.")
            else:
                users = load_users()
                if re_ in users: st.error("Account exists.")
                else:
                    users[re_] = {"name": rn, "password": bcrypt.hashpw(rp.encode(), bcrypt.gensalt()).decode()}
                    save_users(users); st.success("Account created! Switch to Login.")
    st.stop()

def get_conv_file(email): return os.path.join(CONVERSATIONS_DIR, f"{re.sub(r'[^a-zA-Z0-9._-]', '_', email)}.json")
def load_user_convs(email):
    fp = get_conv_file(email)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f: return json.load(f)
        except: return []
    return []
def save_user_convs(email, convs):
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    with open(get_conv_file(email), "w", encoding="utf-8") as f: json.dump(convs, f, ensure_ascii=False, indent=2)

def archive_current():
    if "messages" not in st.session_state or not st.session_state.messages: return
    email = st.session_state.get("user_email", "")
    if not email: return
    fns = st.session_state.get("uploaded_filenames", [])
    fn_str = ", ".join(fns) if isinstance(fns, list) else str(fns)
    chunks = st.session_state.get("doc_chunks", [])
    convs = load_user_convs(email)
    cont_idx = st.session_state.get("continuing_conversation_idx")
    if cont_idx is not None and cont_idx < len(convs):
        convs[cont_idx]["messages"] = st.session_state.messages.copy()
        convs[cont_idx]["doc_chunks"] = [c[:MAX_STORED_CHUNK_CHARS] for c in chunks]
        convs[cont_idx]["filename"] = fn_str
        convs[cont_idx]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        convs.append({"id": hashlib.md5(f"{fn_str}{datetime.now().isoformat()}".encode()).hexdigest()[:8], "filename": fn_str, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": st.session_state.messages.copy(), "doc_chunks": [c[:MAX_STORED_CHUNK_CHARS] for c in chunks]})
    if len(convs) > 100: convs = convs[-100:]
    save_user_convs(email, convs)

def load_pdf(path): return PyPDFLoader(path).load()
def load_docx(path): return Docx2txtLoader(path).load()
def load_pptx(path):
    from pptx import Presentation
    prs = Presentation(path); docs = []
    for sn, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip(): texts.append(para.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        if cell.text.strip(): texts.append(cell.text.strip())
        if texts: docs.append(Document(page_content="\n".join(texts), metadata={"page": sn, "source": path}))
    return docs
def load_xlsx(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True); docs = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]; rows_text = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None]
            if cells: rows_text.append(" | ".join(cells))
        if rows_text: docs.append(Document(page_content="\n".join(rows_text), metadata={"sheet": sheet, "source": path}))
    return docs
def load_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f: return [Document(page_content=f.read(), metadata={"source": path})]
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f: return [Document(page_content=f.read(), metadata={"source": path})]

def load_file(uf):
    suffix = "." + uf.name.split(".")[-1].lower(); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f: f.write(uf.getvalue()); path = f.name
        if suffix == ".pdf": docs = load_pdf(path)
        elif suffix == ".docx": docs = load_docx(path)
        elif suffix == ".pptx": docs = load_pptx(path)
        elif suffix in [".xlsx", ".xls"]: docs = load_xlsx(path)
        elif suffix in [".txt", ".md", ".csv"]: docs = load_text(path)
        else: st.error(f"Unsupported: {suffix}"); return []
        if not docs: st.warning(f"No text from **{uf.name}**.")
        return docs
    except Exception as e:
        st.error(f"Failed to read **{uf.name}**: {e}"); return []
    finally:
        if path and os.path.exists(path):
            try: os.unlink(path)
            except: pass

def search_chunks(query, chunks, k=4):
    qw = set(query.lower().split()); scored = []
    for i, c in enumerate(chunks): scored.append((len(qw & set(c.lower().split())), i, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in scored[:k]]

# ---------- Semantic cache (threshold 0.92) + traced Retrieval/LLM pipeline ----------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        with st.spinner("Loading embedding model (first run downloads ~470MB)..."):
            _embedder = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder

def build_prompt(q, ctx):
    instr = "Think step by step and provide a thorough, detailed answer." if st.session_state.get("answer_mode") == "🧠 Thinking" else "Answer concisely and directly."
    return f"You are a helpful assistant. {instr}\n\nDocument content:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"

def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / n) if n else 0.0

def _cache_fn(x, config: RunnableConfig):
    q = x["query"]
    cache = st.session_state.setdefault("semantic_cache", [])
    best_sim = 0.0; best_idx = -1; q_emb = None
    if cache:
        q_emb = get_embedder().embed_query(q)
        for i, e in enumerate(cache):
            s = _cosine(q_emb, e.get("embedding", []))
            if s > best_sim: best_sim, best_idx = s, i
        if best_idx >= 0 and best_sim >= SEMANTIC_CACHE_THRESHOLD:
            hit = cache[best_idx]
            st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
            return {"hit": True, "similarity": best_sim, "answer": hit["answer"], "contexts": hit.get("contexts", []), "q_emb": q_emb}
    return {"hit": False, "similarity": best_sim, "q_emb": q_emb}
_cache_chain = RunnableLambda(_cache_fn).with_config({"run_name": "SemanticCache"})

_retriever = RunnableLambda(lambda x: search_chunks(x["query"], x["chunks"], x["k"])).with_config({"run_name": "Retrieval"})

def _make_streamlit_reporter(status):
    """Map structured retry/fallback events onto a live st.status widget."""
    def report(ev):
        if status is None: return
        t = ev.get("type")
        try:
            if t == "attempt":
                status.update(label=f"🤖 Calling model (attempt {ev['n']}/{ev['total']})")
            elif t == "rate_limited":
                st.write(f"⚠️ Attempt {ev['n']}: rate-limited (429). Retrying in {ev['wait']:.1f}s…")
            elif t == "exhausted":
                status.update(label="🔄 Fallback model thinking…", state="running")
            elif t == "fallback":
                st.write("Contacting fallback model…")
            elif t == "fallback_done":
                status.update(label="✅ Fallback model responded", state="complete")
            elif t == "done":
                status.update(label="✅ Answer ready", state="complete")
        except Exception:
            pass
    return report

def _make_timeout_reporter(status):
    """Map timeout events onto the same st.status widget (live waiting ticks)."""
    def report(ev):
        if status is None: return
        t = ev.get("type")
        try:
            if t == "calling":
                status.update(label=f"⏳ Calling model (timeout {ev['timeout']:.0f}s)…")
            elif t == "waiting":
                status.update(label=f"⏳ Waiting for model… {ev['elapsed']:.1f}s / {ev['timeout']:.0f}s")
            elif t == "done":
                status.update(label=f"✅ Responded in {ev['elapsed']:.1f}s", state="complete")
            elif t == "timeout":
                status.update(label=f"⏱ Timed out after {ev['timeout']:.0f}s", state="error")
        except Exception:
            pass
    return report

def _answer_fn(x, config: RunnableConfig):
    q, chunks, k = x["query"], x["chunks"], x["k"]
    status = x.get("status")
    c = _cache_chain.invoke({"query": q}, config=config)
    if c["hit"]:
        _make_streamlit_reporter(status)({"type": "done"})
        return {"answer": c["answer"], "contexts": c["contexts"], "cache_hit": True, "similarity": c["similarity"]}
    retrieved = _retriever.invoke({"query": q, "chunks": chunks, "k": k}, config=config)
    ctx = "\n\n".join(retrieved[:MAX_CONTEXT_CHARS // max(DEFAULT_CHUNK_SIZE, 100) + 1])
    if len(ctx) > MAX_CONTEXT_CHARS: ctx = ctx[:MAX_CONTEXT_CHARS] + "...[truncated]"
    pr = build_prompt(q, ctx)
    def _real_call(prompt):
        # Real model call wrapped with a 15s timeout + live waiting ticks (runs in a daemon
        # thread so the wait can be abandoned). config carries the parent callbacks so the
        # HttpxChatLLM run still traces (with token usage) under answer_question.
        return th.call_with_timeout_sync(prompt, timeout_seconds=REAL_MODEL_TIMEOUT, model_fn=lambda p: llm_model.invoke(p, config=config), on_progress=_make_timeout_reporter(status))
    def traced_primary_call(prompt):
        _attempt["n"] += 1
        try:
            return _real_call(prompt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise rlf.RateLimitError(f"429 from primary: {e}")
            raise
    def fallback_call(prompt):
        return rlf.call_fallback_model(prompt)
    a = rlf.answer_with_retry_and_fallback(pr, primary_fn=traced_primary_call, fallback_fn=fallback_call, on_progress=_make_streamlit_reporter(status)).strip()
    q_emb = c["q_emb"] or get_embedder().embed_query(q)
    cache = st.session_state.setdefault("semantic_cache", [])
    cache.append({"question": q, "answer": a, "embedding": q_emb, "contexts": retrieved})
    if len(cache) > 200: del cache[:len(cache) - 200]
    st.session_state["cache_misses"] = st.session_state.get("cache_misses", 0) + 1
    return {"answer": a, "contexts": retrieved, "cache_hit": False, "similarity": c["similarity"]}
_answer_chain = RunnableLambda(_answer_fn).with_config({"run_name": "answer_question"})

def answer_question(q, chunks, k, status=None):
    return _answer_chain.invoke({"query": q, "chunks": chunks, "k": k, "status": status})

def stream_answer(pr, status=None):
    """Real streaming from llm_model.stream() with light 429-retry on stream-start."""
    attempt = 0
    while True:
        attempt += 1
        try:
            for chunk in llm_model.stream(pr):
                txt = getattr(chunk, "text", "") or ""
                if txt:
                    yield txt
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 10:
                if status:
                    try: status.update(label=f"⏳ Rate-limited, retrying ({attempt}/10)…")
                    except Exception: pass
                time.sleep(min(2 ** attempt, 30))
                continue
            raise

def stream_answer_question(q, chunks, k, status=None):
    """Streaming answer with semantic cache + token-truncated retrieval + content fallback."""
    cache = st.session_state.setdefault("semantic_cache", [])
    st.session_state["_last_stream_sim"] = 0.0
    st.session_state["_last_stream_cached"] = False
    q_emb = None
    if cache:
        q_emb = get_embedder().embed_query(q)
        best_sim, best_idx = 0.0, -1
        for i, e in enumerate(cache):
            s = _cosine(q_emb, e.get("embedding", []))
            if s > best_sim: best_sim, best_idx = s, i
        if best_idx >= 0 and best_sim >= SEMANTIC_CACHE_THRESHOLD:
            st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
            st.session_state["_last_stream_sim"] = best_sim
            st.session_state["_last_stream_cached"] = True
            if status:
                try: status.update(label=f"⚡ Cached answer · sim {best_sim:.2f}", state="complete")
                except Exception: pass
            yield cache[best_idx]["answer"]
            return
    if q_emb is not None:
        st.session_state["_last_stream_sim"] = best_sim
    retrieved = search_chunks(q, chunks, k)
    ctx_chunks = str_mod.truncate_context_to_fit(retrieved, MAX_CONTEXT_TOKENS)
    pr = build_prompt(q, "\n\n".join(ctx_chunks))
    if status:
        try: status.update(label=f"🤖 Thinking…")
        except Exception: pass
    full = []
    for delta in stream_answer(pr, status):
        full.append(delta); yield delta
    a = "".join(full)
    if not str_mod.is_valid_response(a):
        a = str_mod.call_with_content_retry(lambda _p: llm_model.invoke(pr).strip(), pr, max_attempts=3)
        yield a
    if q_emb is None: q_emb = get_embedder().embed_query(q)
    cache.append({"question": q, "answer": a, "embedding": q_emb, "contexts": retrieved})
    if len(cache) > 200: del cache[:len(cache) - 200]
    st.session_state["cache_misses"] = st.session_state.get("cache_misses", 0) + 1
    if status:
        try: status.update(label="✅ Answer ready", state="complete")
        except Exception: pass

def preview_pdf(raw, fname):
    b64 = base64.b64encode(raw).decode()
    st.markdown(f"### 📄 {fname}")
    st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800" type="application/pdf"></iframe>', unsafe_allow_html=True)

def preview_docx(raw, fname):
    st.markdown(f"### 📄 {fname}"); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f: f.write(raw); path = f.name
        from docx import Document as D
        doc = D(path)
        for p in doc.paragraphs:
            if p.text.strip():
                s = p.style.name if p.style else "Normal"
                if "Heading 1" in s or "Title" in s: st.markdown(f"# {p.text}")
                elif "Heading 2" in s: st.markdown(f"## {p.text}")
                elif "Heading 3" in s: st.markdown(f"### {p.text}")
                elif "List" in s: st.markdown(f"- {p.text}")
                else: st.write(p.text)
        for t in doc.tables:
            rows = [[c.text for c in r.cells] for r in t.rows]
            if rows: st.table(pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows))
    finally:
        if path and os.path.exists(path): os.unlink(path)

def preview_pptx(raw, fname):
    st.markdown(f"### 📄 {fname}"); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as f: f.write(raw); path = f.name
        from pptx import Presentation
        prs = Presentation(path)
        for sn, slide in enumerate(prs.slides, 1):
            st.markdown(f"---\n**Slide {sn}:**")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        if p.text.strip(): st.write(p.text.strip())
                if shape.has_table:
                    rows = [[c.text.strip() for c in r.cells] for r in shape.table.rows]
                    if rows: st.table(pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows))
    finally:
        if path and os.path.exists(path): os.unlink(path)

def preview_xlsx(raw, fname):
    st.markdown(f"### 📄 {fname}"); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f: f.write(raw); path = f.name
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True)
        for sn in wb.sheetnames:
            ws = wb[sn]; st.markdown(f"**Sheet: {sn}**")
            data = [[str(c) if c is not None else "" for c in r] for r in ws.iter_rows(values_only=True)]
            if data: st.dataframe(pd.DataFrame(data[1:], columns=data[0]) if len(data) > 1 else pd.DataFrame(data), use_container_width=True)
    finally:
        if path and os.path.exists(path): os.unlink(path)

def preview_text(raw, fname):
    st.markdown(f"### 📄 {fname}")
    suffix = "." + fname.split(".")[-1].lower()
    try: content = raw.decode("utf-8")
    except: content = raw.decode("latin-1")
    if suffix == ".md": st.markdown(content)
    elif suffix == ".csv":
        import io; st.dataframe(pd.read_csv(io.StringIO(content)), use_container_width=True)
    else: st.text_area("File content", value=content, height=600, disabled=True)

def render_preview(fname, raw):
    suffix = "." + fname.split(".")[-1].lower()
    if suffix == ".pdf": preview_pdf(raw, fname)
    elif suffix == ".docx": preview_docx(raw, fname)
    elif suffix == ".pptx": preview_pptx(raw, fname)
    elif suffix in [".xlsx", ".xls"]: preview_xlsx(raw, fname)
    elif suffix in [".txt", ".md", ".csv"]: preview_text(raw, fname)
    else: st.warning(f"Preview not available for {suffix}.")

if "user_email" not in st.session_state: show_auth_screen()
user_email = st.session_state.user_email
user_name = st.session_state.get("user_name", user_email)

# ======================== Sidebar ========================
with st.sidebar:
    if st.session_state.get("uploaded_filenames"):
        st.markdown("### 📁 Loaded Files (click to preview)")
        for fname in st.session_state.uploaded_filenames:
            if st.button(f"📄 {fname}", key=f"file_{fname}"):
                st.session_state.preview_file = fname
                if "review_conversation" in st.session_state: del st.session_state.review_conversation
                st.rerun()
        st.divider()

    with st.expander("⚙️ Response Mode"):
        answer_mode = st.radio("", options=["⚡ Instant", "🧠 Thinking"], index=0, key="answer_mode", label_visibility="collapsed")
        if answer_mode == "⚡ Instant":
            st.caption("Quick answers for everyday questions — fast, direct, and to the point.")
        else:
            st.caption("Deep analysis for complex problems — thorough reasoning that may take longer.")
        chunk_size = DEFAULT_CHUNK_SIZE
        chunk_overlap = DEFAULT_CHUNK_OVERLAP
        k = DEFAULT_K
        _h = st.session_state.get("cache_hits", 0); _m = st.session_state.get("cache_misses", 0)
        st.caption(f"Semantic cache · {_h} hits / {_m} misses · threshold {SEMANTIC_CACHE_THRESHOLD}")
    st.divider()

    st.markdown("### 💬 Your Conversations")
    if st.button("➕ New Conversation"):
        archive_current()
        for key in ["messages", "doc_chunks", "uploaded_filenames", "processed_files", "file_raw_bytes", "applied_chunk_size", "applied_chunk_overlap", "preview_file", "review_conversation", "show_uploader"]:
            if key in st.session_state: del st.session_state[key]
        if "continuing_conversation_idx" in st.session_state: del st.session_state["continuing_conversation_idx"]
        st.rerun()
    st.divider()
    convs = load_user_convs(user_email)
    if convs:
        for idx in range(len(convs) - 1, -1, -1):
            c = convs[idx]; fn = c.get("filename", "?"); ts = c.get("timestamp", "")
            fq = next((m["content"][:40] for m in c.get("messages", []) if m["role"] == "user"), "")
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📝 {fn}\n   {ts} | Q: {fq}...", key=f"conv_{idx}"):
                    st.session_state.review_conversation = idx
                    if "preview_file" in st.session_state: del st.session_state.preview_file
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{idx}"):
                    convs.pop(idx); save_user_convs(user_email, convs); st.rerun()
        st.divider()
        if st.button("🗑️ Clear All History"):
            save_user_convs(user_email, []); st.rerun()
    else: st.caption("No conversations yet")
    st.divider()


    st.markdown(f"## 📄 DocChat")
    st.markdown(f"👤 **{user_name}**")
    st.caption(f"📧 {user_email}")
    if st.button("🚪 Logout"):
        archive_current()
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# ======================== Main ========================
st.title(f"👋 Hi, {user_name}!")
st.caption("Click ➕ to upload files, then ask me anything about them.")

if "preview_file" in st.session_state:
    fname = st.session_state.preview_file
    raw_bytes = st.session_state.get("file_raw_bytes", {})
    if st.button("← Back to chat"):
        del st.session_state.preview_file; st.rerun()
    st.divider()
    if fname in raw_bytes: render_preview(fname, raw_bytes[fname])
    else: st.warning(f"File data not found for {fname}.")

elif "review_conversation" in st.session_state:
    convs = load_user_convs(user_email)
    idx = st.session_state.review_conversation
    if idx < len(convs):
        c = convs[idx]
        st.markdown(f"### 📖 {c.get('filename', '?')} — {c.get('timestamp', '')}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to current"): del st.session_state.review_conversation; st.rerun()
        with col2:
            if st.button("▶️ Continue"):
                st.session_state.messages = c.get("messages", []).copy()
                st.session_state.doc_chunks = c.get("doc_chunks", []).copy()
                st.session_state.uploaded_filenames = c.get("filename", "").split(", ")
                st.session_state.applied_chunk_size = DEFAULT_CHUNK_SIZE
                st.session_state.applied_chunk_overlap = DEFAULT_CHUNK_OVERLAP
                st.session_state.continuing_conversation_idx = idx
                del st.session_state.review_conversation; st.rerun()
        st.divider()
        for m in c.get("messages", []):
            with st.chat_message(m["role"]): st.write(m["content"])
        st.divider(); st.markdown("### Continue asking:")
        _pending = st.session_state.pop("pending_retry_followup", None)
        if _pending:
            with st.chat_message("assistant"):
                with st.status("🤖 Thinking… (retry)", expanded=True) as _status:
                    try:
                        res = answer_question(_pending, c.get("doc_chunks", []), DEFAULT_K, status=_status); a = res["answer"]
                        st.caption("⚡ " + ("Cached" if res["cache_hit"] else "Fresh") + " answer · similarity " + str(round(res["similarity"], 2)))
                        st.write_stream(a[i:i+3] for i in range(0, len(a), 3)); st.session_state.messages.append({"role": "assistant", "content": a})
                        show_langsmith_trace()
                        convs[idx]["messages"] = st.session_state.messages.copy()
                        save_user_convs(user_email, convs)
                    except Exception as e:
                        _status.update(label="❌ Retry failed", state="error"); st.error(f"Failed: {e}")
        q = st.chat_input("Ask a follow-up...")
        if q:
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            with st.chat_message("assistant"):
                with st.status("🤖 Thinking…", expanded=True) as _status:
                    try:
                        res = answer_question(q, c.get("doc_chunks", []), DEFAULT_K, status=_status); a = res["answer"]
                        st.caption("⚡ " + ("Cached" if res["cache_hit"] else "Fresh") + " answer · similarity " + str(round(res["similarity"], 2)))
                        st.write_stream(a[i:i+3] for i in range(0, len(a), 3)); st.session_state.messages.append({"role": "assistant", "content": a})
                        show_langsmith_trace()
                        convs[idx]["messages"] = st.session_state.messages.copy()
                        save_user_convs(user_email, convs)
                        
                    except TimeoutError as te:
                        _status.update(label="⏱ Timed out", state="error"); st.warning(f"⏱ {te}")
                        if st.button("🔄 Retry", key="retry_followup_timeout"):
                            st.session_state["pending_retry_followup"] = q
                            st.rerun()
                    except Exception as e:
                        _status.update(label="❌ Failed", state="error"); st.error(f"Failed: {e}")
    else:
        del st.session_state.review_conversation; st.rerun()

else:
    if st.button("➕", help="Click to upload files"):
        st.session_state.show_uploader = not st.session_state.get("show_uploader", False)
        st.rerun()

    if st.session_state.get("show_uploader", False):
        uploaded_files = st.file_uploader("Choose from desktop", type=["pdf", "docx", "pptx", "xlsx", "xls", "txt", "md", "csv"], label_visibility="visible", accept_multiple_files=True)
    else:
        uploaded_files = None

    if not uploaded_files and not st.session_state.get("doc_chunks"):
        st.markdown("""<div class="upload-box"><h3>Get started</h3><p>Click ➕ above to upload files</p><p>Then ask any question about the documents</p><p>Your conversations are saved automatically</p></div>""", unsafe_allow_html=True)
    else:
        if uploaded_files:
            cur_names = [f.name for f in uploaded_files]
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = {}; st.session_state.doc_chunks = []
                st.session_state.uploaded_filenames = []; st.session_state.file_raw_bytes = {}
                if "messages" not in st.session_state: st.session_state.messages = []
            new_files = []
            for uf in uploaded_files:
                raw = uf.getvalue(); fh = hashlib.sha256(raw).hexdigest()
                if st.session_state.processed_files.get(uf.name) != fh: new_files.append(uf)
            old_n = set(st.session_state.processed_files.keys()); cur_n = set(cur_names)
            if old_n - cur_n:
                st.session_state.doc_chunks = []; st.session_state.processed_files = {}
                st.session_state.uploaded_filenames = []; st.session_state.file_raw_bytes = {}
                new_files = list(uploaded_files)
            if new_files:
                for uf in new_files:
                    with st.spinner(f"Reading {uf.name}..."):
                        raw = uf.getvalue(); st.session_state.file_raw_bytes[uf.name] = raw
                        docs = load_file(uf)
                        if not docs: continue
                        sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        chunks = sp.split_documents(docs)
                        st.session_state.doc_chunks.extend([d.page_content for d in chunks])
                        st.session_state.processed_files[uf.name] = hashlib.sha256(raw).hexdigest()
                        if uf.name not in st.session_state.uploaded_filenames: st.session_state.uploaded_filenames.append(uf.name)
                st.session_state.applied_chunk_size = chunk_size
                st.session_state.applied_chunk_overlap = chunk_overlap
                st.success(f"✅ File uploaded! {len(st.session_state.uploaded_filenames)} file(s) loaded. Click file names in sidebar to preview.")
                st.rerun()

        if st.session_state.get("doc_chunks"):
            loaded_fns = st.session_state.get("uploaded_filenames", [])
            if loaded_fns:
                st.success("📁 Loaded: " + ", ".join(loaded_fns))
            _lus = st.session_state.get("last_upload_stats")
            if _lus:
                st.info(f"⏱ Processed {_lus['n']} file(s) ({_lus['mode']}) in {_lus['elapsed']:.1f}s")
            ap_s = st.session_state.get("applied_chunk_size", chunk_size)
            st.success("✅ File(s) uploaded — ready to chat!")
            ap_o = st.session_state.get("applied_chunk_overlap", chunk_overlap)
            st.caption(f"Mode: {st.session_state.get('answer_mode', '⚡ Instant')} | {len(st.session_state.get('doc_chunks', []))} chunks loaded")
            for msg_idx, m in enumerate(st.session_state.messages):
                with st.chat_message(m["role"]):
                    st.write(m["content"])
                    if m["role"] == "user":
                        col_e, col_r = st.columns([1, 1])
                        with col_e:
                            if st.button("✏️ Edit", key=f"edit_{msg_idx}"):
                                st.session_state.editing_msg = msg_idx
                                st.rerun()
                        with col_r:
                            if st.button("🔄 Resend", key=f"resend_{msg_idx}"):
                                st.session_state.messages = st.session_state.messages[:msg_idx]
                                edited_q = m["content"]
                                st.session_state.messages.append({"role": "user", "content": edited_q})
                                with st.status("🤖 Thinking…", expanded=True) as _status:
                                    try:
                                        res = answer_question(edited_q, st.session_state.doc_chunks, k, status=_status); a = res["answer"]
                                        st.caption("⚡ " + ("Cached" if res["cache_hit"] else "Fresh") + " answer · similarity " + str(round(res["similarity"], 2)))
                                        st.session_state.messages.append({"role": "assistant", "content": a})
                                        st.session_state.messages.extend(after_msgs)
                                    except Exception as e:
                                        st.error(f"Failed: {e}")
                                st.rerun()
            if "editing_msg" in st.session_state:
                edit_idx = st.session_state.editing_msg
                orig_text = st.session_state.messages[edit_idx]["content"] if edit_idx < len(st.session_state.messages) else ""
                edited = st.text_area("Edit your message:", value=orig_text, key="edit_text", height=100)
                col_s, col_c = st.columns(2)
                with col_s:
                    if st.button("✅ Send edited"):
                        after_msgs = []
                        if edit_idx + 2 < len(st.session_state.messages):
                            after_msgs = st.session_state.messages[edit_idx + 2:]
                        st.session_state.messages = st.session_state.messages[:edit_idx]
                        st.session_state.messages.append({"role": "user", "content": edited})
                        del st.session_state.editing_msg
                        with st.status("🤖 Thinking…", expanded=True) as _status:
                            try:
                                res = answer_question(edited, st.session_state.doc_chunks, k, status=_status); a = res["answer"]
                                st.caption("⚡ " + ("Cached" if res["cache_hit"] else "Fresh") + " answer · similarity " + str(round(res["similarity"], 2)))
                                st.session_state.messages.append({"role": "assistant", "content": a})
                                st.session_state.messages.extend(after_msgs)
                                show_langsmith_trace()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        st.rerun()
                with col_c:
                    if st.button("❌ Cancel"):
                        del st.session_state.editing_msg
                        st.rerun()
            else:
                q = st.chat_input("Ask a question about the documents...")
                if q:
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.chat_message("user"): st.write(q)
                    with st.chat_message("assistant"):
                        _status = st.status("🤖 Thinking…", expanded=False)
                        try:
                            try:
                                a = st.write_stream(stream_answer_question(q, st.session_state.doc_chunks, k, _status))
                                if not str_mod.is_valid_response(a):
                                    raise ValueError("empty stream")
                                st.caption(f"~{str_mod.count_tokens(a)} tokens · {'⚡ Cached' if st.session_state.get('_last_stream_cached') else '✨ Fresh'} · sim {st.session_state.get('_last_stream_sim', 0.0):.2f}")
                            except Exception:
                                _status.update(label="🤖 Thinking… (retry/timeout/fallback)")
                                res = answer_question(q, st.session_state.doc_chunks, k, status=_status); a = res["answer"]
                                st.write_stream(a[i:i+3] for i in range(0, len(a), 3))
                                st.caption(("⚡ Cached" if res["cache_hit"] else "✨ Fresh") + f" · sim {res['similarity']:.2f} · fallback")
                            st.session_state.messages.append({"role": "assistant", "content": a})
                            show_langsmith_trace()
                        except Exception as e:
                            _status.update(label="❌ Failed", state="error")
                            st.error(f"Failed: {e}")