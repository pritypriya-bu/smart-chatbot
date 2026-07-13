"""
app.py — Smart Chatbot (free AI/ML, Python)

Features:
  • File upload (CSV / Excel / TSV / JSON / TXT / MD / PDF / code files)
  - Tabular files are shown as an interactive data grid
  - Sort / search / filter data via natural language chat
  - Ask any question about a text or PDF file
  - Fix errors, add/remove lines in code files (with diff + download)
  - Create files from chat: Excel / PDF / PPTX / chart / AI image

Run:
    streamlit run app.py
"""

from __future__ import annotations
import os
import json
import uuid
import datetime
import streamlit as st
import pandas as pd

from llm import LLM, LLMError, ollama_models
from loaders import load_file
from dataops import make_plan, apply_plan, data_summary
from codeops import ask_code, make_diff, line_stats, lang_for_ext
from creators import (detect_creation, create_excel, create_pdf,
                      create_pptx, create_chart, create_image,
                      _refine_image_prompt)
from weather import weather_summary, is_weather_query
from websearch import web_search, is_search_query
from rag import RagIndex, HAS_EMBEDDINGS as rag_module_has_embeddings

st.set_page_config(page_title="Smart Chatbot", page_icon="🤖", layout="wide")

# ---- light styling -------------------------------------------------------
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1100px; padding-bottom: 6rem;}
  .stChatMessage {font-size: 0.95rem; border-radius: 14px;}
  div[data-testid="stMetricValue"] {font-size: 1.1rem;}

  /* tidy sidebar buttons */
  section[data-testid="stSidebar"] button {border-radius: 8px;}
  section[data-testid="stSidebar"] .stButton p {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  hr {margin: 0.6rem 0;}
</style>
""", unsafe_allow_html=True)


# ---- session state -------------------------------------------------------
# Each conversation has its own full state (messages + loaded file + data + code).
# "Live" state lives in these keys; on switch we snapshot the previous chat and restore the new one.
LIVE_KEYS = ["messages", "file", "df_view", "df_original",
             "code_current", "code_history", "created", "_file_sig"]

# Chats are persisted to disk here so they survive app restarts
CHATS_DIR = os.path.join(os.path.expanduser("~"), ".smart_chatbot")
CHATS_FILE = os.path.join(CHATS_DIR, "chats.json")


def save_chats(conversations):
    """Persist only chat text (id/title/messages) to disk - not loaded data/files."""
    try:
        os.makedirs(CHATS_DIR, exist_ok=True)
        light = [{"id": c["id"], "title": c.get("title", "New chat"),
                  "custom": c.get("custom", False),
                  "messages": c.get("messages") or []}
                 for c in conversations]
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False)
    except Exception:
        pass   # never let a save failure crash the app


def load_chats():
    """Load previously saved chats from disk into full conversation structures."""
    try:
        with open(CHATS_FILE, encoding="utf-8") as f:
            light = json.load(f)
    except Exception:
        return []
    convs = []
    for c in light:
        base = _new_conv()
        base["id"] = c.get("id", base["id"])
        base["title"] = c.get("title", "New chat")
        base["custom"] = c.get("custom", False)
        base["messages"] = c.get("messages") or []
        convs.append(base)
    return convs


def _new_conv():
    return {"id": uuid.uuid4().hex[:8], "title": "New chat", "custom": False,
            "messages": [], "file": None, "df_view": None, "df_original": None,
            "code_current": None, "code_history": [], "created": None, "_file_sig": None}


def _conv_title(c):
    """Manually renamed -> wahi title; warna pehle user message se auto."""
    if c.get("custom") and c.get("title"):
        return c["title"]
    msgs = st.session_state.messages if c["id"] == st.session_state.get("active_id") else c["messages"]
    first = next((m["content"] for m in (msgs or []) if m.get("role") == "user"), None)
    if not first:
        return "New chat"
    first = first.strip().replace("\n", " ")
    return first[:38] + ("…" if len(first) > 38 else "")


def _snapshot_active():
    """Snapshot the live state into the active conversation (before switching)."""
    cid = st.session_state.get("active_id")
    for c in st.session_state.conversations:
        if c["id"] == cid:
            for k in LIVE_KEYS:
                c[k] = st.session_state.get(k)
            if not c.get("custom"):           # manually rename kiya to mat badlo
                c["title"] = _conv_title(c)
            return


def _activate(cid):
    """Conversation cid ko live state me laao."""
    for c in st.session_state.conversations:
        if c["id"] == cid:
            for k in LIVE_KEYS:
                st.session_state[k] = c.get(k)
            st.session_state.active_id = cid
            return


def _start_new_conversation():
    _snapshot_active()
    c = _new_conv()
    st.session_state.conversations.insert(0, c)   # newest sabse upar
    _activate(c["id"])
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("file", None)
    ss.setdefault("df_view", None)
    ss.setdefault("df_original", None)
    ss.setdefault("code_current", None)
    ss.setdefault("code_history", [])
    ss.setdefault("created", None)
    ss.setdefault("conversations", [])
    ss.setdefault("active_id", None)
    ss.setdefault("renaming_id", None)
    ss.setdefault("kb_index", None)       # RAG index (multiple docs)
    ss.setdefault("kb_docs", [])          # KB me kaunse docs hain
    ss.setdefault("kb_sig", None)         # signature of files currently in the KB
    if not ss.conversations:              # first run - load previously saved chats from disk
        loaded = load_chats()
        if loaded:
            ss.conversations = loaded
            _activate(loaded[0]["id"])
        else:
            c = _new_conv()
            ss.conversations.append(c)
            ss.active_id = c["id"]


def _persist():
    """Snapshot the active chat and save all conversations to disk."""
    _snapshot_active()
    save_chats(st.session_state.conversations)


init_state()


def _cb_new_chat():
    _start_new_conversation()
    save_chats(st.session_state.conversations)


def _cb_switch(cid):
    if cid != st.session_state.active_id:
        _snapshot_active()
        save_chats(st.session_state.conversations)
        _activate(cid)
        st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def _cb_delete(cid):
    _snapshot_active()
    convs = [c for c in st.session_state.conversations if c["id"] != cid]
    st.session_state.conversations = convs
    if not convs:                                  # deleted the last chat - start a new one
        c = _new_conv()
        st.session_state.conversations.append(c)
        _activate(c["id"])
        st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    elif st.session_state.active_id == cid:        # active delete hui -> doosri par jao
        _activate(convs[0]["id"])
        st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    save_chats(st.session_state.conversations)


def _cb_rename_start(cid):
    st.session_state["renaming_id"] = cid


def _cb_rename_cancel(cid):
    st.session_state["renaming_id"] = None
    st.session_state.pop(f"rename_{cid}", None)


def _cb_rename_save(cid):
    new = (st.session_state.get(f"rename_{cid}", "") or "").strip()
    for c in st.session_state.conversations:
        if c["id"] == cid and new:
            c["title"] = new[:60]
            c["custom"] = True
            break
    st.session_state["renaming_id"] = None
    st.session_state.pop(f"rename_{cid}", None)
    save_chats(st.session_state.conversations)


def _pick(items, name):
    for i, it in enumerate(items):
        if name in it:
            return i
    return 0


# ---- sidebar: chats + language + model config ---------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.25rem;font-weight:700;letter-spacing:.3px;'>"
        "🤖 Smart Chatbot</div>"
        "<div style='font-size:.75rem;opacity:.6;margin-bottom:.6rem;'>"
        "v1.6 · free AI stack</div>",
        unsafe_allow_html=True,
    )
    st.button("🆕 New chat", on_click=_cb_new_chat, use_container_width=True,
              type="primary")

    # ---- conversation history list (switch / rename / delete) ----
    st.caption("CHATS")
    for c in st.session_state.conversations:
        if st.session_state.get("renaming_id") == c["id"]:
            # rename mode: text box + save/cancel
            st.text_input("Rename chat", value=_conv_title(c),
                          key=f"rename_{c['id']}", label_visibility="collapsed")
            rc1, rc2 = st.columns(2)
            rc1.button("Save", key=f"save_{c['id']}",
                       on_click=_cb_rename_save,
                       args=(c["id"],), use_container_width=True)
            rc2.button("Cancel", key=f"cancel_{c['id']}",
                       on_click=_cb_rename_cancel,
                       args=(c["id"],), use_container_width=True)
        else:
            is_active = c["id"] == st.session_state.active_id
            label = ("● " if is_active else "") + _conv_title(c)
            col_chat, col_menu = st.columns([5, 1])
            col_chat.button(label, key=f"conv_{c['id']}", on_click=_cb_switch,
                            args=(c["id"],), use_container_width=True)
            with col_menu:
                # 3-dot popover menu (ChatGPT/Claude style)
                with st.popover("⋯", use_container_width=True):
                    st.button("✏️  Rename", key=f"edit_{c['id']}",
                              on_click=_cb_rename_start, args=(c["id"],),
                              use_container_width=True)
                    st.button("🗑️  Delete", key=f"del_{c['id']}",
                              on_click=_cb_delete, args=(c["id"],),
                              use_container_width=True)

    st.divider()

    # ---- language toggle (always visible - most used control) ----
    lang_choice = st.radio("🌐 Language", ["English", "हिंदी"],
                           horizontal=True, key="cfg_lang")
    LANG = "Hindi" if lang_choice == "हिंदी" else "English"

    st.toggle("🧠 Agent mode", key="cfg_agent",
              help=("Multi-part requests ko steps me todta hai aur tools "
                    "sequence me chalata hai. Har request par kuch extra "
                    "LLM calls lagti hain."
                    if LANG == "Hindi" else
                    "Breaks multi-part requests into steps and runs tools "
                    "in sequence (e.g. \"get weather for 3 cities, put it in "
                    "Excel, and chart it\"). Uses a couple of extra LLM calls "
                    "per request."))

    # ---- voice input (browser Web Speech API, no API key) ----
    try:
        from streamlit_mic_recorder import speech_to_text
        st.caption("🎙️ " + ("Bolo (mic dabao)" if LANG == "Hindi"
                            else "Voice input"))
        _mic_lang = "hi-IN" if LANG == "Hindi" else "en-US"
        _spoken = speech_to_text(
            language=_mic_lang,
            start_prompt="🎤  " + ("Bolna shuru karo" if LANG == "Hindi"
                                    else "Tap to speak"),
            stop_prompt="⏹  " + ("Ruko" if LANG == "Hindi" else "Stop"),
            just_once=True,
            use_container_width=True,
            key=f"stt_{st.session_state.active_id}",
        )
        if _spoken:
            st.session_state["_voice_prompt"] = _spoken
    except ImportError:
        pass

    # ---- model settings (collapsed into an expander to reduce clutter) ----
    with st.expander("⚙️ AI model", expanded=False):
        provider = st.selectbox(
            "Provider (all free)",
            ["ollama", "groq", "gemini"],
            help="Ollama = 100% local, no key. Groq/Gemini = free tier, one API key.",
            key="cfg_provider",
        )

        model = None
        groq_key = gemini_key = None
        ollama_host = "http://localhost:11434"

        if provider == "ollama":
            ollama_host = st.text_input("Ollama host", "http://localhost:11434",
                                        key="cfg_ollama_host")
            installed = ollama_models(ollama_host)
            if installed:
                model = st.selectbox("Model", installed,
                                     index=_pick(installed, "qwen2.5-coder"),
                                     key="cfg_ollama_model")
            else:
                model = st.text_input("Model", "qwen2.5-coder",
                                      key="cfg_ollama_model_txt")
                st.caption("⚠️ Ollama not detected. Install from ollama.com, "
                           "then `ollama pull qwen2.5-coder`")
        elif provider == "groq":
            groq_key = st.text_input("GROQ_API_KEY", type="password",
                                     help="https://console.groq.com/keys",
                                     key="cfg_groq_key")
            model = st.text_input("Model", "llama-3.3-70b-versatile",
                                  key="cfg_groq_model")
        elif provider == "gemini":
            gemini_key = st.text_input("GEMINI_API_KEY", type="password",
                                       help="https://aistudio.google.com/apikey",
                                       key="cfg_gemini_key")
            model = st.text_input("Model", "gemini-2.0-flash", key="cfg_gemini_model")

    # ---- provider readiness indicator ----
    _ready = (provider == "ollama") or \
             (provider == "groq" and bool(groq_key)) or \
             (provider == "gemini" and bool(gemini_key))
    st.caption(("🟢 " if _ready else "🔴 ") +
               f"{provider} · {model or '-'}" +
               ("" if _ready else "  — add your API key above"))

    with st.expander("🔎 Web search (optional)", expanded=False):
        tavily_key = st.text_input(
            "TAVILY_API_KEY (optional)", type="password",
            help="Get a free key at https://tavily.com. Leave blank to use DuckDuckGo instead.",
            key="cfg_tavily_key",
        )
        st.caption("Powers live queries: news, scores, prices. "
                   "DuckDuckGo (no key) is used when this is blank.")

    st.divider()
    st.subheader("📂 File")
    up = st.file_uploader(
        "Upload a file",
        type=["csv", "tsv", "xlsx", "xls", "json", "txt", "md", "log",
              "pdf", "py", "js", "ts", "java", "c", "cpp", "cs", "go",
              "rb", "php", "rs", "sql", "sh", "html", "css", "jsx", "tsx"],
        key=f"uploader_{st.session_state.get('uploader_key', 0)}",
    )

    st.subheader("📚 Knowledge base (RAG)")
    _rag_badge = ("🟢 semantic (embeddings)" if rag_module_has_embeddings
                  else "🟡 keyword (TF-IDF) — install `chromadb sentence-transformers` for semantic")
    st.caption(_rag_badge)
    st.caption("Upload multiple documents, then ask questions across them. "
               "Leave the single-file uploader empty.")
    kb_files = st.file_uploader(
        "Add documents",
        type=["txt", "md", "log", "pdf", "csv", "tsv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        key="kb_uploader",
    )
    if st.session_state.get("kb_docs"):
        st.caption("📗 In KB: " + ", ".join(st.session_state.kb_docs))
        if st.button("🧹 Clear knowledge base", use_container_width=True):
            st.session_state.kb_index = None
            st.session_state.kb_docs = []
            st.session_state.kb_sig = None
            st.rerun()


def t(en, hi):
    """Choose string by current language."""
    return hi if LANG == "Hindi" else en


def with_lang(system):
    """Append a language directive to a system prompt."""
    return system + f"\n\nIMPORTANT: Write your entire response in {LANG}."


def _chat_with_context(llm, system, temperature=0.4, limit=12):
    """Send the system prompt plus recent conversation history (short-term memory).

    The current user prompt is already the last item in ``st.session_state.messages``.
    """
    msgs = [{"role": "system", "content": system}]
    for m in st.session_state.messages[-limit:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return llm.chat(msgs, temperature)


def get_llm():
    return LLM(provider=provider, model=model, ollama_host=ollama_host,
               groq_key=groq_key, gemini_key=gemini_key)


# ---- handle new upload ---------------------------------------------------
def handle_upload(uploaded):
    sig = (uploaded.name, uploaded.size)
    if st.session_state.get("_file_sig") == sig:
        return
    data = uploaded.getvalue()
    try:
        loaded = load_file(uploaded.name, data)
    except Exception as e:
        st.error(f"Could not load the file: {e}")
        return
    st.session_state.file = loaded
    st.session_state._file_sig = sig
    if loaded["kind"] == "table":
        st.session_state.df_original = loaded["df"].copy()
        st.session_state.df_view = loaded["df"].copy()
        st.session_state.code_current = None
    elif loaded["kind"] == "code":
        st.session_state.code_current = loaded["text"]
        st.session_state.code_history = []
        st.session_state.df_view = None
    else:  # text/pdf
        st.session_state.code_current = None
        st.session_state.df_view = None

    kind_label = {"table": "data table", "code": "code file", "text": "text/PDF"}[loaded["kind"]]
    loaded_msg = t(f"✅ **{uploaded.name}** loaded ({kind_label}). ",
                   f"✅ **{uploaded.name}** load ho gayi ({kind_label}). ")
    st.session_state.messages.append({
        "role": "assistant",
        "content": loaded_msg + _hint(loaded["kind"]),
    })


def _hint(kind):
    if kind == "table":
        return t("Now type things like: *“sort by price”*, *“rows where city = Mumbai”*, "
                 "*“find John”*, *“add a column total = price*quantity”*, *“add 10 records”*, "
                 "or any question like *“what's the average salary?”*",
                 "Now type things like: *\"sort by price\"*, *\"rows where city = Mumbai\"*, "
                 "*\"find John\"*, *\"add a column total = price*quantity\"*, "
                 "*\"add 10 records\"*, or any question like *\"what's the average salary?\"*")
    if kind == "code":
        return t("Try: *“fix the errors”*, *“add error handling”*, *“what does this function do?”*, "
                 "*“remove unused imports”*.",
                 "Try: *\"fix the errors\"*, *\"add error handling\"*, "
                 "*\"what does this function do?\"*, *\"remove unused imports\"*.")
    return t("Ask any question about this text/PDF and I'll answer.",
             "Ask any question about this text/PDF - I will answer.")


if up is not None:
    handle_upload(up)


# ---- build knowledge base (RAG) when KB files change ----
def _doc_text(name, data):
    loaded = load_file(name, data)
    if loaded["kind"] == "table":
        return loaded["df"].to_csv(index=False)
    return loaded.get("text") or ""


if kb_files:
    sig = tuple((f.name, f.size) for f in kb_files)
    if st.session_state.get("kb_sig") != sig:
        docs = []
        for f in kb_files:
            try:
                docs.append({"name": f.name, "text": _doc_text(f.name, f.getvalue())})
            except Exception as e:
                st.warning(f"Could not load {f.name}: {e}")
        idx = RagIndex()
        idx.build(docs)
        st.session_state.kb_index = idx
        st.session_state.kb_docs = [d["name"] for d in docs]
        st.session_state.kb_sig = sig


# ====================== MAIN LAYOUT =======================================
st.title("🤖 Smart Chatbot")
st.caption(t(
    "Free AI · Understands files · Data grid · Code fix · Creates Excel/PDF/PPTX/chart/image",
    "Free AI · File samajhta hai · Data grid · Code fix · Excel/PDF/PPTX/chart/image banata hai"
))

file = st.session_state.file

# ---------- welcome card (first run, nothing loaded, no chat yet) ----------
if file is None and not st.session_state.messages and not st.session_state.get("kb_docs"):
    with st.container(border=True):
        st.markdown(t(
            "**👋 Welcome!** Try one of these:\n"
            "- 📂 Upload a **CSV/Excel** in the sidebar, then: *\"sort by price\"* or *\"add a column total = price × qty\"*\n"
            "- 🐞 Upload a **code file**, then: *\"fix the errors\"*\n"
            "- 📚 Add PDFs to the **Knowledge base**, then ask questions across them\n"
            "- 🎨 No file needed: *\"create an image of a mountain sunrise\"*, "
            "*\"weather in Mumbai\"*, *\"latest AI news\"*, "
            "*\"create a 5-slide presentation on Python\"*",
            "**👋 Welcome!** Ye try karo:\n"
            "- 📂 Sidebar me **CSV/Excel** upload karo, phir: *\"price se sort karo\"* ya *\"column total = price × qty add karo\"*\n"
            "- 🐞 **Code file** upload karo, phir: *\"errors fix karo\"*\n"
            "- 📚 **Knowledge base** me PDFs daalo, phir unse sawaal poocho\n"
            "- 🎨 Bina file ke bhi: *\"create an image of a mountain sunrise\"*, "
            "*\"Mumbai ka weather\"*, *\"latest AI news\"*, "
            "*\"Python par 5-slide presentation banao\"*",
        ))

# ---------- DATA GRID (tabular files) ----------
if file and file["kind"] == "table" and st.session_state.df_view is not None:
    dfv = st.session_state.df_view
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (view)", len(dfv))
    c2.metric("Columns", len(dfv.columns))
    c3.metric("Total rows", len(st.session_state.df_original))

    st.dataframe(dfv, use_container_width=True, height=380)

    cdl, crs = st.columns([1, 1])
    cdl.download_button("⬇️ Current view (CSV)",
                        dfv.to_csv(index=False).encode("utf-8"),
                        file_name=f"view_{file['name'].rsplit('.',1)[0]}.csv",
                        mime="text/csv")
    if crs.button("↺ Reset view"):
        st.session_state.df_view = st.session_state.df_original.copy()
        st.rerun()

# ---------- CODE VIEW ----------
elif file and file["kind"] == "code" and st.session_state.code_current is not None:
    lang = lang_for_ext(file["ext"]) or "text"
    st.subheader(f"📝 {file['name']}")
    st.code(st.session_state.code_current, language=lang, line_numbers=True)
    st.download_button("⬇️ Current code download",
                       st.session_state.code_current.encode("utf-8"),
                       file_name=file["name"])
    if st.session_state.code_history and st.button("↶ Undo last change"):
        st.session_state.code_current = st.session_state.code_history.pop()
        st.rerun()

# ---------- TEXT / PDF VIEW ----------
elif file and file["kind"] == "text":
    with st.expander(f"📄 {file['name']} — content preview", expanded=False):
        st.text(file["text"][:5000] + ("\n\n...[truncated]" if len(file["text"]) > 5000 else ""))


# ---------- CREATED FILE (Excel/PDF/PPTX/chart/image) ----------
created = st.session_state.created
if created:
    st.subheader("📦 Created file")
    if created["kind"] == "image":
        st.image(created["bytes"], caption=created.get("caption", ""),
                 use_container_width=True)
    cdc1, cdc2 = st.columns([1, 3])
    cdc1.download_button(f"⬇️ {created['name']}", created["bytes"],
                         file_name=created["name"], mime=created["mime"])
    if cdc2.button("✖️ Clear created file"):
        st.session_state.created = None
        st.rerun()


st.divider()

# ---------- CHAT HISTORY ----------
def render_message(role, content):
    """Assistant messages on the left, user messages pushed to the right."""
    if role == "user":
        _, right = st.columns([1, 3])
        with right:
            with st.chat_message("user"):
                st.markdown(content)
    else:
        left, _ = st.columns([3, 1])
        with left:
            with st.chat_message("assistant"):
                st.markdown(content)


for m in st.session_state.messages:
    render_message(m["role"], m["content"])


# ====================== CHAT HANDLER ======================================
def route_table(prompt, llm):
    """Tabular file route: try a structured data-op plan first, then fall back to Q&A."""
    df = st.session_state.df_view
    plan = make_plan(llm, prompt, df, lang=LANG)
    action = (plan.get("action") or "answer").lower()

    # These actions change the DATA structure - always apply them to the ORIGINAL
    structural = ("add", "add_column", "delete_column", "rename_column")
    if action in structural:
        result, msg, modified = apply_plan(st.session_state.df_original, plan,
                                           user_prompt=prompt)
        if modified:
            st.session_state.df_original = result
            st.session_state.df_view = result.copy()
        expl = plan.get("explanation") or msg or "Done."
        # If the guard rail rejected an unsupported formula, message is the
        # helpful explanation - don't wrap it in the LLM's (wrong) explanation.
        if not modified and action == "add_column":
            return msg
        return f"{expl}\n\n_{msg or ''}_"

    if action in ("search", "filter", "sort", "reset"):
        base = st.session_state.df_view if action in ("search", "filter", "sort") \
            else st.session_state.df_original
        result, msg, modified = apply_plan(base, plan, user_prompt=prompt)
        if modified:
            st.session_state.df_view = result
        expl = plan.get("explanation") or msg or "Done."
        return f"{expl}\n\n_{msg or ''}_"

    # answer: data summary + poori conversation (memory) ke saath poochho
    summary = data_summary(df)
    sys = with_lang("You are a data analyst. Answer using the table data below, and use the "
                    "earlier conversation for context (remember what the user told you). "
                    "Be concise and show numbers. If a calculation is needed, do it.\n\n"
                    f"Table summary:\n{summary}")
    return _chat_with_context(llm, sys, temperature=0.2)


def route_code(prompt, llm):
    code = st.session_state.code_current
    expl, new_code = ask_code(llm, code, file["ext"], prompt, lang=LANG)
    if new_code and new_code.strip() and new_code.strip() != code.strip():
        added, removed = line_stats(code, new_code)
        diff = make_diff(code, new_code, file["name"])
        st.session_state.code_history.append(code)
        st.session_state.code_current = new_code
        changes = t(f"**Changes:** +{added} / −{removed} lines.",
                    f"**Badlaav:** +{added} / −{removed} lines.")
        body = expl + "\n\n" + changes
        if diff:
            body += f"\n\n```diff\n{diff[:6000]}\n```"
        body += t("\n\n_Code panel updated. You can download / undo above._",
                  "\n\n_The code panel has been updated. You can download or undo it above._")
        return body
    return expl or t("No change seemed necessary.", "Koi change zaroori nahi laga.")


def route_text(prompt, llm):
    text = file["text"]
    ctx = text if len(text) < 30000 else text[:30000] + "\n...[truncated]"
    sys = with_lang("You are a helpful assistant. Use the document below to answer, and remember "
                    "earlier parts of this conversation. If the answer isn't in the document, "
                    f"say so. Be concise.\n\nDocument ({file['name']}):\n{ctx}")
    return _chat_with_context(llm, sys, temperature=0.3)


def is_datetime_query(prompt):
    p = prompt.lower()
    keys = ["today's date", "today date", "current date", "what date", "what is the date",
            "date today", "which day is", "what day is", "current time", "what time is",
            "what's the time", "aaj ki date", "aaj ki tareekh", "aaj ki tarikh",
            "aaj kaunsa din", "aaj kya din", "abhi time", "abhi kya time",
            "time kya hai", "kya time"]
    return any(k in p for k in keys)


def route_rag(prompt, llm):
    """If the KB is relevant, answer from documents; for tasks (MCQs/summary) use the whole doc."""
    idx = st.session_state.get("kb_index")
    p = prompt.lower()
    task_words = ["mcq", "quiz", "question", "summary", "summarize", "notes",
                  "key point", "make questions", "flashcard", "explain the document",
                  "explain this document", "important points"]
    wants_task = any(w in p for w in task_words)

    # ---- TASK (mcq/summary/notes) ----
    if wants_task and idx and idx.chunks:
        # Is the user explicitly referring to the documents (or does the topic match)?
        refers_doc = any(w in p for w in
                         ["document", "the doc", "this doc", "the pdf", "the file",
                          "from it", "in it", "in the file", "in this file", "above",
                          "attached", "uploaded", "knowledge base"])
        for fname in st.session_state.get("kb_docs", []):
            base = fname.rsplit(".", 1)[0].lower()
            if base and base in p:
                refers_doc = True
        probe = idx.search(prompt, k=3)
        rel = probe[0]["score"] if probe else 0.0

        if refers_doc or rel >= 0.05:
            # Build the answer from the whole document (broad context)
            budget, parts, total = 8000, [], 0
            for c in idx.chunks:
                if total + len(c["text"]) > budget:
                    break
                parts.append(f"[From {c['source']}]\n{c['text']}")
                total += len(c["text"])
            context = "\n\n".join(parts)
            used = ", ".join(sorted({c["source"] for c in idx.chunks}))
            sys = with_lang("Create exactly what the user asked (MCQs, summary, notes, etc.) using "
                            "ONLY the document content below. Base every item on this content — do "
                            "NOT use outside/general knowledge. Do NOT write a 'Sources' line.\n\n"
                            f"Document content:\n{context}")
            ans = _chat_with_context(llm, sys, temperature=0.3)
            ans = "\n".join(ln for ln in ans.splitlines() if "sources:" not in ln.lower()).rstrip()
            return ans + f"\n\n_📚 Sources: {used}_"
        # Topic is unrelated to the KB (e.g. 'python mcq' with a kerala.pdf) - use general knowledge
        sys = with_lang("You are a helpful assistant. Create exactly what the user asked using "
                        "your own knowledge. (A knowledge base is loaded but this topic isn't "
                        "related to it, so do not use or cite it.)")
        return _chat_with_context(llm, sys, temperature=0.4)

    # ---- normal question -> query se relevant chunks dhoondho ----
    hits = idx.search(prompt, k=6) if idx else []
    top = hits[0]["score"] if hits else 0.0

    # Off-topic / greeting / general knowledge - answer normally, skip RAG.
    # BUT: if this looks like a live-info question, use web search instead of
    # asking the LLM (which doesn't have real-time knowledge).
    if not hits or top < 0.05:
        if is_search_query(prompt):
            return route_search(prompt, llm)
        sys = with_lang("You are a friendly, knowledgeable assistant. Answer the question "
                        "normally using your own knowledge and the conversation so far. "
                        "(A document knowledge base is loaded, but this question isn't about it.)")
        return _chat_with_context(llm, sys, temperature=0.5)

    context = "\n\n".join(f"[From {h['source']}]\n{h['text']}" for h in hits)
    used = ", ".join(sorted({h["source"] for h in hits}))
    sys = with_lang("Answer the user's question using ONLY the knowledge-base excerpts below. "
                    "Quote exact lines for definitions. Do NOT write a 'Sources' line yourself. "
                    f"If the excerpts don't have the answer, say so.\n\nExcerpts:\n{context}")
    ans = _chat_with_context(llm, sys, temperature=0.3)
    ans = "\n".join(ln for ln in ans.splitlines() if "sources:" not in ln.lower()).rstrip()
    return ans + f"\n\n_📚 Sources: {used}_"


def route_datetime(prompt):
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %d %B %Y")
    time_str = now.strftime("%I:%M %p")
    return t(f"📅 Today is **{date_str}**.\n\n🕐 Current time: **{time_str}** (your device's time).",
             f"📅 Aaj hai **{date_str}**.\n\n🕐 Abhi ka time: **{time_str}** (aapke device ka time).")


def route_search(prompt, llm):
    """Live internet search - feed results to the LLM for a natural answer."""
    ok, results, src = web_search(prompt, tavily_key=tavily_key, max_results=5)
    if not ok:
        return f"❌ {results}"
    sys = with_lang("You are a helpful assistant. Answer the user's question using ONLY the web "
                    "search results below. Be accurate and concise, and mention the source "
                    "site(s). If the results don't contain the answer, say so honestly.\n\n"
                    f"Web search results (via {src}):\n{results}")
    return _chat_with_context(llm, sys, temperature=0.3)


def route_weather(prompt, llm):
    """Live weather: city nikaalo -> Open-Meteo se data -> LLM se natural jawab."""
    city = ""
    j = llm.ask_json('Extract the city or place name from this weather question. '
                     'Return ONLY JSON: {"city": "..."} or {"city": ""} if none is mentioned.',
                     prompt)
    if isinstance(j, dict):
        city = (j.get("city") or "").strip()
    if not city:
        return t("Which city's weather would you like? Please tell me the city name.",
                 "Aap kis city ka weather chahte ho? City ka naam batao.")

    ok, wx = weather_summary(city)
    if not ok:
        return f"❌ {wx}"

    sys = with_lang("You are a helpful assistant. Use the live weather data below to answer the "
                    "user's question naturally and briefly. Do not invent numbers beyond the "
                    f"data.\n\n{wx}")
    return _chat_with_context(llm, sys, temperature=0.3)


def route_create(prompt, llm, target):
    """Create a new file. The result is stored in ``st.session_state.created``."""
    cur_df = st.session_state.df_view if st.session_state.df_view is not None else None
    # text/code file ho to uska text; table ho to CSV; warna recent chat (jaise
    # "convert these questions to pdf" -> pichle jawab me jo MCQs the).
    if file and file["kind"] in ("text", "code"):
        ctx_text = file["text"]
    elif cur_df is not None:
        ctx_text = "Table data (CSV):\n" + cur_df.head(200).to_csv(index=False)
    else:
        recent = st.session_state.messages[-6:]
        ctx_text = "\n\n".join(f"{m['role']}: {m['content']}"
                               for m in recent if m.get("content")) or None

    if target == "excel":
        b, name, msg = create_excel(llm, prompt, df=cur_df)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        kind = "file"
    elif target == "pdf":
        b, name, msg = create_pdf(llm, prompt, context_text=ctx_text)
        mime, kind = "application/pdf", "file"
    elif target == "pptx":
        b, name, msg = create_pptx(llm, prompt)
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        kind = "file"
    elif target == "chart":
        b, name, msg = create_chart(llm, prompt, cur_df)
        mime, kind = "image/png", "image"
    elif target == "image":
        # Follow-up ("another image", etc.) - reuse the previous image's subject
        p_low = prompt.lower().strip()
        follow_up_only = (
            len(prompt.split()) <= 4
            and any(w in p_low for w in ["another", "aother", "one more", "ek aur",
                                          "again", "same", "one"])
        )
        prev = st.session_state.get("last_image_subject")
        if follow_up_only and prev:
            refined = prev
        else:
            refined = _refine_image_prompt(llm, prompt, context_text=ctx_text)
        b, name, msg = create_image(prompt, subject_override=refined)
        st.session_state.last_image_subject = refined
        mime, kind = "image/png", "image"
    else:
        return None, False

    st.session_state.created = {"bytes": b, "name": name, "mime": mime,
                               "kind": kind, "caption": msg}
    note = t("\n\n_See the **📦 Created file** section above to download or preview._",
             "\n\n_Upar **📦 Created file** section me download/preview available hai._")
    return f"✅ {msg}{note}", True


placeholder = t("Type your question or command...", "Apna sawaal ya command likho...") if file \
    else t("Upload a file and type here...", "File upload karke yahan likho...")

_typed = st.chat_input(placeholder)
# Voice text (if any) is captured in the sidebar and stored in session_state.
_voice_text = st.session_state.pop("_voice_prompt", None)
prompt = _typed or (_voice_text.strip() if _voice_text else None)

INTENT_SYS = """You are an intent classifier for a chatbot. Read the user's message and return ONLY JSON:
{"intent": "create_file|weather|datetime|web_search|normal", "format": "pdf|excel|pptx|chart|image|none"}

Rules:
- create_file: the user wants you to PRODUCE a downloadable FILE (a PDF, Excel/spreadsheet, PowerPoint, a chart image, or an AI-generated image). Also for "convert/turn/export ... into <format>". Set "format" accordingly.
    Also include follow-up phrasings like "another image", "one more image", "different picture", "same but ...", "make it again", "generate a new one" — these are still create_file with format="image" (or the previously discussed format).
    Do NOT choose create_file when the user only wants text, an explanation, MCQs/notes as text, or merely mentions a filename as a SOURCE (e.g. "make mcqs from report.pdf").
- weather: current weather / temperature / forecast for a place.
- datetime: today's date or the current time.
- web_search: needs live/current info from the internet (news, scores, prices, "latest ...").
- normal: everything else — questions, chit-chat, data questions about a table, code help, or Q&A about a document.
Use "none" for format unless intent is create_file."""


def classify_intent(llm, prompt):
    """Ask the LLM to classify the user's intent. Returns (intent, format) or (None, None)."""
    # short recent context (last 2 turns) so follow-ups like "another image" resolve
    recent = st.session_state.messages[-4:]
    hint = " | ".join(f"{m['role']}: {m['content'][:80]}" for m in recent if m.get("content"))
    user_msg = f"Recent conversation: {hint}\n\nCurrent user message: {prompt}" if hint else prompt
    j = llm.ask_json(INTENT_SYS, user_msg)
    if isinstance(j, dict) and j.get("intent"):
        fmt = j.get("format")
        return j.get("intent"), (fmt if fmt and fmt != "none" else None)
    return None, None


def _has_special_signal(prompt):
    """Does this message hint at a special intent? Only then do we call the LLM classifier."""
    p = prompt.lower()
    followup = any(w in p for w in
                   ["another", "one more", "again", "same but", "different one",
                    "new one", "aur ek", "ek aur", "dobara"])
    return (detect_creation(prompt) is not None
            or is_weather_query(prompt)
            or is_datetime_query(prompt)
            or is_search_query(prompt)
            or followup)


def decide_route(llm, prompt):
    """
    Hybrid router: clearly-normal messages go straight to 'normal'. When there is a
    hint of a special intent we ask the LLM to classify (ChatGPT-like); if that fails
    we fall back to keyword detection. Returns (route, format).
    """
    if not _has_special_signal(prompt):
        return "normal", None

    intent, fmt = None, None
    try:
        intent, fmt = classify_intent(llm, prompt)
    except Exception:
        intent = None

    if intent == "create_file" and fmt in ("pdf", "excel", "pptx", "chart", "image"):
        return "create", fmt
    if intent == "weather":
        return "weather", None
    if intent == "datetime":
        return "datetime", None
    if intent == "web_search":
        return "search", None
    if intent == "normal":
        return "normal", None

    # Classifier failed/unsure -> fall back to keyword detectors (safe default)
    tgt = detect_creation(prompt)
    if tgt:
        return "create", tgt
    if is_weather_query(prompt):
        return "weather", None
    if is_datetime_query(prompt):
        return "datetime", None
    if is_search_query(prompt):
        return "search", None
    return "normal", None


def route_single(llm, prompt):
    """Run one turn through the normal (single-step) router. Returns (reply, created_now)."""
    created_now = False
    route, fmt = decide_route(llm, prompt)
    if route == "create":
        reply, created_now = route_create(prompt, llm, fmt)
    elif route == "weather":
        reply = route_weather(prompt, llm)
    elif route == "datetime":
        reply = route_datetime(prompt)
    elif route == "search":
        reply = route_search(prompt, llm)
    elif file is None and st.session_state.get("kb_docs"):
        reply = route_rag(prompt, llm)
    elif file is None:
        sys = with_lang("You are a friendly, knowledgeable assistant. Remember details "
               "the user shared earlier in this conversation (like their name or any "
               "value they mentioned) and use them when asked. If relevant, remind "
               "them they can upload a CSV/Excel/PDF/code file or ask to create an "
               "Excel/PDF/PPTX/chart/image.")
        reply = _chat_with_context(llm, sys, temperature=0.5)
    elif file["kind"] == "table":
        reply = route_table(prompt, llm)
    elif file["kind"] == "code":
        reply = route_code(prompt, llm)
    else:
        reply = route_text(prompt, llm)
    return reply, created_now


# ==========================================================================
# AGENT MODE - plan a sequence of tool calls, execute them, then synthesize.
# ==========================================================================
AGENT_PLANNER_SYS = """You are a planning agent. Break the user's request into a short ordered list of tool calls.

Available tools:
- "web_search":  {"query": "..."}   - live internet info (news, prices, scores, current facts)
- "weather":     {"city": "..."}    - current weather for a city
- "datetime":    {}                  - today's date / current time
- "rag":         {"query": "..."}    - search the user's uploaded knowledge base
- "data_op":     {"instruction": "..."} - operate on the loaded table (sort/filter/add column/etc.)
- "create_file": {"format": "excel|pdf|pptx|chart|image", "instruction": "..."} - make a downloadable file
- "answer":      {"instruction": "..."} - reason or write text using previous step results

Return ONLY JSON:
{"steps": [{"tool": "...", "input": {...}, "why": "short phrase"}], "final": "what the final answer should contain"}

Rules:
- Use the FEWEST steps needed (1-5). A simple request can be a single step.
- Only use "create_file" when the user wants a downloadable file.
- Later steps can rely on earlier results (the executor passes prior results as context).
- Do not invent tools. Return valid JSON only.
"""


def _agent_run_tool(llm, tool, inp, prior, original_prompt):
    """Execute one agent tool. Returns (result_text, created_now)."""
    tool = (tool or "").lower()
    inp = inp or {}
    if tool == "web_search":
        ok, txt, src = web_search(inp.get("query", original_prompt), tavily_key=tavily_key)
        return (txt if ok else f"(search failed) {txt}"), False
    if tool == "weather":
        ok, txt = weather_summary(inp.get("city", ""))
        return txt, False
    if tool == "datetime":
        now = datetime.datetime.now()
        return f"{now:%A, %d %B %Y, %I:%M %p} (device time)", False
    if tool == "rag":
        idx = st.session_state.get("kb_index")
        hits = idx.search(inp.get("query", original_prompt), k=5) if idx else []
        return ("\n\n".join(h["text"] for h in hits) or "(no relevant documents found)"), False
    if tool == "data_op":
        df = st.session_state.df_view
        if df is None:
            return "(no table is loaded)", False
        plan = make_plan(llm, inp.get("instruction", original_prompt), df, lang=LANG)
        structural = plan.get("action") in ("add", "add_column", "delete_column",
                                            "rename_column", "reset")
        base = st.session_state.df_original if structural else df
        res, msg, mod = apply_plan(base, plan,
                                   user_prompt=inp.get("instruction", original_prompt))
        if mod:
            if plan.get("action") in ("add", "add_column", "delete_column", "rename_column"):
                st.session_state.df_original = res
            st.session_state.df_view = res.copy() if hasattr(res, "copy") else res
        return (msg or "done"), False
    if tool == "create_file":
        fmt = (inp.get("format") or "pdf").lower()
        instr = inp.get("instruction", original_prompt)
        if prior:
            instr = f"{instr}\n\nUse this context:\n{prior[:6000]}"
        try:
            reply, made = route_create(instr, llm, fmt)
            return reply, made
        except Exception as e:
            return f"(could not create {fmt}: {e})", False
    if tool == "answer":
        sys = with_lang("Use the prior step results below to answer.\n\n" + (prior or ""))
        return llm.ask(sys, inp.get("instruction", original_prompt), temperature=0.3), False
    return f"(unknown tool: {tool})", False


def run_agent(llm, prompt):
    """
    Plan and execute a multi-step task. Returns (reply_or_None, created_now).
    Returns (None, False) if planning fails, so the caller can fall back.
    """
    ctx = " | ".join(f"{m['role']}: {m['content'][:80]}"
                     for m in st.session_state.messages[-4:] if m.get("content"))
    try:
        plan = llm.ask_json(AGENT_PLANNER_SYS, f"Recent context: {ctx}\n\nUser request: {prompt}")
    except Exception:
        plan = None
    steps = (plan or {}).get("steps") or []
    if not steps:
        return None, False   # let caller fall back to single-step routing

    created_now = False
    transcript = []
    with st.status(t("🧠 Planning and running tools...", "🧠 Plan bana raha hoon..."),
                   expanded=True) as status:
        st.write(f"**{t('Plan', 'Plan')}:** {len(steps)} " + t("step(s)", "step"))
        for i, step in enumerate(steps, 1):
            tool = step.get("tool", "?")
            why = step.get("why") or tool
            st.write(f"**{i}. `{tool}`** — {why}")
            out, made = _agent_run_tool(llm, tool, step.get("input", {}),
                                        "\n\n".join(transcript), prompt)
            created_now = created_now or made
            transcript.append(f"[Step {i} - {tool}] {out}")
        status.update(label=t("✅ Done", "✅ Ho gaya"), state="complete", expanded=True)

    # Synthesize the final answer.
    # For "create_file" steps we already have a good message; return it as-is.
    if len(steps) == 1 and steps[0].get("tool") == "create_file":
        return transcript[-1].split("] ", 1)[-1], created_now

    joined = "\n\n".join(transcript)
    goal = (plan or {}).get("final") or "Summarize the results for the user."
    sys = with_lang("You are completing a task using the results of the steps below. "
                    "Write a clear, natural, helpful final answer for the user based "
                    "on those results. Do not mention 'Step 1' or the tools; just "
                    "answer the user's question. If the results contain URLs or "
                    "citations, keep the most useful ones.\n\n"
                    f"Goal: {goal}\n\nStep results:\n{joined}")
    try:
        final = _chat_with_context(llm, sys, temperature=0.3)
    except Exception:
        final = joined
    return final, created_now


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    created_now = False
    try:
        llm = get_llm()
        if st.session_state.get("cfg_agent"):
            reply, created_now = run_agent(llm, prompt)
            if reply is None:   # planning failed -> normal single-step
                with st.spinner(t("Thinking...", "Soch raha hoon...")):
                    reply, created_now = route_single(llm, prompt)
        else:
            with st.spinner(t("Thinking...", "Soch raha hoon...")):
                reply, created_now = route_single(llm, prompt)
    except LLMError as e:
        reply = f"❌ {e}"
    except Exception as e:
        reply = f"❌ Unexpected error: {e}"

    render_message("assistant", reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    _persist()   # save this exchange to disk
    # Refresh so the grid / code panel / created-file section update
    if created_now or (file and file["kind"] in ("table", "code")):
        st.rerun()
