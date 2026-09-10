
import hashlib
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="PECA Legal Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K = 5
MIN_RELEVANCE = 0.18


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(16,185,129,.06), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(59,130,246,.05), transparent 28%),
            #f7f8fa;
    }

    [data-testid="stHeader"] {
        background: rgba(247,248,250,.90);
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #065f46, #10b981);
        color: white;
        font-size: 22px;
        box-shadow: 0 8px 22px rgba(5,150,105,.20);
    }

    .eyebrow {
        color: #059669;
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 2.55rem;
        line-height: 1.08;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }

    .hero-subtitle {
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.55;
        margin-top: 8px;
        max-width: 780px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 14px;
        padding: 7px 11px;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-size: .78rem;
        font-weight: 700;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
    }

    /* Cards */
    .panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(17,24,39,.04);
    }

    .stat-card {
        padding: 16px 18px;
        min-height: 92px;
    }

    .stat-label {
        color: #6b7280;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .stat-value {
        margin-top: 7px;
        color: #111827;
        font-size: 1.45rem;
        font-weight: 800;
    }

    .stat-sub {
        color: #9ca3af;
        font-size: .74rem;
        margin-top: 3px;
    }

    .section-title {
        color: #111827;
        font-size: 1.04rem;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .section-caption {
        color: #6b7280;
        font-size: .84rem;
        margin-bottom: 10px;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 8px 0;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 15px;
        padding: 14px 16px;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: #111827;
        color: white;
        border-bottom-right-radius: 5px;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 7px 22px rgba(17,24,39,.035);
        border-bottom-left-radius: 5px;
    }

    /* Source cards */
    .source-card {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 15px;
        margin: 8px 0;
    }

    .source-meta {
        color: #6b7280;
        font-size: .76rem;
        margin-bottom: 8px;
    }

    .source-meta strong {
        color: #374151;
    }

    .source-text {
        color: #374151;
        font-size: .84rem;
        line-height: 1.58;
    }

    .relevance {
        float: right;
        color: #047857;
        font-weight: 700;
    }

    .grounded-note {
        border: 1px solid #d1fae5;
        background: #f0fdf4;
        border-radius: 12px;
        padding: 11px 13px;
        color: #065f46;
        font-size: .82rem;
    }

    .warning-note {
        border: 1px solid #fde68a;
        background: #fffbeb;
        border-radius: 12px;
        padding: 11px 13px;
        color: #92400e;
        font-size: .82rem;
    }

    .footer {
        color: #9ca3af;
        text-align: center;
        font-size: .75rem;
        padding-top: 18px;
    }

    /* Sidebar */
    .side-brand {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .side-muted {
        color: #9ca3af;
        font-size: .78rem;
        line-height: 1.55;
    }

    /* Streamlit controls */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #10b981;
        color: #047857;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def get_groq_api_key() -> str:
    """Read the Groq API key from Streamlit Secrets or environment."""
    try:
        secret = st.secrets.get("GROQ_API_KEY")
        if secret:
            return str(secret).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> str:
    patterns = [
        r"\bSection\s+([0-9]+[A-Za-z]?(?:\([^)]+\))?)",
        r"\bSECTION\s+([0-9]+[A-Za-z]?(?:\([^)]+\))?)",
        r"\bSec\.\s*([0-9]+[A-Za-z]?(?:\([^)]+\))?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"Section {match.group(1)}"

    return "Section not detected"


def extract_pages(pdf_file) -> List[Dict]:
    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_page(text: str) -> List[str]:
    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + CHUNK_SIZE, len(text))

        if target_end < len(text):
            # Prefer a natural boundary near the target end.
            boundary_window = text[max(start, target_end - 160):target_end]
            boundaries = [
                boundary_window.rfind("\n\n"),
                boundary_window.rfind(". "),
                boundary_window.rfind("; "),
                boundary_window.rfind(" "),
            ]

            best = max(boundaries)

            if best >= 0:
                target_end = max(
                    start + 1,
                    target_end - 160 + best + 1
                )

        chunk = text[start:target_end].strip()

        if chunk:
            chunks.append(chunk)

        if target_end >= len(text):
            break

        start = max(
            target_end - CHUNK_OVERLAP,
            start + 1,
        )

    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    chunks = []

    for page in pages:
        for part_index, text in enumerate(
            chunk_page(page["text"]),
            start=1,
        ):
            chunks.append({
                "id": len(chunks),
                "page": page["page"],
                "chunk_on_page": part_index,
                "section": detect_section(text),
                "text": text,
            })

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def embed_texts(texts: Tuple[str, ...]) -> np.ndarray:
    model = load_embedding_model()

    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


def word_set(text: str) -> set:
    stop_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "what", "who", "how", "does", "do",
        "can", "this", "that", "with", "from", "about", "under", "by",
        "which", "where", "when", "why",
    }

    words = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return {word for word in words if word not in stop_words}


def lexical_score(question: str, text: str) -> float:
    q = word_set(question)
    t = word_set(text)

    if not q:
        return 0.0

    return min(
        len(q.intersection(t)) / len(q),
        1.0,
    )


def retrieve(
    question: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[Dict]:

    model = load_embedding_model()

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    norm = np.linalg.norm(
        query_vector,
        axis=1,
        keepdims=True,
    )

    query_vector = query_vector / np.maximum(norm, 1e-12)

    semantic_scores = np.dot(
        embeddings,
        query_vector[0],
    )

    results = []

    for index, chunk in enumerate(chunks):
        lexical = lexical_score(
            question,
            chunk["text"],
        )

        # Semantic similarity remains primary.
        combined = (
            0.85 * float(semantic_scores[index])
            + 0.15 * lexical
        )

        item = dict(chunk)
        item["semantic_score"] = float(semantic_scores[index])
        item["lexical_score"] = float(lexical)
        item["score"] = float(combined)

        results.append(item)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


def build_context(sources: List[Dict]) -> str:
    blocks = []

    for item in sources:
        blocks.append(
            f"""
SOURCE_ID: {item["id"]}
PAGE: {item["page"]}
SECTION: {item["section"]}

SOURCE_TEXT:
{item["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def build_grounded_prompt(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    context = build_context(sources)

    recent_history = history[-6:]
    history_lines = []

    for message in recent_history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role in {"user", "assistant"} and content:
            history_lines.append(
                f"{role.upper()}: {content}"
            )

    history_text = "\n".join(history_lines)

    return f"""
You are the PECA Legal Information Assistant.

SOURCE OF TRUTH:
Only the retrieved passages from the uploaded Prevention of Electronic
Crimes Act, 2016 document are authoritative for this answer.

STRICT RULES:
1. Use only the supplied source context.
2. Do not invent legal sections, penalties, procedures, authorities,
   dates, exceptions, or interpretations.
3. If the supplied context does not support the answer, respond exactly:
   "I couldn't find enough relevant information in the uploaded document
   to answer that reliably."
4. Explain the document in simple language while preserving its meaning.
5. When supported, cite the page and section naturally, e.g.
   "[Section 21, Page 24]".
6. Never fabricate citations.
7. Do not provide personal legal advice.
8. For punishment or legal consequences, state what the document says
   without adding outside legal conclusions.
9. Keep the answer concise and useful.

RECENT CONVERSATION:
{history_text if history_text else "(none)"}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
""".strip()


def answer_question(
    question: str,
    sources: List[Dict],
    history: List[Dict],
    api_key: str,
) -> str:

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful, document-grounded assistant. "
                    "Never add unsupported legal information."
                ),
            },
            {
                "role": "user",
                "content": build_grounded_prompt(
                    question,
                    sources,
                    history,
                ),
            },
        ],
        temperature=0.1,
        max_completion_tokens=900,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The model returned an empty response."
        )

    return content.strip()


def document_hash(uploaded_file) -> str:
    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


def clear_document_state() -> None:
    for key in [
        "document_hash",
        "document_name",
        "pages",
        "chunks",
        "embeddings",
        "messages",
        "latest_sources",
        "latest_answer",
    ]:
        st.session_state.pop(key, None)


def index_document(uploaded_file) -> None:
    current_hash = document_hash(uploaded_file)

    if st.session_state.get("document_hash") == current_hash:
        return

    clear_document_state()

    with st.spinner("Indexing document..."):
        pages = extract_pages(uploaded_file)

        if not pages:
            raise ValueError(
                "No readable text was found in the PDF. "
                "A scanned PDF may require OCR."
            )

        chunks = build_chunks(pages)

        if not chunks:
            raise ValueError(
                "No searchable text chunks were created."
            )

        embeddings = embed_texts(
            tuple(chunk["text"] for chunk in chunks)
        )

    st.session_state.document_hash = current_hash
    st.session_state.document_name = uploaded_file.name
    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.messages = []
    st.session_state.latest_sources = []
    st.session_state.latest_answer = ""


# ============================================================
# API KEY
# ============================================================

API_KEY = get_groq_api_key()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="side-brand">🇵🇰 PECA Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-muted">'
        "Document-grounded Q&amp;A for Pakistan's "
        "Prevention of Electronic Crimes Act, 2016."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    if API_KEY:
        st.success("Groq connected")
    else:
        st.error("Groq key missing")

    st.markdown("---")

    st.markdown("**Document**")

    uploaded_pdf = st.file_uploader(
        "Upload the official PECA PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.caption(
        "Your PDF is processed for the current app session."
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.session_state.latest_answer = ""
        st.rerun()

    if st.button(
        "Reset document",
        use_container_width=True,
    ):
        clear_document_state()
        st.rerun()

    st.markdown("---")

    st.markdown("**Suggested questions**")

    suggested = [
        "What does the document say about cyber harassment?",
        "What punishment is mentioned for this offence?",
        "Which section covers this offence?",
        "What does the Act say about complaints?",
        "What powers does the Authority have?",
    ]

    for index, question in enumerate(suggested):
        if st.button(
            question,
            use_container_width=True,
            key=f"suggested_{index}",
        ):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown("---")

    st.markdown(
        '<div class="side-muted">'
        "<b>Legal notice</b><br>"
        "This tool summarizes information from the uploaded document. "
        "It is not a substitute for professional legal advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">⚖</div>
        <div>
            <div class="eyebrow">Document-grounded AI</div>
        </div>
    </div>

    <div class="hero-title">PECA Legal Information Assistant</div>

    <div class="hero-subtitle">
        Ask questions about Pakistan's Prevention of Electronic Crimes Act,
        2016. The assistant retrieves relevant passages from your document
        before generating an answer.
    </div>

    <div class="status-pill">
        <span class="status-dot"></span>
        Strict document-grounded mode
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_pdf is None:
    st.markdown(
        """
        <div class="panel" style="padding:24px;">
            <div class="section-title">Start with the official document</div>
            <div class="section-caption">
                Upload the PECA PDF from the left sidebar to create your
                searchable document index.
            </div>
            <div class="grounded-note">
                <b>How it works:</b>
                PDF → text extraction → smart chunks → embeddings →
                semantic retrieval → Groq LLM → cited answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="panel stat-card">
                <div class="stat-label">Step 01</div>
                <div class="stat-value">Upload</div>
                <div class="stat-sub">Add the official PECA PDF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="panel stat-card">
                <div class="stat-label">Step 02</div>
                <div class="stat-value">Retrieve</div>
                <div class="stat-sub">Find relevant provisions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="panel stat-card">
                <div class="stat-label">Step 03</div>
                <div class="stat-value">Explain</div>
                <div class="stat-sub">Generate a grounded answer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">PECA Legal Information Assistant · RAG MVP</div>',
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# PROCESS DOCUMENT
# ============================================================

try:
    index_document(uploaded_pdf)
except Exception as exc:
    st.error(f"Document processing failed: {exc}")
    st.stop()


# ============================================================
# DOCUMENT STATS
# ============================================================

pages = st.session_state["pages"]
chunks = st.session_state["chunks"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="panel stat-card">
            <div class="stat-label">Document</div>
            <div class="stat-value" style="font-size:1.05rem;">
                {st.session_state["document_name"][:24]}
            </div>
            <div class="stat-sub">Uploaded PDF</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="panel stat-card">
            <div class="stat-label">Pages</div>
            <div class="stat-value">{len(pages)}</div>
            <div class="stat-sub">Readable pages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="panel stat-card">
            <div class="stat-label">Searchable chunks</div>
            <div class="stat-value">{len(chunks)}</div>
            <div class="stat-sub">Indexed passages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="panel stat-card">
            <div class="stat-label">LLM</div>
            <div class="stat-value" style="font-size:1.05rem;">
                GPT-OSS 120B
            </div>
            <div class="stat-sub">via Groq</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("")
st.markdown(
    '<div class="grounded-note">'
    "<b>Grounding:</b> Answers are generated from retrieved passages "
    "from the uploaded document. The source passages remain visible below."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("")


# ============================================================
# DOCUMENT DETAILS
# ============================================================

with st.expander("📄 Document details", expanded=False):
    first_page_preview = pages[0]["text"][:2200]

    st.markdown(
        f"**File:** `{st.session_state['document_name']}`"
    )

    st.markdown("**First-page preview**")

    st.write(first_page_preview)


# ============================================================
# CHAT
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


pending_question = st.session_state.get("pending_question")
typed_question = st.chat_input(
    "Ask about the PECA document..."
)

question = typed_question or pending_question

if question:
    st.session_state.pending_question = None

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):

            with st.spinner("Retrieving relevant provisions..."):
                sources = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    top_k=TOP_K,
                )

            if not sources:
                raise RuntimeError(
                    "No relevant document passages were retrieved."
                )

            best_score = sources[0]["score"]

            if best_score < MIN_RELEVANCE:
                answer = (
                    "I couldn't find enough relevant information in the "
                    "uploaded document to answer that reliably."
                )
            else:
                with st.spinner("Generating grounded answer..."):
                    answer = answer_question(
                        question,
                        sources,
                        st.session_state.messages[:-1],
                        API_KEY,
                    )

            st.markdown(answer)

            st.session_state.latest_sources = sources
            st.session_state.latest_answer = answer

    except Exception as exc:
        answer = (
            "I couldn't generate the answer because the AI service "
            f"returned an error: `{exc}`"
        )

        st.session_state.latest_sources = []

        with st.chat_message("assistant"):
            st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })


# ============================================================
# SOURCES
# ============================================================

if st.session_state.get("latest_sources"):
    st.markdown("---")

    st.markdown(
        '<div class="section-title">📚 Sources used for the latest answer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Inspect the retrieved passages that were provided to the LLM."
        "</div>",
        unsafe_allow_html=True,
    )

    for source in st.session_state["latest_sources"]:
        source_title = (
            f"Page {source['page']} · "
            f"{source['section']} · "
            f"Relevance {source['score']:.3f}"
        )

        with st.expander(source_title):
            st.markdown(
                f"""
                <div class="source-meta">
                    <strong>Page:</strong> {source["page"]}
                    &nbsp; · &nbsp;
                    <strong>Section:</strong> {source["section"]}
                    <span class="relevance">
                        {source["score"]:.3f}
                    </span>
                </div>

                <div class="source-text">
                    {source["text"]}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DOWNLOAD LAST ANSWER
# ============================================================

if st.session_state.get("latest_answer"):
    st.markdown("")

    st.download_button(
        "⬇️ Download latest answer",
        data=st.session_state["latest_answer"],
        file_name="peca_answer.txt",
        mime="text/plain",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PECA Legal Information Assistant · Retrieval-Augmented Generation MVP<br>
        Information grounded in the uploaded document · Not legal advice
    </div>
    """,
    unsafe_allow_html=True,
)
