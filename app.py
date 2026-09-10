
import hashlib
import os
import re
from io import BytesIO
from typing import Dict, List, Tuple
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="PECA Legal Information Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "PECA Legal Information Assistant"
APP_TAGLINE = (
    "Ask questions about Pakistan's Prevention of Electronic Crimes Act, "
    "2016 — grounded in the source document."
)

# GitHub repository supplied for the project PDF.
PDF_BLOB_URL = (
    "https://github.com/codewithMeesum/Peca-Rag_Assistant/blob/main/"
    "PECA%202026.pdf"
)
PDF_RAW_URL = (
    "https://raw.githubusercontent.com/codewithMeesum/"
    "Peca-Rag_Assistant/main/PECA%202026.pdf"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K = 6


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(16,185,129,.07),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(59,130,246,.05),
                transparent 28%
            ),
            #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(248,250,252,.88);
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .eyebrow {
        color: #059669;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .hero-title {
        color: #0f172a;
        font-size: 2.55rem;
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -.035em;
        margin: 0;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 820px;
        margin-top: 9px;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 13px;
        border: 1px solid #a7f3d0;
        background: #ecfdf5;
        color: #047857;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: .76rem;
        font-weight: 750;
    }

    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
    }

    .panel {
        background: rgba(255,255,255,.94);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        box-shadow: 0 12px 32px rgba(15,23,42,.045);
    }

    .stat {
        min-height: 104px;
        padding: 16px 18px;
    }

    .stat-label {
        color: #64748b;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
    }

    .stat-value {
        color: #0f172a;
        margin-top: 6px;
        font-size: 1.42rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .stat-note {
        color: #94a3b8;
        margin-top: 4px;
        font-size: .73rem;
    }

    .grounding {
        border: 1px solid #bbf7d0;
        background: #f0fdf4;
        color: #166534;
        border-radius: 12px;
        padding: 12px 14px;
        font-size: .82rem;
        line-height: 1.55;
    }

    .legal-note {
        border: 1px solid #fde68a;
        background: #fffbeb;
        color: #92400e;
        border-radius: 12px;
        padding: 12px 14px;
        font-size: .8rem;
        line-height: 1.55;
    }

    .source-card {
        border: 1px solid #e2e8f0;
        background: #ffffff;
        border-radius: 13px;
        padding: 14px 15px;
        margin: 7px 0;
    }

    .source-meta {
        color: #64748b;
        font-size: .75rem;
        margin-bottom: 8px;
    }

    .source-meta strong {
        color: #334155;
    }

    .source-text {
        color: #334155;
        font-size: .84rem;
        line-height: 1.62;
    }

    .side-title {
        color: #f8fafc;
        font-size: 1.03rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .side-text {
        color: #94a3b8;
        font-size: .78rem;
        line-height: 1.55;
    }

    .footer {
        color: #94a3b8;
        text-align: center;
        font-size: .74rem;
        padding-top: 20px;
        line-height: 1.55;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        border-color: #dbe3ed;
    }

    .stButton > button:hover {
        border-color: #10b981;
        color: #047857;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 15px;
    }

    @media (max-width: 850px) {
        .hero-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:
    try:
        value = st.secrets.get("GROQ_API_KEY")
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


API_KEY = get_api_key()


# ============================================================
# PDF LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def download_default_pdf() -> bytes:
    """Download the project's default PECA PDF from GitHub."""
    request = Request(
        PDF_RAW_URL,
        headers={"User-Agent": "PECA-RAG-Assistant/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        data = response.read()

    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The configured GitHub source did not return a valid PDF."
        )

    return data


def pdf_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# SECTION DETECTION
# ============================================================

def find_section_markers(text: str) -> List[Tuple[int, str]]:
    """
    Detect common legal-section heading patterns.

    This is intentionally conservative: section metadata is a best-effort
    label while page number and source text remain the primary evidence.
    """
    markers = []

    patterns = [
        re.compile(
            r"(?im)^\s*(section\s+"
            r"\d+[A-Za-z]?(?:\([^)]+\))?)[\.:]?\s*(.*)$"
        ),
        re.compile(
            r"(?im)^\s*(\d+[A-Za-z]?(?:\([^)]+\))?)\.\s+"
            r"([A-Z][^\n]{2,120})$"
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(text):
            number = match.group(1).strip()
            title = match.group(2).strip()

            if number.lower().startswith("section"):
                label = f"{number}"
            else:
                label = f"Section {number}"

            if title:
                label = f"{label} — {title}"

            markers.append((match.start(), label))

    markers.sort(key=lambda item: item[0])

    # Remove near-duplicate markers.
    clean = []
    for position, label in markers:
        if clean and position - clean[-1][0] < 10:
            continue
        clean.append((position, label))

    return clean


def section_for_position(
    markers: List[Tuple[int, str]],
    position: int,
) -> str:
    current = "Section not detected"

    for marker_position, label in markers:
        if marker_position > position:
            break
        current = label

    return current


# ============================================================
# CHUNKING
# ============================================================

def split_text(text: str) -> List[Tuple[int, int, str]]:
    """
    Return character positions plus text so section metadata can be
    preserved as the chunk is created.
    """
    if len(text) <= CHUNK_SIZE:
        return [(0, len(text), text)]

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + CHUNK_SIZE, len(text))
        end = target_end

        if target_end < len(text):
            window_start = max(start, target_end - 180)
            window = text[window_start:target_end]

            boundaries = [
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("; "),
                window.rfind(" "),
            ]

            best = max(boundaries)

            if best >= 20:
                end = window_start + best + 1

        chunk = text[start:end].strip()

        if chunk:
            actual_start = text.find(
                chunk,
                start,
                min(len(text), end + 20),
            )

            if actual_start < 0:
                actual_start = start

            actual_end = actual_start + len(chunk)

            chunks.append(
                (
                    actual_start,
                    actual_end,
                    chunk,
                )
            )

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    chunks = []

    for page in pages:
        page_text = page["text"]
        markers = find_section_markers(page_text)

        for chunk_index, (start, end, text) in enumerate(
            split_text(page_text),
            start=1,
        ):
            chunks.append(
                {
                    "id": len(chunks),
                    "page": page["page"],
                    "chunk_on_page": chunk_index,
                    "section": section_for_position(
                        markers,
                        start,
                    ),
                    "text": text,
                }
            )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data(show_spinner=False)
def embed_chunks(texts: Tuple[str, ...]) -> np.ndarray:
    model = load_embedding_model()

    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / np.maximum(norms, 1e-12)


# ============================================================
# RETRIEVAL
# ============================================================

def lexical_tokens(text: str) -> set:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "what", "who", "how", "does", "do",
        "can", "this", "that", "with", "from", "about", "which", "where",
        "when", "why", "tell", "me", "please",
    }

    tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())

    return {
        token
        for token in tokens
        if token not in stopwords
    }


def lexical_score(question: str, text: str) -> float:
    question_tokens = lexical_tokens(question)
    text_tokens = lexical_tokens(text)

    if not question_tokens:
        return 0.0

    overlap = question_tokens.intersection(text_tokens)

    return min(
        len(overlap) / len(question_tokens),
        1.0,
    )


def section_query_boost(question: str, chunk: Dict) -> float:
    """
    Give a small boost when a user explicitly asks about a section
    number present in the retrieved chunk.
    """
    matches = re.findall(
        r"\b(?:section|sec\.?)\s*(\d+[A-Za-z]?)",
        question.lower(),
    )

    if not matches:
        return 0.0

    section_text = chunk["section"].lower()

    for number in matches:
        if number in section_text:
            return 1.0

    return 0.0


def retrieve_chunks(
    question: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[Dict]:

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    query_norm = np.linalg.norm(
        query_embedding,
        axis=1,
        keepdims=True,
    )

    query_embedding = (
        query_embedding
        / np.maximum(query_norm, 1e-12)
    )

    semantic_scores = np.dot(
        embeddings,
        query_embedding[0],
    )

    results = []

    for index, chunk in enumerate(chunks):
        lexical = lexical_score(
            question,
            chunk["text"],
        )

        section_boost = section_query_boost(
            question,
            chunk,
        )

        combined = (
            0.82 * float(semantic_scores[index])
            + 0.15 * lexical
            + 0.03 * section_boost
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

    # Light diversity: avoid returning many chunks from the same page
    # when other highly relevant pages are available.
    selected = []
    page_counts = {}

    for item in results:
        page = item["page"]
        page_count = page_counts.get(page, 0)

        if page_count >= 2 and len(selected) < top_k - 1:
            continue

        selected.append(item)
        page_counts[page] = page_count + 1

        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# GROQ ANSWERING
# ============================================================

def build_context(sources: List[Dict]) -> str:
    blocks = []

    for source in sources:
        blocks.append(
            f"""
SOURCE {source["id"]}
PAGE: {source["page"]}
SECTION: {source["section"]}

TEXT:
{source["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def build_grounded_prompt(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    context = build_context(sources)

    previous = []

    for message in history[-6:]:
        role = message.get("role")
        content = message.get("content")

        if role in {"user", "assistant"} and content:
            previous.append(
                f"{role.upper()}: {content}"
            )

    history_text = (
        "\n".join(previous)
        if previous
        else "(No previous conversation.)"
    )

    return f"""
You are a careful document-grounded information assistant for the
Prevention of Electronic Crimes Act, 2016 (PECA).

SOURCE OF TRUTH:
Use ONLY the retrieved passages provided below.

STRICT RULES:
- Do not invent or guess legal provisions.
- Do not add outside laws, amendments, penalties, dates, procedures,
  or interpretations unless they are explicitly supported by the
  retrieved source text.
- If the retrieved passages do not contain enough information, say:
  "I couldn't find enough relevant information in the uploaded document
  to answer that reliably."
- Explain the source in plain, understandable language.
- Preserve the legal meaning of the source.
- When possible, cite the supporting page and section like:
  [Section 21, Page 24].
- Never fabricate a section number or page number.
- Do not present the response as professional legal advice.
- For questions about punishments or consequences, clearly state that
  the answer reflects what the document says.
- Be concise unless the user asks for detail.

RECENT CONVERSATION:
{history_text}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer only from the retrieved document context.
""".strip()


def generate_answer(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit Secrets."
        )

    client = Groq(api_key=API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document-grounded assistant. "
                    "Do not add unsupported legal information."
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

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "The model returned an empty response."
        )

    return answer.strip()


# ============================================================
# STATE
# ============================================================

def clear_document_state() -> None:
    for key in [
        "doc_hash",
        "doc_name",
        "doc_source",
        "pdf_bytes",
        "pages",
        "chunks",
        "embeddings",
        "messages",
        "latest_sources",
        "pending_question",
    ]:
        st.session_state.pop(key, None)


def prepare_document(
    pdf_bytes: bytes,
    name: str,
    source: str,
) -> None:

    current_hash = pdf_hash(pdf_bytes)

    if st.session_state.get("doc_hash") == current_hash:
        return

    clear_document_state()

    with st.spinner("Preparing the PECA document for search..."):
        pages = extract_pages(pdf_bytes)

        if not pages:
            raise ValueError(
                "No readable text was extracted from this PDF. "
                "A scanned PDF may require OCR."
            )

        chunks = build_chunks(pages)

        if not chunks:
            raise ValueError(
                "No searchable chunks were created."
            )

        embeddings = embed_chunks(
            tuple(chunk["text"] for chunk in chunks)
        )

    st.session_state.doc_hash = current_hash
    st.session_state.doc_name = name
    st.session_state.doc_source = source
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.messages = []
    st.session_state.latest_sources = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="side-title">🇵🇰 PECA Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-text">'
        "A focused RAG assistant for the Prevention of Electronic "
        "Crimes Act, 2016."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    if API_KEY:
        st.success("Groq connected")
    else:
        st.error("Groq API key missing")

    st.markdown("---")

    st.markdown("**Source document**")

    use_default = st.radio(
        "Document mode",
        [
            "Official PECA document",
            "Upload another PDF",
        ],
        label_visibility="collapsed",
    )

    uploaded_pdf = None

    if use_default == "Official PECA document":
        st.markdown(
            '<div class="side-text">'
            "The project loads its PECA PDF automatically, so visitors "
            "do not need to download and upload it manually."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'[📥 Download source PDF]({PDF_BLOB_URL})'
        )

        try:
            default_bytes = download_default_pdf()
            prepare_document(
                default_bytes,
                "PECA 2026.pdf",
                PDF_BLOB_URL,
            )
        except Exception as exc:
            st.error(
                "The default PECA PDF could not be loaded automatically."
            )
            st.caption(
                "You can use the upload mode below or open the source PDF "
                "from the project repository."
            )
            st.caption(str(exc))

    else:
        uploaded_pdf = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
        )

        if uploaded_pdf is not None:
            try:
                prepare_document(
                    uploaded_pdf.getvalue(),
                    uploaded_pdf.name,
                    "User-uploaded PDF",
                )
            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")

    st.markdown("---")

    if st.button(
        "🧹 Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.rerun()

    if st.button(
        "↻ Reset document",
        use_container_width=True,
    ):
        clear_document_state()
        st.rerun()

    st.markdown("---")

    st.markdown("**Try asking**")

    suggestions = [
        "What does the document say about cyber harassment?",
        "What punishment is mentioned for cyber-stalking?",
        "Which section discusses cyber-bullying?",
        "What does the Act say about complaints?",
        "What powers does the Authority have?",
    ]

    for index, suggestion in enumerate(suggestions):
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion
            st.rerun()

    st.markdown("---")

    st.markdown(
        '<div class="side-text">'
        "<b>Legal notice</b><br>"
        "This application is an informational tool grounded in the "
        "uploaded document. It is not professional legal advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="eyebrow">Document-grounded AI · RAG</div>

    <div class="hero-title">
        🇵🇰 PECA Legal Information Assistant
    </div>

    <div class="hero-subtitle">
        Ask questions about Pakistan's Prevention of Electronic Crimes Act,
        2016. The assistant retrieves relevant passages from the source
        document before generating an answer.
    </div>

    <div class="pill">
        <span class="dot"></span>
        Grounded answers with visible source evidence
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# EMPTY / LOAD STATE
# ============================================================

if "chunks" not in st.session_state:
    st.info(
        "The PECA document is being prepared. If it does not load, "
        "use the upload option in the sidebar."
    )
    st.stop()


# ============================================================
# STATS
# ============================================================

pages = st.session_state["pages"]
chunks = st.session_state["chunks"]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="panel stat">
            <div class="stat-label">Pages</div>
            <div class="stat-value">{len(pages)}</div>
            <div class="stat-note">Readable pages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="panel stat">
            <div class="stat-label">Indexed</div>
            <div class="stat-value">{len(chunks)}</div>
            <div class="stat-note">Searchable chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="panel stat">
            <div class="stat-label">Embeddings</div>
            <div class="stat-value">MiniLM-L6</div>
            <div class="stat-note">Local semantic vectors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="panel stat">
            <div class="stat-label">Generation</div>
            <div class="stat-value">GPT-OSS 120B</div>
            <div class="stat-note">via Groq</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

st.markdown(
    '<div class="grounding">'
    "<b>How answers are produced:</b> "
    "Question → retrieval → relevant PECA passages → grounded prompt → "
    "LLM response → source evidence."
    "</div>",
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# DOCUMENT DETAILS
# ============================================================

with st.expander(
    f"📄 {st.session_state['doc_name']}",
    expanded=False,
):
    st.markdown(
        f"**Source:** {st.session_state['doc_source']}"
    )

    st.markdown("**Document preview**")

    preview = pages[0]["text"][:3000]

    st.write(preview)

    st.markdown(
        f'[Open source PDF in GitHub]({PDF_BLOB_URL})'
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

pending_question = st.session_state.get(
    "pending_question"
)

typed_question = st.chat_input(
    "Ask a question about the PECA document..."
)

question = typed_question or pending_question

if question:
    st.session_state.pending_question = None

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    sources = []

    try:
        with st.chat_message("assistant"):

            with st.spinner("Searching the PECA document..."):
                sources = retrieve_chunks(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    top_k=TOP_K,
                )

            if not sources:
                answer = (
                    "I couldn't find enough relevant information in the "
                    "uploaded document to answer that reliably."
                )
            else:
                with st.spinner("Generating a grounded answer..."):
                    answer = generate_answer(
                        question,
                        sources,
                        st.session_state.messages[:-1],
                    )

            st.markdown(answer)

        st.session_state.latest_sources = sources

    except Exception as exc:
        answer = (
            "I couldn't generate the answer because the AI service "
            f"returned an error: `{exc}`"
        )

        st.session_state.latest_sources = []

        with st.chat_message("assistant"):
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# SOURCES
# ============================================================

latest_sources = st.session_state.get(
    "latest_sources",
    [],
)

if latest_sources:
    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        "📚 Evidence used for the latest answer"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "These passages were retrieved from the PECA document and "
        "provided to the language model."
    )

    for source in latest_sources:
        label = (
            f"Page {source['page']} · "
            f"{source['section']} · "
            f"retrieval {source['score']:.3f}"
        )

        with st.expander(label):
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-meta">
                        <strong>Page:</strong> {source["page"]}
                        &nbsp; · &nbsp;
                        <strong>Section:</strong> {source["section"]}
                        <br>
                        Semantic: {source["semantic_score"]:.3f}
                        &nbsp; · &nbsp;
                        Lexical: {source["lexical_score"]:.3f}
                    </div>

                    <div class="source-text">
                        {source["text"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# LEGAL NOTE
# ============================================================

st.markdown("")
st.markdown(
    '<div class="legal-note">'
    "<b>Important:</b> This tool summarizes and explains information "
    "contained in the selected document. It does not replace advice "
    "from a qualified legal professional."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PECA Legal Information Assistant · RAG MVP<br>
        Document-grounded retrieval · Source evidence visible to the user
    </div>
    """,
    unsafe_allow_html=True,
)
