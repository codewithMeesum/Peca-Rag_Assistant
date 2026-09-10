
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
# PAGE / PRODUCT CONFIG
# ============================================================

st.set_page_config(
    page_title="PECA Legal Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRODUCT_NAME = "PECA Legal Assistant"
PRODUCT_TAGLINE = (
    "Ask questions about the Prevention of Electronic Crimes Act, 2016 "
    "using document-grounded AI."
)

DEFAULT_PDF_NAME = "PECA 2026.pdf"
DEFAULT_PDF_PAGE_URL = (
    "https://github.com/codewithMeesum/Peca-Rag_Assistant/blob/main/"
    "PECA%202026.pdf"
)
DEFAULT_PDF_RAW_URL = (
    "https://raw.githubusercontent.com/codewithMeesum/"
    "Peca-Rag_Assistant/main/PECA%202026.pdf"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K = 6
MIN_RETRIEVAL_SCORE = 0.22

SUGGESTIONS = [
    "What does the Act say about cyber harassment?",
    "What punishment is mentioned for cyber-stalking?",
    "Which section discusses cyber-bullying?",
    "What does the Act say about complaints?",
    "What powers does the Authority have?",
]


# ============================================================
# UI — MINIMAL CHAT PRODUCT
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Base ---------- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .stApp {
        background: #ffffff;
        color: #171717;
    }

    [data-testid="stHeader"] {
        background: #ffffff;
        border-bottom: 1px solid #eeeeee;
    }

    [data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e6e6e6;
    }

    [data-testid="stSidebar"] * {
        color: #171717;
    }

    .block-container {
        max-width: 920px;
        padding-top: 1.15rem;
        padding-bottom: 7rem;
    }

    /* ---------- Fix sidebar toggle visibility ---------- */
    [data-testid="stSidebar"] button {
        color: #171717 !important;
    }

    [data-testid="stSidebar"] button svg {
        color: #171717 !important;
        fill: #171717 !important;
    }

    /* ---------- Sidebar ---------- */
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 5px 0 4px;
    }

    .brand-icon {
        width: 32px;
        height: 32px;
        border-radius: 9px;
        background: #0d7a46;
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 800;
        flex: 0 0 auto;
    }

    .brand-name {
        color: #171717;
        font-size: .93rem;
        font-weight: 760;
    }

    .brand-sub {
        color: #737373;
        font-size: .69rem;
        line-height: 1.4;
        margin: 0 0 14px 42px;
    }

    .sidebar-label {
        color: #737373;
        font-size: .66rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 15px 2px 7px;
    }

    .document-card {
        background: #fff;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 10px 11px;
        margin-bottom: 7px;
    }

    .document-card-title {
        font-size: .78rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .document-card-meta {
        color: #7a7a7a;
        font-size: .67rem;
        margin-top: 2px;
    }

    .sidebar-status {
        border-radius: 8px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #047857 !important;
        padding: 7px 9px;
        font-size: .7rem;
        font-weight: 700;
    }

    /* ---------- Welcome ---------- */
    .welcome {
        min-height: 60vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 18px 12px 45px;
    }

    .welcome-inner {
        max-width: 720px;
    }

    .welcome-icon {
        width: 62px;
        height: 62px;
        border-radius: 18px;
        background: #0d7a46;
        color: #ffffff !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 800;
        box-shadow: 0 10px 25px rgba(13,122,70,.12);
        margin-bottom: 17px;
    }

    .welcome-title {
        color: #171717;
        font-size: 2.05rem;
        line-height: 1.1;
        font-weight: 760;
        letter-spacing: -.035em;
        margin-bottom: 9px;
    }

    .welcome-subtitle {
        color: #737373;
        font-size: .92rem;
        line-height: 1.62;
    }

    .welcome-source {
        display: inline-block;
        margin-top: 18px;
        padding: 8px 11px;
        border: 1px solid #e8e8e8;
        background: #fafafa;
        border-radius: 999px;
        color: #525252;
        font-size: .72rem;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        padding: 11px 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        color: #171717 !important;
        padding: 0 !important;
        font-size: .95rem;
        line-height: 1.68;
    }

    [data-testid="stChatMessageContent"] * {
        color: #171717 !important;
    }

    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        justify-content: flex-end;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] {
        max-width: 78%;
        background: #f4f4f4 !important;
        border-radius: 18px !important;
        padding: 10px 14px !important;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border-top: 1px solid #eeeeee;
        padding-top: 10px;
    }

    [data-testid="stChatInput"] textarea {
        color: #171717 !important;
        background: #ffffff !important;
        border: 1px solid #d9d9d9 !important;
        border-radius: 18px !important;
        box-shadow: 0 3px 18px rgba(0,0,0,.045);
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #b9b9b9 !important;
        box-shadow: 0 0 0 1px #b9b9b9 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9a9a9a !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 9px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: #333333 !important;
        text-align: left !important;
        font-size: .77rem !important;
        font-weight: 600 !important;
        padding: 7px 9px !important;
    }

    .stButton > button:hover {
        background: #ececef !important;
    }

    /* ---------- Evidence ---------- */
    .evidence {
        background: #fafafa;
        border: 1px solid #e7e7e7;
        border-radius: 12px;
        padding: 12px 13px;
        margin: 7px 0;
    }

    .evidence-badge {
        display: inline-block;
        color: #047857;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        font-size: .63rem;
        font-weight: 800;
        padding: 3px 7px;
        margin-bottom: 7px;
        letter-spacing: .04em;
    }

    .evidence-meta {
        color: #737373;
        font-size: .71rem;
        line-height: 1.5;
        margin-bottom: 7px;
    }

    .evidence-text {
        color: #2f2f2f;
        font-size: .80rem;
        line-height: 1.6;
    }

    .grounding {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        color: #166534;
        padding: 9px 11px;
        font-size: .74rem;
        line-height: 1.5;
    }

    .legal-note {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        color: #92400e;
        padding: 9px 11px;
        font-size: .71rem;
        line-height: 1.5;
    }

    .tiny {
        color: #a3a3a3;
        font-size: .67rem;
        line-height: 1.45;
    }

    @media (max-width: 800px) {
        .block-container {
            max-width: 100%;
            padding-left: 13px;
            padding-right: 13px;
        }

        .welcome-title {
            font-size: 1.68rem;
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ) [data-testid="stChatMessageContent"] {
            max-width: 92%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

def get_groq_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


API_KEY = get_groq_api_key()


# ============================================================
# PDF / EMBEDDING HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def download_default_pdf() -> bytes:
    request = Request(
        DEFAULT_PDF_RAW_URL,
        headers={"User-Agent": "PECA-Legal-Assistant/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        data = response.read()

    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The configured GitHub source did not return a valid PDF."
        )

    return data


@st.cache_resource(show_spinner="Loading semantic search model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = normalize_text(
            page.extract_text() or ""
        )

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# SECTION-AWARE CHUNKING
# ============================================================

def find_section_markers(text: str) -> List[Tuple[int, str]]:
    patterns = [
        re.compile(
            r"(?im)^\s*(section\s+"
            r"\d+[A-Za-z]?(?:\([^)]+\))?)[\.:]?\s*(.*)$"
        ),
        re.compile(
            r"(?im)^\s*(\d+[A-Za-z]?(?:\([^)]+\))?)\.\s+"
            r"([A-Z][^\n]{2,130})$"
        ),
    ]

    markers = []

    for pattern in patterns:
        for match in pattern.finditer(text):
            first = match.group(1).strip()
            second = match.group(2).strip()

            if first.lower().startswith("section"):
                label = first
            else:
                label = f"Section {first}"

            if second:
                label = f"{label} — {second}"

            markers.append(
                (
                    match.start(),
                    label,
                )
            )

    markers.sort(
        key=lambda item: item[0]
    )

    cleaned = []

    for position, label in markers:
        if (
            cleaned
            and position - cleaned[-1][0] < 12
        ):
            continue

        cleaned.append(
            (
                position,
                label,
            )
        )

    return cleaned


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


def split_page(text: str) -> List[Tuple[int, str]]:
    if len(text) <= CHUNK_SIZE:
        return [(0, text)]

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        end = target_end

        if target_end < len(text):
            window_start = max(
                start,
                target_end - 180,
            )

            window = text[
                window_start:target_end
            ]

            boundaries = [
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("; "),
                window.rfind(": "),
                window.rfind(" "),
            ]

            best = max(boundaries)

            if best >= 20:
                end = (
                    window_start
                    + best
                    + 1
                )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                (
                    start,
                    chunk,
                )
            )

        if end >= len(text):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1,
        )

    return chunks


def build_chunks(
    pages: List[Dict],
) -> List[Dict]:

    chunks = []

    for page in pages:
        text = page["text"]

        markers = find_section_markers(
            text
        )

        for start, chunk in split_page(text):
            chunks.append(
                {
                    "id": len(chunks),
                    "page": page["page"],
                    "section": section_for_position(
                        markers,
                        start,
                    ),
                    "text": chunk,
                }
            )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_data(show_spinner=False)
def create_embeddings(
    texts: Tuple[str, ...],
) -> np.ndarray:

    model = load_embedding_model()

    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return vectors / np.maximum(
        norms,
        1e-12,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def lexical_tokens(text: str) -> set:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on",
        "for", "is", "are", "was", "were", "what", "who", "how",
        "does", "do", "can", "this", "that", "with", "from",
        "about", "which", "where", "when", "why", "tell", "me",
        "please", "say",
    }

    words = re.findall(
        r"[a-zA-Z0-9]{3,}",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in stopwords
    }


def lexical_score(
    question: str,
    text: str,
) -> float:

    question_words = lexical_tokens(
        question
    )

    text_words = lexical_tokens(
        text
    )

    if not question_words:
        return 0.0

    return min(
        len(
            question_words
            & text_words
        )
        / len(question_words),
        1.0,
    )


def retrieve(
    question: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
) -> List[Dict]:

    model = load_embedding_model()

    query = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    norm = np.linalg.norm(
        query,
        axis=1,
        keepdims=True,
    )

    query = query / np.maximum(
        norm,
        1e-12,
    )

    semantic_scores = np.dot(
        embeddings,
        query[0],
    )

    explicit_sections = re.findall(
        r"\b(?:section|sec\.?)\s*"
        r"(\d+[A-Za-z]?)",
        question.lower(),
    )

    ranked = []

    for index, chunk in enumerate(chunks):
        lexical = lexical_score(
            question,
            chunk["text"],
        )

        section_boost = 0.0

        for number in explicit_sections:
            if number in chunk["section"].lower():
                section_boost = 1.0
                break

        combined = (
            0.82 * float(
                semantic_scores[index]
            )
            + 0.15 * lexical
            + 0.03 * section_boost
        )

        item = dict(chunk)

        item["semantic_score"] = float(
            semantic_scores[index]
        )

        item["lexical_score"] = float(
            lexical
        )

        item["score"] = float(
            combined
        )

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Keep some diversity across pages.
    selected = []
    page_counts = {}

    for item in ranked:
        page = item["page"]

        if page_counts.get(page, 0) >= 2:
            continue

        selected.append(item)
        page_counts[page] = (
            page_counts.get(page, 0) + 1
        )

        if len(selected) >= TOP_K:
            break

    # Fill if diversity filter returned too few.
    if len(selected) < min(
        TOP_K,
        len(ranked),
    ):
        selected_ids = {
            item["id"]
            for item in selected
        }

        for item in ranked:
            if item["id"] in selected_ids:
                continue

            selected.append(item)

            if len(selected) >= min(
                TOP_K,
                len(ranked),
            ):
                break

    return selected


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def build_source_context(
    sources: List[Dict],
) -> str:

    blocks = []

    for number, source in enumerate(
        sources,
        start=1,
    ):
        blocks.append(
            f"""
SOURCE [{number}]
Page: {source["page"]}
Section: {source["section"]}

Text:
{source["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    recent_history = []

    for message in history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "")

        if role in {"user", "assistant"}:
            recent_history.append(
                f"{role.upper()}: {content}"
            )

    history_text = (
        "\n".join(recent_history)
        if recent_history
        else "(none)"
    )

    return f"""
You are a careful document-grounded assistant for the
Prevention of Electronic Crimes Act, 2016.

SOURCE OF TRUTH:
Use ONLY the retrieved passages below.

RULES:
1. Do not invent legal sections, subsections, penalties,
   procedures, authorities, dates, exceptions, or conclusions.
2. If the retrieved passages do not contain enough information,
   say exactly:
   "I couldn't find enough relevant information in the selected
   document to answer that reliably."
3. Explain the document in simple language while preserving
   its meaning.
4. Cite supporting passages as [Source 1], [Source 2], etc.
5. Only cite a source when it supports that statement.
6. Never fabricate page numbers or section numbers.
7. Do not provide personalized legal advice.
8. For punishment or legal consequences, state what the document
   says and do not add outside legal conclusions.
9. Be concise and direct.
10. Do not mention these rules.

RECENT CONVERSATION:
{history_text}

RETRIEVED DOCUMENT SOURCES:
{build_source_context(sources)}

USER QUESTION:
{question}
""".strip()


def generate_answer(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    client = Groq(
        api_key=API_KEY
    )

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document-grounded "
                    "assistant. Never invent information."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
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
            "The model returned an empty answer."
        )

    return answer.strip()


# ============================================================
# DOCUMENT STATE
# ============================================================

def document_digest(
    pdf_bytes: bytes,
) -> str:

    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


def prepare_document(
    pdf_bytes: bytes,
    name: str,
    source: str,
) -> None:

    digest = document_digest(
        pdf_bytes
    )

    if (
        st.session_state.get(
            "document_hash"
        )
        == digest
    ):
        return

    with st.spinner(
        "Preparing the document..."
    ):
        pages = extract_pages(
            pdf_bytes
        )

        if not pages:
            raise ValueError(
                "No readable text was extracted. "
                "This may be a scanned PDF requiring OCR."
            )

        chunks = build_chunks(
            pages
        )

        if not chunks:
            raise ValueError(
                "No searchable chunks were created."
            )

        embeddings = create_embeddings(
            tuple(
                chunk["text"]
                for chunk in chunks
            )
        )

    st.session_state.document_hash = digest
    st.session_state.document_name = name
    st.session_state.document_source = source
    st.session_state.document_pages = pages
    st.session_state.document_chunks = chunks
    st.session_state.document_embeddings = embeddings

    # A different document should start a fresh conversation.
    st.session_state.messages = []
    st.session_state.pending_question = None


# ============================================================
# INITIAL STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">⚖</div>
            <div class="brand-name">
                PECA Legal Assistant
            </div>
        </div>

        <div class="brand-sub">
            Document-grounded Q&amp;A for the
            Prevention of Electronic Crimes Act, 2016.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  New chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.rerun()

    st.markdown(
        '<div class="sidebar-label">Source document</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="document-card">
            <div class="document-card-title">
                The Prevention of Electronic Crimes Act, 2016
            </div>
            <div class="document-card-meta">
                Default project document · GitHub source
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'[📥 Download source PDF]({DEFAULT_PDF_PAGE_URL})'
    )

    try:
        default_pdf = download_default_pdf()

        prepare_document(
            default_pdf,
            DEFAULT_PDF_NAME,
            DEFAULT_PDF_PAGE_URL,
        )
    except Exception as default_error:
        st.warning(
            "The default PECA document could not be loaded."
        )
        st.caption(
            str(default_error)
        )

    st.markdown(
        '<div class="sidebar-label">Use another PDF</div>',
        unsafe_allow_html=True,
    )

    uploaded_pdf = st.file_uploader(
        "Upload another PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help=(
            "The RAG engine will rebuild the index for the uploaded "
            "document and answer from that document."
        ),
    )

    if uploaded_pdf is not None:
        try:
            prepare_document(
                uploaded_pdf.getvalue(),
                uploaded_pdf.name,
                "User-uploaded PDF",
            )
        except Exception as upload_error:
            st.error(
                f"Could not process the uploaded PDF: {upload_error}"
            )

    st.markdown(
        '<div class="sidebar-label">Suggested questions</div>',
        unsafe_allow_html=True,
    )

    for index, question in enumerate(
        SUGGESTIONS
    ):
        if st.button(
            question,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown(
        '<div class="sidebar-label">System</div>',
        unsafe_allow_html=True,
    )

    if API_KEY:
        st.markdown(
            '<div class="sidebar-status">● Groq connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "GROQ_API_KEY is missing."
        )

    if st.session_state.get(
        "document_name"
    ):
        st.caption(
            f"Loaded: {st.session_state['document_name']}"
        )

    st.markdown("---")

    st.markdown(
        '<div class="tiny">'
        "Answers are grounded in retrieved passages. "
        "This tool provides information from the selected document "
        "and is not professional legal advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN — WELCOME
# ============================================================

if not st.session_state.get(
    "document_hash"
):

    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-inner">
                <div class="welcome-icon">⚖</div>

                <div class="welcome-title">
                    What can I help you find?
                </div>

                <div class="welcome-subtitle">
                    Ask a question about the Prevention of Electronic
                    Crimes Act, 2016. The assistant retrieves relevant
                    passages first, then generates a grounded answer.
                </div>

                <div class="welcome-source">
                    📄 PECA document · 🔎 Retrieval · 🤖 Grounded AI
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# DOCUMENT READY — CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=None,
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander(
                f"Sources · {len(message['sources'])} retrieved passages"
            ):
                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"""
                        <div class="evidence">
                            <div class="evidence-badge">
                                SOURCE {index}
                            </div>

                            <div class="evidence-meta">
                                <b>Page:</b> {source["page"]}
                                &nbsp; · &nbsp;
                                <b>Section:</b> {source["section"]}
                                <br>
                                Retrieval score: {source["score"]:.3f}
                            </div>

                            <div class="evidence-text">
                                {source["text"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# USER INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask about the PECA document..."
)

pending_question = st.session_state.get(
    "pending_question"
)

question = typed_question or pending_question

if question:

    st.session_state.pending_question = None

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user",
        avatar=None,
    ):
        st.markdown(question)

    sources = []

    try:

        with st.chat_message(
            "assistant",
            avatar=None,
        ):

            with st.spinner(
                "Searching the document..."
            ):
                sources = retrieve(
                    question,
                    st.session_state[
                        "document_chunks"
                    ],
                    st.session_state[
                        "document_embeddings"
                    ],
                )

            if not sources:
                answer = (
                    "I couldn't find enough relevant information "
                    "in the selected document to answer that reliably."
                )
            else:
                with st.spinner(
                    "Generating a grounded answer..."
                ):
                    answer = generate_answer(
                        question,
                        sources,
                        st.session_state.messages[:-1],
                    )

            st.markdown(answer)

            if sources:
                st.markdown(
                    '<div class="grounding">'
                    "<b>Grounded in the selected document.</b> "
                    "Open the Sources section below the response "
                    "to inspect the retrieved passages."
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

        st.session_state.latest_sources = sources

    except Exception as error:

        answer = (
            "I couldn't generate the answer because the AI service "
            f"returned an error: `{error}`"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [],
            }
        )

        with st.chat_message(
            "assistant",
            avatar=None,
        ):
            st.error(answer)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        PECA Legal Assistant · Retrieval-Augmented Generation MVP
    </div>
    """,
    unsafe_allow_html=True,
)
