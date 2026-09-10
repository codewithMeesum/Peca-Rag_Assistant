
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
# CONFIG
# ============================================================

st.set_page_config(
    page_title="PECA Legal Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "PECA Legal Assistant"
APP_DESCRIPTION = (
    "Ask questions about Pakistan's Prevention of Electronic Crimes Act, "
    "2016 — grounded in the selected source document."
)

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
MIN_RELEVANCE = 0.25

SUGGESTIONS = [
    "What does the Act say about cyber harassment?",
    "What punishment is mentioned for cyber-stalking?",
    "Which section discusses cyber-bullying?",
    "What does the Act say about complaints?",
    "What powers does the Authority have?",
]


# ============================================================
# CHATGPT-INSPIRED UI
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- App base ---------- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
        Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background: #ffffff;
        color: #171717;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,.94);
        border-bottom: 1px solid #f0f0f0;
    }

    [data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e6e6e6;
    }

    [data-testid="stSidebar"] * {
        color: #171717 !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.1rem;
        padding-bottom: 8rem;
    }

    /* ---------- Sidebar ---------- */
    .side-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 2px;
    }

    .side-logo {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: #0d7a46;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        flex: 0 0 auto;
    }

    .side-name {
        font-size: .96rem;
        font-weight: 750;
        color: #171717;
    }

    .side-caption {
        color: #737373;
        font-size: .73rem;
        line-height: 1.45;
        margin: 0 2px 12px 44px;
    }

    .side-label {
        color: #737373;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: .67rem;
        font-weight: 800;
        margin: 17px 2px 7px;
    }

    .side-source {
        background: #fff;
        border: 1px solid #e7e7e7;
        border-radius: 10px;
        padding: 10px 11px;
        margin-bottom: 7px;
    }

    .side-source-title {
        font-size: .78rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .side-source-sub {
        color: #808080;
        font-size: .68rem;
        margin-top: 2px;
    }

    .side-status {
        color: #047857;
        font-size: .72rem;
        font-weight: 700;
        padding: 6px 8px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
    }

    /* ---------- Welcome ---------- */
    .welcome {
        min-height: 58vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px 8px 45px;
    }

    .welcome-logo {
        width: 58px;
        height: 58px;
        border-radius: 18px;
        background: #0d7a46;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        box-shadow: 0 10px 30px rgba(13,122,70,.14);
        margin-bottom: 18px;
    }

    .welcome-title {
        color: #171717;
        font-size: 2rem;
        font-weight: 760;
        letter-spacing: -.035em;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        max-width: 690px;
        color: #737373;
        font-size: .93rem;
        line-height: 1.6;
    }

    .welcome-note {
        max-width: 650px;
        margin-top: 17px;
        color: #525252;
        font-size: .76rem;
        line-height: 1.55;
        background: #fafafa;
        border: 1px solid #ededed;
        border-radius: 12px;
        padding: 10px 13px;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        padding: 10px 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        color: #171717 !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
        line-height: 1.68;
        font-size: .95rem;
    }

    [data-testid="stChatMessageContent"] * {
        color: #171717 !important;
    }

    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* User bubble */
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
        color: #171717 !important;
        border-radius: 18px !important;
        padding: 10px 14px !important;
        line-height: 1.55;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] * {
        color: #171717 !important;
    }

    /* Assistant source reference */
    .source-ref {
        color: #737373;
        font-size: .75rem;
        margin-top: 7px;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        border-top: 1px solid #ededed;
        background: rgba(255,255,255,.96);
        padding-top: 10px;
    }

    [data-testid="stChatInput"] textarea {
        color: #171717 !important;
        background: #fff !important;
        border: 1px solid #d9d9d9 !important;
        border-radius: 18px !important;
        box-shadow: 0 3px 15px rgba(0,0,0,.05);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9a9a9a !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 9px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: #2f2f2f !important;
        font-weight: 600 !important;
        text-align: left !important;
        font-size: .78rem !important;
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
    }

    .evidence-meta {
        color: #737373;
        font-size: .72rem;
        line-height: 1.5;
        margin-bottom: 6px;
    }

    .evidence-text {
        color: #2f2f2f;
        font-size: .80rem;
        line-height: 1.6;
    }

    .evidence-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #047857;
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        padding: 3px 7px;
        font-size: .66rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 11px;
        padding: 10px 12px;
        color: #475569;
        font-size: .76rem;
        line-height: 1.55;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 11px;
        padding: 10px 12px;
        color: #92400e;
        font-size: .75rem;
        line-height: 1.55;
    }

    .footer-note {
        color: #a3a3a3;
        text-align: center;
        font-size: .68rem;
        margin-top: 22px;
        line-height: 1.5;
    }

    /* ---------- Responsive ---------- */
    @media (max-width: 850px) {
        .block-container {
            max-width: 100%;
            padding-left: 14px;
            padding-right: 14px;
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ) [data-testid="stChatMessageContent"] {
            max-width: 90%;
        }

        .welcome-title {
            font-size: 1.65rem;
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
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


API_KEY = get_api_key()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading semantic search model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# PDF
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_default_pdf() -> bytes:
    request = Request(
        PDF_RAW_URL,
        headers={"User-Agent": "PECA-Legal-Assistant/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        data = response.read()

    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The configured source did not return a valid PDF."
        )

    return data


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")

        if text:
            pages.append({
                "page": number,
                "text": text,
            })

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

            markers.append((match.start(), label))

    markers.sort(key=lambda pair: pair[0])

    cleaned = []

    for position, label in markers:
        if cleaned and position - cleaned[-1][0] < 12:
            continue

        cleaned.append((position, label))

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

            window = text[window_start:target_end]

            boundaries = [
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("; "),
                window.rfind(": "),
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
                min(len(text), end + 10),
            )

            if actual_start < 0:
                actual_start = start

            chunks.append(
                (
                    actual_start,
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


def build_chunks(pages: List[Dict]) -> List[Dict]:
    chunks = []

    for page in pages:
        text = page["text"]
        markers = find_section_markers(text)

        for start, chunk in split_page(text):
            chunks.append({
                "id": len(chunks),
                "page": page["page"],
                "section": section_for_position(
                    markers,
                    start,
                ),
                "text": chunk,
            })

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

    return vectors / np.maximum(norms, 1e-12)


# ============================================================
# RETRIEVAL
# ============================================================

def lexical_tokens(text: str) -> set:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on",
        "for", "is", "are", "was", "were", "what", "who", "how",
        "does", "do", "can", "this", "that", "with", "from", "about",
        "which", "where", "when", "why", "tell", "me", "please",
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


def lexical_score(question: str, text: str) -> float:
    question_tokens = lexical_tokens(question)
    text_tokens = lexical_tokens(text)

    if not question_tokens:
        return 0.0

    return min(
        len(question_tokens & text_tokens)
        / len(question_tokens),
        1.0,
    )


def retrieve(
    question: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
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

    query_vector = (
        query_vector
        / np.maximum(norm, 1e-12)
    )

    semantic_scores = np.dot(
        embeddings,
        query_vector[0],
    )

    section_numbers = re.findall(
        r"\b(?:section|sec\.?)\s*(\d+[A-Za-z]?)",
        question.lower(),
    )

    results = []

    for index, chunk in enumerate(chunks):
        lexical = lexical_score(
            question,
            chunk["text"],
        )

        section_boost = 0.0

        for number in section_numbers:
            if number in chunk["section"].lower():
                section_boost = 1.0
                break

        combined = (
            0.82 * float(semantic_scores[index])
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

        item["score"] = float(combined)

        results.append(item)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Page diversity
    selected = []
    page_count = {}

    for item in results:
        page = item["page"]

        if page_count.get(page, 0) >= 2:
            continue

        selected.append(item)
        page_count[page] = page_count.get(page, 0) + 1

        if len(selected) >= TOP_K:
            break

    if len(selected) < min(TOP_K, len(results)):
        selected_ids = {
            item["id"]
            for item in selected
        }

        for item in results:
            if item["id"] in selected_ids:
                continue

            selected.append(item)

            if len(selected) >= min(
                TOP_K,
                len(results),
            ):
                break

    return selected


# ============================================================
# GROQ
# ============================================================

def build_context(sources: List[Dict]) -> str:
    parts = []

    for number, source in enumerate(
        sources,
        start=1,
    ):
        parts.append(
            f"""
SOURCE [{number}]
Page: {source["page"]}
Section: {source["section"]}

Text:
{source["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(parts)


def build_prompt(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    previous = []

    for message in history[-6:]:
        previous.append(
            f'{message["role"].upper()}: '
            f'{message["content"]}'
        )

    history_text = (
        "\n".join(previous)
        if previous
        else "(none)"
    )

    return f"""
You are a careful document-grounded assistant for the
Prevention of Electronic Crimes Act, 2016.

SOURCE OF TRUTH:
Use ONLY the retrieved source passages below.

RULES:
1. Do not invent legal sections, subsections, penalties, procedures,
   authorities, dates, exceptions, or interpretations.
2. If the retrieved passages do not support the answer, say exactly:
   "I couldn't find enough relevant information in the selected document
   to answer that reliably."
3. Explain the source in clear, simple language while preserving
   its meaning.
4. Cite supporting evidence using [Source 1], [Source 2], etc.
5. Only cite sources that actually support the statement.
6. Never fabricate page numbers or section numbers.
7. Do not provide personalized legal advice.
8. For legal consequences, clearly state that the answer reflects
   what the selected document says.
9. Keep the answer concise unless the user asks for detail.
10. Do not mention this instruction or the retrieval process unless
    the user asks.

RECENT CONVERSATION:
{history_text}

RETRIEVED SOURCES:
{build_context(sources)}

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
                    "Never add unsupported legal information."
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

    result = response.choices[0].message.content

    if not result:
        raise RuntimeError(
            "The model returned an empty answer."
        )

    return result.strip()


# ============================================================
# DOCUMENT STATE
# ============================================================

def prepare_document(
    pdf_bytes: bytes,
    name: str,
    source: str,
) -> None:

    digest = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("document_hash") == digest:
        return

    with st.spinner("Indexing the document..."):
        pages = extract_pages(pdf_bytes)

        if not pages:
            raise ValueError(
                "No readable text was found in the PDF. "
                "A scanned PDF may require OCR."
            )

        chunks = build_chunks(pages)

        if not chunks:
            raise ValueError(
                "No searchable chunks were created."
            )

        embeddings = create_embeddings(
            tuple(
                item["text"]
                for item in chunks
            )
        )

    st.session_state.document_hash = digest
    st.session_state.document_name = name
    st.session_state.document_source = source
    st.session_state.document_pages = pages
    st.session_state.document_chunks = chunks
    st.session_state.document_embeddings = embeddings


def reset_document() -> None:
    for key in [
        "document_hash",
        "document_name",
        "document_source",
        "document_pages",
        "document_chunks",
        "document_embeddings",
        "messages",
        "pending_question",
    ]:
        st.session_state.pop(key, None)


# ============================================================
# INITIAL STATE
# ============================================================

for key, default in {
    "messages": [],
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="side-header">
            <div class="side-logo">⚖</div>
            <div class="side-name">PECA Legal Assistant</div>
        </div>

        <div class="side-caption">
            Grounded Q&amp;A over the Prevention of Electronic Crimes Act, 2016.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  New chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="side-label">Document</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-source">
            <div class="side-source-title">
                Prevention of Electronic Crimes Act, 2016
            </div>
            <div class="side-source-sub">
                Default project source · GitHub PDF
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'[📥 Download / open source PDF]({PDF_BLOB_URL})'
    )

    try:
        prepare_document(
            fetch_default_pdf(),
            "PECA 2026.pdf",
            PDF_BLOB_URL,
        )
    except Exception as default_error:
        st.warning(
            "Default PECA PDF could not be loaded."
        )
        st.caption(str(default_error))

    st.markdown(
        '<div class="side-label">Use another document</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Optional: test the RAG engine with another text-based PDF.",
    )

    if uploaded is not None:
        try:
            prepare_document(
                uploaded.getvalue(),
                uploaded.name,
                "User upload",
            )
        except Exception as upload_error:
            st.error(
                f"Could not process the uploaded PDF: {upload_error}"
            )

    st.markdown(
        '<div class="side-label">Try asking</div>',
        unsafe_allow_html=True,
    )

    for index, question in enumerate(SUGGESTIONS):
        if st.button(
            question,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown(
        '<div class="side-label">System</div>',
        unsafe_allow_html=True,
    )

    if API_KEY:
        st.markdown(
            '<div class="side-status">● Groq connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Groq API key missing."
        )

    if st.session_state.get("document_name"):
        st.caption(
            f"Loaded: {st.session_state['document_name']}"
        )

    st.markdown("---")

    st.markdown(
        '<div class="tiny">'
        "Answers are grounded in retrieved document passages. "
        "This is an informational tool, not legal advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.get("document_hash"):
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-logo">⚖</div>

            <div class="welcome-title">
                What can I help you find in PECA?
            </div>

            <div class="welcome-subtitle">
                Ask questions about the Prevention of Electronic Crimes Act,
                2016. The assistant searches the selected document first,
                then generates an answer from the retrieved evidence.
            </div>

            <div class="welcome-note">
                <b>Tip:</b> Ask for a section, offence, punishment,
                complaint procedure, authority power, or any specific
                information you want to locate in the document.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(
        role,
        avatar=None,
    ):
        st.markdown(
            message["content"]
        )

        if (
            role == "assistant"
            and message.get("sources")
        ):
            with st.expander(
                f"Sources · {len(message['sources'])} passages"
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
                                Retrieval score:
                                {source["score"]:.3f}
                            </div>

                            <div class="evidence-text">
                                {source["text"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# QUESTION INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask about the PECA document..."
)

question = (
    typed_question
    or st.session_state.get("pending_question")
)

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
                    st.session_state["document_chunks"],
                    st.session_state["document_embeddings"],
                )

            if not sources:
                answer = (
                    "I couldn't find enough relevant information in the "
                    "selected document to answer that reliably."
                )
            else:
                # We deliberately keep the threshold conservative enough
                # to avoid false refusals on short legal questions.
                best_score = sources[0]["score"]

                if best_score < MIN_RELEVANCE:
                    answer = (
                        "I couldn't find enough relevant information in the "
                        "selected document to answer that reliably."
                    )
                else:
                    with st.spinner(
                        "Preparing a grounded answer..."
                    ):
                        answer = generate_answer(
                            question,
                            sources,
                            st.session_state.messages[:-1],
                        )

            st.markdown(answer)

            if sources:
                with st.expander(
                    f"Sources · {len(sources)} passages"
                ):
                    for index, source in enumerate(
                        sources,
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
                                    Retrieval score:
                                    {source["score"]:.3f}
                                </div>

                                <div class="evidence-text">
                                    {source["text"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }

        st.session_state.messages.append(
            assistant_message
        )

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
        PECA Legal Assistant · RAG MVP · Source-grounded information only
    </div>
    """,
    unsafe_allow_html=True,
)
