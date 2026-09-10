
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
    page_title="PECA Legal Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "PECA Legal Assistant"

# Project source PDF
DEFAULT_PDF_NAME = "PECA 2026.pdf"
DEFAULT_PDF_URL = (
    "https://github.com/codewithMeesum/Peca-Rag_Assistant/blob/main/"
    "PECA%202026.pdf"
)
DEFAULT_PDF_RAW_URL = (
    "https://raw.githubusercontent.com/codewithMeesum/"
    "Peca-Rag_Assistant/main/PECA%202026.pdf"
)

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# Retrieval
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K = 6
MIN_RETRIEVAL_SCORE = 0.18

SUGGESTIONS = [
    "What does the Act say about cyber harassment?",
    "What punishment is mentioned for cyber-stalking?",
    "Which section discusses cyber-bullying?",
    "What does the Act say about complaints?",
    "What powers does the Authority have?",
]


# ============================================================
# UI — CHATGPT-INSPIRED
# ============================================================

st.markdown(
    """
    <style>
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
        border-right: 1px solid #e5e5e5;
    }

    [data-testid="stSidebar"] * {
        color: #171717 !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg {
        color: #333333 !important;
        fill: #333333 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        background: #ffffff !important;
        border: 1px solid #e4e4e4 !important;
        border-radius: 8px !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.1rem;
        padding-bottom: 8rem;
    }

    /* Sidebar */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 3px;
    }

    .brand-icon {
        width: 32px;
        height: 32px;
        border-radius: 9px;
        background: #0d7a46;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex: 0 0 auto;
    }

    .brand-name {
        color: #171717 !important;
        font-size: .94rem;
        font-weight: 760;
    }

    .brand-caption {
        color: #737373 !important;
        font-size: .69rem;
        line-height: 1.45;
        margin: 0 0 13px 42px;
    }

    .side-label {
        color: #737373 !important;
        font-size: .66rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 16px 2px 7px;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 10px 11px;
        margin-bottom: 7px;
    }

    .source-title {
        color: #262626 !important;
        font-size: .77rem;
        font-weight: 700;
        line-height: 1.4;
    }

    .source-meta {
        color: #808080 !important;
        font-size: .67rem;
        margin-top: 2px;
    }

    .status {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 7px 9px;
        color: #047857 !important;
        font-size: .70rem;
        font-weight: 700;
    }

    .tiny {
        color: #777777 !important;
        font-size: .67rem;
        line-height: 1.5;
    }

    /* Welcome */
    .welcome {
        min-height: 62vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 15px 10px 45px;
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
        margin-bottom: 18px;
        box-shadow: 0 9px 28px rgba(13,122,70,.12);
    }

    .welcome-title {
        color: #171717 !important;
        font-size: 2.02rem;
        line-height: 1.1;
        font-weight: 780;
        letter-spacing: -.038em;
        margin-bottom: 9px;
    }

    .welcome-text {
        color: #737373 !important;
        font-size: .92rem;
        line-height: 1.62;
    }

    .welcome-pill {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 11px;
        border-radius: 999px;
        border: 1px solid #e8e8e8;
        background: #fafafa;
        color: #555555 !important;
        font-size: .70rem;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 10px 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
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

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border-top: 1px solid #eeeeee;
        padding-top: 10px;
    }

    [data-testid="stChatInput"] textarea {
        color: #171717 !important;
        background: #ffffff !important;
        border: 1px solid #d7d7d7 !important;
        border-radius: 18px !important;
        box-shadow: 0 3px 18px rgba(0,0,0,.045);
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #bdbdbd !important;
        box-shadow: 0 0 0 1px #bdbdbd !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9a9a9a !important;
    }

    /* Buttons */
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

    /* Source evidence */
    .evidence {
        background: #fafafa;
        border: 1px solid #e7e7e7;
        border-radius: 12px;
        padding: 12px 13px;
        margin: 7px 0;
    }

    .evidence-badge {
        display: inline-block;
        padding: 3px 7px;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857 !important;
        border: 1px solid #bbf7d0;
        font-size: .62rem;
        font-weight: 800;
        letter-spacing: .04em;
        margin-bottom: 7px;
    }

    .evidence-meta {
        color: #737373 !important;
        font-size: .71rem;
        line-height: 1.5;
        margin-bottom: 7px;
    }

    .evidence-text {
        color: #2f2f2f !important;
        font-size: .81rem;
        line-height: 1.6;
    }

    .grounded {
        margin-top: 9px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534 !important;
        border-radius: 10px;
        padding: 8px 10px;
        font-size: .73rem;
        line-height: 1.5;
    }

    .legal {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e !important;
        border-radius: 10px;
        padding: 9px 10px;
        font-size: .71rem;
        line-height: 1.5;
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
        secret_key = st.secrets.get("GROQ_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


API_KEY = get_groq_api_key()


# ============================================================
# DOCUMENT HELPERS
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
            "The configured GitHub file is not a valid PDF."
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

def find_section_markers(
    text: str,
) -> List[Tuple[int, str]]:

    patterns = [
        re.compile(
            r"(?im)^\s*(section\s+"
            r"\d+[A-Za-z]?(?:\([^)]+\))?)"
            r"[\.:]?\s*(.*)$"
        ),
        re.compile(
            r"(?im)^\s*(\d+[A-Za-z]?"
            r"(?:\([^)]+\))?)\.\s+"
            r"([A-Z][^\n]{2,130})$"
        ),
    ]

    markers = []

    for pattern in patterns:
        for match in pattern.finditer(text):
            first = match.group(1).strip()
            second = match.group(2).strip()

            if first.lower().startswith(
                "section"
            ):
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
        key=lambda pair: pair[0]
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


def section_at(
    markers: List[Tuple[int, str]],
    position: int,
) -> str:

    current = "Section not detected"

    for marker_position, label in markers:
        if marker_position > position:
            break

        current = label

    return current


def split_page(
    text: str,
) -> List[Tuple[int, str]]:

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
                    "section": section_at(
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

@st.cache_resource(
    show_spinner="Loading semantic search model..."
)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


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
    stop_words = {
        "the", "a", "an", "and", "or", "of", "to",
        "in", "on", "for", "is", "are", "was", "were",
        "what", "who", "how", "does", "do", "can",
        "this", "that", "with", "from", "about",
        "which", "where", "when", "why", "tell",
        "me", "please", "say", "has", "have", "had",
    }

    words = re.findall(
        r"[a-zA-Z0-9]{3,}",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in stop_words
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
        ) / len(question_words),
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

    requested_sections = re.findall(
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

        for number in requested_sections:
            if number in chunk["section"].lower():
                section_boost = 1.0
                break

        score = (
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
            score
        )

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Diversity across pages
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
# GROQ / GROUNDED GENERATION
# ============================================================

def build_context(
    sources: List[Dict],
) -> str:

    blocks = []

    for index, source in enumerate(
        sources,
        start=1,
    ):
        blocks.append(
            f"""
SOURCE [{index}]
Page: {source["page"]}
Section: {source["section"]}

Original document text:
{source["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(
    question: str,
    sources: List[Dict],
    history: List[Dict],
    language: str,
    answer_style: str,
) -> str:

    previous = []

    for message in history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "")

        if role in {"user", "assistant"}:
            previous.append(
                f"{role.upper()}: {content}"
            )

    history_text = (
        "\n".join(previous)
        if previous
        else "(none)"
    )

    if language == "English":
        language_rule = (
            "Answer in clear, natural English."
        )
    elif language == "Urdu":
        language_rule = (
            "Answer in clear, natural Urdu. Preserve the correct "
            "meaning of legal terms. When useful, include the original "
            "English legal term in parentheses."
        )
    else:
        language_rule = (
            "Answer in clear, natural Roman Urdu written in Latin "
            "script. Preserve important English legal terms when "
            "that improves clarity. Do not use Urdu script."
        )

    if answer_style == "Simple":
        style_rule = (
            "Use simple language for an ordinary reader. "
            "Prefer short sections and bullets."
        )
    else:
        style_rule = (
            "Give a more detailed explanation while preserving the "
            "meaning of the source document."
        )

    return f"""
You are a careful document-grounded information assistant for the
Prevention of Electronic Crimes Act, 2016.

SOURCE OF TRUTH:
The only authoritative material for this answer is the retrieved
document text supplied below.

STRICT ACCURACY RULES:
1. Use only the supplied document context.
2. Never invent a section, subsection, penalty, authority, procedure,
   date, exception, or legal conclusion.
3. Do not add outside law or current legal advice.
4. If the retrieved context does not support the answer, say:
   "I couldn't find enough relevant information in the selected document
   to answer that reliably."
5. Never fabricate a page number or section number.
6. Cite supporting statements as [Source 1], [Source 2], etc.
7. Only cite a source when it supports the statement.
8. Explain the document without changing its legal meaning.
9. Do not present the response as personalized legal advice.
10. {language_rule}
11. {style_rule}

STRUCTURED ANSWER RULE:
Adapt the structure to the question.
- Definition → summary + relevant provision + sources.
- Offence/punishment → summary + provision + consequence + sources.
- Practical question → clear steps + important note + sources.
- Comparison → compact table when useful.
- Do not invent practical actions that are not supported by the document.

RECENT CONVERSATION:
{history_text}

RETRIEVED DOCUMENT CONTEXT:
{build_context(sources)}

USER QUESTION:
{question}
""".strip()


def generate_answer(
    question: str,
    sources: List[Dict],
    history: List[Dict],
    language: str,
    answer_style: str,
) -> str:

    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit "
            "Cloud → Settings → Secrets."
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
                    "You are a precise, document-grounded "
                    "assistant. Never invent information."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
                    question,
                    sources,
                    history,
                    language,
                    answer_style,
                ),
            },
        ],
        temperature=0.1,
        max_completion_tokens=900,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The model returned an empty answer."
        )

    return content.strip()


# ============================================================
# DOCUMENT STATE
# ============================================================

def prepare_document(
    pdf_bytes: bytes,
    name: str,
    source: str,
) -> None:

    digest = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if (
        st.session_state.get(
            "document_hash"
        )
        == digest
    ):
        return

    with st.spinner(
        "Preparing the document for search..."
    ):
        pages = extract_pages(
            pdf_bytes
        )

        if not pages:
            raise ValueError(
                "No readable text was found. "
                "A scanned PDF may require OCR."
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
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.latest_sources = []


def reset_document_state() -> None:
    for key in [
        "document_hash",
        "document_name",
        "document_source",
        "document_pages",
        "document_chunks",
        "document_embeddings",
        "messages",
        "pending_question",
        "latest_sources",
    ]:
        st.session_state.pop(
            key,
            None
        )


# ============================================================
# INITIAL STATE
# ============================================================

for key, value in {
    "messages": [],
    "pending_question": None,
    "latest_sources": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">⚖</div>
            <div class="brand-name">
                PECA Legal Assistant
            </div>
        </div>

        <div class="brand-caption">
            Document-grounded Q&amp;A for the Prevention of
            Electronic Crimes Act, 2016.
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
        '<div class="side-label">Source document</div>',
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Document source",
        [
            "PECA document",
            "Upload PDF",
        ],
        horizontal=False,
        label_visibility="collapsed",
        key="source_mode",
    )

    # --------------------------------------------------------
    # Default PECA mode
    # --------------------------------------------------------
    if source_mode == "PECA document":

        st.markdown(
            """
            <div class="source-card">
                <div class="source-title">
                    Prevention of Electronic Crimes Act, 2016
                </div>
                <div class="source-meta">
                    Default project source · GitHub PDF
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'[📥 Download source PDF]({DEFAULT_PDF_URL})'
        )

        try:
            default_pdf = download_default_pdf()

            prepare_document(
                default_pdf,
                DEFAULT_PDF_NAME,
                DEFAULT_PDF_URL,
            )

        except Exception as default_error:
            st.error(
                "Could not load the default PECA PDF."
            )
            st.caption(
                str(default_error)
            )

    # --------------------------------------------------------
    # Custom PDF mode
    # --------------------------------------------------------
    else:

        st.markdown(
            '<div class="tiny">'
            "Upload any text-based PDF. The app will build a new "
            "retrieval index and answer from that document only."
            "</div>",
            unsafe_allow_html=True,
        )

        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
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
                    f"Could not process the PDF: {upload_error}"
                )

    st.markdown(
        '<div class="side-label">Answer language</div>',
        unsafe_allow_html=True,
    )

    language = st.radio(
        "Answer language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="language",
    )

    st.markdown(
        '<div class="side-label">Answer style</div>',
        unsafe_allow_html=True,
    )

    answer_style = st.radio(
        "Answer style",
        [
            "Simple",
            "Detailed",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="answer_style",
    )

    st.markdown(
        '<div class="side-label">Try asking</div>',
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
        '<div class="side-label">System</div>',
        unsafe_allow_html=True,
    )

    if API_KEY:
        st.markdown(
            '<div class="status">● Groq connected</div>',
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
        '<div class="legal">'
        "<b>Legal notice:</b> This app provides document-grounded "
        "information and is not a substitute for professional legal advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN WELCOME
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

                <div class="welcome-text">
                    Ask a question about the Prevention of Electronic
                    Crimes Act, 2016. The assistant retrieves relevant
                    passages first, then generates a grounded answer.
                </div>

                <div class="welcome-pill">
                    📄 Document &nbsp;·&nbsp;
                    🔎 Retrieval &nbsp;·&nbsp;
                    🤖 Grounded AI
                </div>

            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# CHAT
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
                                Retrieval:
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
# QUESTION
# ============================================================

typed_question = st.chat_input(
    "Ask about the selected document..."
)

pending_question = st.session_state.get(
    "pending_question"
)

question = (
    typed_question
    or pending_question
)

if question:

    st.session_state.pending_question = None

    # User
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

            if (
                not sources
                or sources[0]["score"]
                < MIN_RETRIEVAL_SCORE
            ):

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
                        language,
                        answer_style,
                    )

            st.markdown(answer)

            if sources:

                st.markdown(
                    '<div class="grounded">'
                    "<b>Grounded in the selected document.</b> "
                    "Open the Sources section attached to this "
                    "answer to inspect the retrieved passages."
                    "</div>",
                    unsafe_allow_html=True,
                )

    except Exception as error:

        answer = (
            "I couldn't generate the answer because the AI service "
            f"returned an error: `{error}`"
        )

        sources = []

        with st.chat_message(
            "assistant",
            avatar=None,
        ):
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    st.session_state.latest_sources = sources


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">'
    "PECA Legal Assistant · Retrieval-Augmented Generation MVP"
    "</div>",
    unsafe_allow_html=True,
)
