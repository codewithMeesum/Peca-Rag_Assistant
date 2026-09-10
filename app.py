
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

WELCOME_QUESTIONS = [
    "What does the Act say about cyber harassment?",
    "What punishment is mentioned for cyber-stalking?",
    "Which section discusses cyber-bullying?",
    "What does the Act say about complaints?",
    "What powers does the Authority have?",
]


# ============================================================
# CHATGPT-STYLE UI
# ============================================================

st.markdown(
    """
    <style>
    /* Remove extra Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,.92);
        border-bottom: 1px solid #f1f1f1;
    }

    [data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }

    [data-testid="stSidebar"] * {
        color: #171717 !important;
    }

    .block-container {
        max-width: 860px;
        padding-top: 1.25rem;
        padding-bottom: 7rem;
    }

    /* Sidebar */
    .sidebar-brand {
        font-weight: 800;
        font-size: 1.02rem;
        color: #171717;
        margin: 2px 0 3px 2px;
    }

    .sidebar-desc {
        color: #737373;
        font-size: .76rem;
        line-height: 1.45;
        margin: 0 2px 15px;
    }

    .sidebar-section {
        color: #737373;
        font-size: .70rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 18px 2px 7px;
    }

    .source-box {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 11px 12px;
        margin-bottom: 10px;
    }

    .source-title {
        font-weight: 700;
        font-size: .82rem;
        margin-bottom: 2px;
    }

    .source-meta {
        color: #737373;
        font-size: .72rem;
    }

    /* Main welcome */
    .welcome {
        min-height: 54vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 30px 10px;
    }

    .welcome-mark {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: #0d7a46;
        color: #ffffff;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 23px;
        margin-bottom: 17px;
    }

    .welcome-title {
        color: #171717;
        font-size: 2.0rem;
        font-weight: 750;
        letter-spacing: -.035em;
        margin-bottom: 7px;
    }

    .welcome-sub {
        color: #737373;
        font-size: .92rem;
        line-height: 1.55;
        max-width: 650px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 12px 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #171717 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 1px 0 !important;
        font-size: .96rem;
        line-height: 1.65;
    }

    [data-testid="stChatMessageContent"] * {
        color: #171717 !important;
    }

    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        justify-content: flex-end;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: #f4f4f4 !important;
        color: #171717 !important;
        border-radius: 18px !important;
        padding: 11px 15px !important;
        max-width: 78%;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] * {
        color: #171717 !important;
    }

    /* Assistant */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
        padding-left: 0 !important;
        max-width: 100%;
    }

    /* Hide default avatars for a cleaner ChatGPT-like look */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* Input */
    [data-testid="stChatInput"] {
        background: #ffffff;
    }

    [data-testid="stChatInput"] textarea {
        color: #171717 !important;
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #a3a3a3 !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: 1px solid transparent;
        background: transparent;
        color: #262626;
        text-align: left;
        font-size: .80rem;
        padding: 8px 10px;
    }

    .stButton > button:hover {
        background: #ececef;
        border-color: transparent;
    }

    /* Source cards */
    .evidence-card {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 13px 14px;
        margin: 7px 0;
        color: #262626;
    }

    .evidence-meta {
        color: #737373;
        font-size: .72rem;
        margin-bottom: 7px;
    }

    .evidence-text {
        color: #262626;
        font-size: .82rem;
        line-height: 1.6;
    }

    .grounded {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        color: #166534;
        padding: 9px 11px;
        font-size: .76rem;
        line-height: 1.5;
    }

    .legal {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        color: #92400e;
        padding: 9px 11px;
        font-size: .74rem;
        line-height: 1.5;
    }

    .tiny {
        color: #737373;
        font-size: .72rem;
    }

    @media (max-width: 850px) {
        .block-container {
            max-width: 100%;
        }

        .welcome-title {
            font-size: 1.7rem;
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
# API
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
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading search model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# PDF
# ============================================================

@st.cache_data(show_spinner=False)
def download_default_pdf() -> bytes:
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
# CHUNKING + SECTION DETECTION
# ============================================================

def section_markers(text: str) -> List[Tuple[int, str]]:
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

    markers.sort(key=lambda item: item[0])

    cleaned = []

    for position, label in markers:
        if cleaned and position - cleaned[-1][0] < 12:
            continue

        cleaned.append((position, label))

    return cleaned


def section_at(markers: List[Tuple[int, str]], position: int) -> str:
    current = "Section not detected"

    for marker_position, label in markers:
        if marker_position > position:
            break
        current = label

    return current


def split_page(text: str) -> List[Tuple[int, str]]:
    if len(text) <= CHUNK_SIZE:
        return [(0, text)]

    output = []
    start = 0

    while start < len(text):
        target = min(
            start + CHUNK_SIZE,
            len(text),
        )

        end = target

        if target < len(text):
            window_start = max(
                start,
                target - 180,
            )

            window = text[window_start:target]

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
            output.append((start, chunk))

        if end >= len(text):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1,
        )

    return output


def build_chunks(pages: List[Dict]) -> List[Dict]:
    chunks = []

    for page in pages:
        markers = section_markers(page["text"])

        for chunk_start, chunk_text in split_page(
            page["text"]
        ):
            chunks.append(
                {
                    "id": len(chunks),
                    "page": page["page"],
                    "section": section_at(
                        markers,
                        chunk_start,
                    ),
                    "text": chunk_text,
                }
            )

    return chunks


# ============================================================
# EMBEDDINGS + RETRIEVAL
# ============================================================

@st.cache_data(show_spinner=False)
def embed_texts(texts: Tuple[str, ...]) -> np.ndarray:
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


def tokens(text: str) -> set:
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
    q = tokens(question)
    t = tokens(text)

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

    query = query / np.maximum(norm, 1e-12)

    semantic = np.dot(
        embeddings,
        query[0],
    )

    ranked = []

    section_numbers = re.findall(
        r"\b(?:section|sec\.?)\s*(\d+[A-Za-z]?)",
        question.lower(),
    )

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

        score = (
            0.82 * float(semantic[index])
            + 0.15 * lexical
            + 0.03 * section_boost
        )

        item = dict(chunk)
        item["semantic_score"] = float(
            semantic[index]
        )
        item["lexical_score"] = float(
            lexical
        )
        item["score"] = float(score)

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Keep a little source diversity across pages.
    selected = []
    page_counts = {}

    for item in ranked:
        page = item["page"]

        if page_counts.get(page, 0) >= 2:
            continue

        selected.append(item)
        page_counts[page] = page_counts.get(page, 0) + 1

        if len(selected) >= top_k:
            break

    # Always return top results if diversity filtering was too strict.
    if len(selected) < min(top_k, len(ranked)):
        selected_ids = {item["id"] for item in selected}

        for item in ranked:
            if item["id"] not in selected_ids:
                selected.append(item)

            if len(selected) >= min(
                top_k,
                len(ranked),
            ):
                break

    return selected


# ============================================================
# GROQ
# ============================================================

def source_context(sources: List[Dict]) -> str:
    parts = []

    for item in sources:
        parts.append(
            f"""
SOURCE_ID: {item["id"]}
PAGE: {item["page"]}
SECTION: {item["section"]}

SOURCE_TEXT:
{item["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(parts)


def grounded_prompt(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    previous = []

    for message in history[-6:]:
        if message.get("role") in {"user", "assistant"}:
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
You are a careful document-grounded information assistant for the
Prevention of Electronic Crimes Act, 2016 (PECA).

SOURCE OF TRUTH:
Use only the retrieved passages from the selected PDF.

RULES:
- Never invent a section, subsection, penalty, procedure, authority,
  date, exception, or legal conclusion.
- If the retrieved text does not support the answer, say:
  "I couldn't find enough relevant information in the selected document
  to answer that reliably."
- Explain the source in simple language without changing its meaning.
- Cite a page and section only when it is explicitly present in the source.
- Never fabricate citations.
- Do not provide personalized legal advice.
- For legal consequences, state what the document says.
- Keep the answer concise and useful.

RECENT CHAT:
{history_text}

RETRIEVED SOURCE MATERIAL:
{source_context(sources)}

QUESTION:
{question}
""".strip()


def generate_answer(
    question: str,
    sources: List[Dict],
    history: List[Dict],
) -> str:

    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from supplied document evidence. "
                    "Do not invent legal information."
                ),
            },
            {
                "role": "user",
                "content": grounded_prompt(
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
# DOCUMENT STATE
# ============================================================

def state_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_document(
    pdf_bytes: bytes,
    name: str,
    source: str,
) -> None:

    digest = state_hash(pdf_bytes)

    if st.session_state.get("document_hash") == digest:
        return

    with st.spinner("Preparing the document..."):
        pages = extract_pages(pdf_bytes)

        if not pages:
            raise ValueError(
                "No readable text was found in the PDF."
            )

        chunks = build_chunks(pages)

        if not chunks:
            raise ValueError(
                "No searchable text chunks were created."
            )

        embeddings = embed_texts(
            tuple(
                chunk["text"]
                for chunk in chunks
            )
        )

    st.session_state.document_hash = digest
    st.session_state.document_name = name
    st.session_state.document_source = source
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
        '<div class="sidebar-brand">🇵🇰 PECA Legal Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-desc">'
        "Grounded Q&amp;A over the Prevention of Electronic Crimes Act, 2016."
        "</div>",
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
        '<div class="sidebar-section">Source document</div>',
        unsafe_allow_html=True,
    )

    document_mode = st.radio(
        "Document",
        [
            "PECA document",
            "Upload PDF",
        ],
        label_visibility="collapsed",
    )

    if document_mode == "PECA document":
        st.markdown(
            f"""
            <div class="source-box">
                <div class="source-title">
                    The Prevention of Electronic Crimes Act, 2016
                </div>
                <div class="source-meta">
                    Project source · PDF
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'[📥 Download source PDF]({PDF_BLOB_URL})'
        )

        try:
            prepare_document(
                download_default_pdf(),
                "PECA 2026.pdf",
                PDF_BLOB_URL,
            )
        except Exception as exc:
            st.error(
                "Could not load the default source PDF."
            )
            st.caption(str(exc))

            st.markdown(
                '<div class="sidebar-section">Fallback</div>',
                unsafe_allow_html=True,
            )

            fallback = st.file_uploader(
                "Upload PECA PDF",
                type=["pdf"],
                label_visibility="collapsed",
            )

            if fallback is not None:
                prepare_document(
                    fallback.getvalue(),
                    fallback.name,
                    "User upload",
                )

    else:
        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                prepare_document(
                    uploaded.getvalue(),
                    uploaded.name,
                    "User upload",
                )
            except Exception as exc:
                st.error(
                    f"Could not process the PDF: {exc}"
                )

    st.markdown(
        '<div class="sidebar-section">Suggested questions</div>',
        unsafe_allow_html=True,
    )

    for index, question in enumerate(WELCOME_QUESTIONS):
        if st.button(
            question,
            key=f"question_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown(
        '<div class="sidebar-section">Status</div>',
        unsafe_allow_html=True,
    )

    if API_KEY:
        st.success("Groq connected")
    else:
        st.error("Groq API key missing")

    if st.session_state.get("document_name"):
        st.caption(
            f"Loaded: {st.session_state['document_name']}"
        )

    st.markdown(
        '<div class="legal">'
        "<b>Legal notice:</b> This app is an informational tool "
        "grounded in the selected document. It is not professional legal advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# WELCOME STATE
# ============================================================

if not st.session_state.get("document_hash"):
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-mark">⚖</div>
            <div class="welcome-title">
                PECA Legal Information Assistant
            </div>
            <div class="welcome-sub">
                Ask questions about the Prevention of Electronic Crimes Act,
                2016. Answers are grounded in the selected source document
                and the retrieved evidence is available for verification.
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
    with st.chat_message(
        message["role"],
        avatar=None,
    ):
        st.markdown(message["content"])


# ============================================================
# INPUT
# ============================================================

pending = st.session_state.get(
    "pending_question"
)

typed = st.chat_input(
    "Ask about the PECA document..."
)

question = typed or pending

if question:
    st.session_state.pending_question = None

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
                "Searching the PECA document..."
            ):
                sources = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    top_k=TOP_K,
                )

            if not sources:
                answer = (
                    "I couldn't find enough relevant information in the "
                    "selected document to answer that reliably."
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

    except Exception as exc:
        answer = (
            "I couldn't generate the answer because the AI service "
            f"returned an error: `{exc}`"
        )

        with st.chat_message(
            "assistant",
            avatar=None,
        ):
            st.error(answer)

        sources = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.latest_sources = sources


# ============================================================
# SOURCES
# ============================================================

sources = st.session_state.get(
    "latest_sources",
    [],
)

if sources:
    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section" style="color:#737373;">'
        "Sources used for the latest answer"
        "</div>",
        unsafe_allow_html=True,
    )

    for source in sources:
        with st.expander(
            f"Page {source['page']} · "
            f"{source['section']} · "
            f"relevance {source['score']:.3f}"
        ):
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-meta">
                        <b>Page:</b> {source["page"]}
                        &nbsp; · &nbsp;
                        <b>Section:</b> {source["section"]}
                        <br>
                        Semantic similarity:
                        {source["semantic_score"]:.3f}
                        &nbsp; · &nbsp;
                        Lexical match:
                        {source["lexical_score"]:.3f}
                    </div>

                    <div class="evidence-text">
                        {source["text"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="grounded">'
        "<b>Why this matters:</b> the source passages above are the "
        "document evidence provided to the language model."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="tiny" style="text-align:center;margin-top:24px;">'
    "PECA Legal Information Assistant · Document-grounded RAG MVP"
    "</div>",
    unsafe_allow_html=True,
)
