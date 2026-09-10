import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PECA Legal Information Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K = 5

APP_TITLE = "🇵🇰 PECA Legal Information Assistant"
APP_SUBTITLE = (
    "Ask questions about the Prevention of Electronic Crimes Act, 2016 "
    "using grounded document retrieval."
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .source-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 14px;
        margin: 8px 0;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.86rem;
    }

    .disclaimer {
        border-left: 4px solid #d97706;
        padding: 10px 14px;
        background: rgba(217,119,6,.08);
        border-radius: 6px;
        font-size: 0.9rem;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SECRET / API KEY
# ============================================================


def get_groq_api_key() -> str:
    """Read GROQ_API_KEY from Streamlit Secrets or environment."""
    try:
        secret_value = st.secrets.get("GROQ_API_KEY")
        if secret_value:
            return str(secret_value).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


api_key = get_groq_api_key()

# ============================================================
# MODEL
# ============================================================


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformer embedding model once."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# PDF PROCESSING
# ============================================================


def normalize_text(text: str) -> str:
    """Clean PDF-extracted text without destroying useful legal formatting."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> str:
    """
    Try to identify a PECA-style section reference from a chunk.
    Returns a readable label when one is found.
    """
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


def extract_pdf_pages(pdf_file) -> List[Dict]:
    """Extract page-aware text from a PDF."""
    reader = PdfReader(pdf_file)
    pages: List[Dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = normalize_text(page_text)

        if page_text:
            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                }
            )

    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Create overlapping chunks while preserving text order."""
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Build page-aware chunks so every retrieved result retains its source page.
    """
    chunks: List[Dict] = []

    for page_item in pages:
        page_number = page_item["page"]
        page_text = page_item["text"]

        page_chunks = chunk_text(
            page_text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "id": len(chunks),
                    "page": page_number,
                    "chunk_on_page": chunk_index,
                    "section": detect_section(chunk),
                    "text": chunk,
                }
            )

    return chunks


# ============================================================
# EMBEDDINGS + RETRIEVAL
# ============================================================


@st.cache_data(show_spinner=False)
def embed_chunks(chunk_texts: Tuple[str, ...]) -> np.ndarray:
    """
    Create normalized embeddings for all chunks.
    Caching avoids recomputing them while the same document is active.
    """
    model = load_embedding_model()

    embeddings = model.encode(
        list(chunk_texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def tokenize_for_lexical_match(text: str) -> set:
    """Tokenize content for a small lexical retrieval signal."""
    stop_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "what", "who", "how", "does", "do",
        "can", "this", "that", "with", "from", "about", "under", "by",
    }

    tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return {token for token in tokens if token not in stop_words}


def lexical_score(question: str, chunk: str) -> float:
    """Compute a lightweight word-overlap score."""
    question_tokens = tokenize_for_lexical_match(question)
    chunk_tokens = tokenize_for_lexical_match(chunk)

    if not question_tokens:
        return 0.0

    overlap = len(question_tokens.intersection(chunk_tokens))
    return min(overlap / len(question_tokens), 1.0)


def retrieve_chunks(
    question: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Hybrid retrieval:
      semantic cosine similarity + lightweight lexical overlap.
    """
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
    query_embedding = query_embedding / np.maximum(query_norm, 1e-12)

    semantic_scores = np.dot(
        embeddings,
        query_embedding[0],
    )

    ranked = []

    for index, chunk in enumerate(chunks):
        lexical = lexical_score(question, chunk["text"])

        # Semantic similarity gets the primary weight.
        combined = (0.82 * float(semantic_scores[index])) + (0.18 * lexical)

        item = dict(chunk)
        item["semantic_score"] = float(semantic_scores[index])
        item["lexical_score"] = float(lexical)
        item["score"] = float(combined)

        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return ranked[:top_k]


# ============================================================
# GROQ ANSWERING
# ============================================================


def build_context(retrieved_chunks: List[Dict]) -> str:
    """Build a clearly labeled source context for the LLM."""
    blocks = []

    for item in retrieved_chunks:
        blocks.append(
            f"""
SOURCE ID: {item["id"]}
PAGE: {item["page"]}
SECTION: {item["section"]}

TEXT:
{item["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def build_answer_prompt(
    question: str,
    context: str,
    chat_history: List[Dict],
) -> str:
    """Create a grounded prompt that prevents unsupported legal claims."""
    recent_history = chat_history[-6:]

    history_text = ""
    for message in recent_history:
        role = message.get("role", "")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            history_text += f"{role.upper()}: {content}\n"

    return f"""
You are a careful document-grounded legal information assistant.

DOCUMENT SCOPE:
The only authoritative source for this answer is the retrieved text
from the uploaded Prevention of Electronic Crimes Act, 2016 document.

RULES:
1. Answer ONLY from the supplied document context.
2. Do not invent sections, penalties, authorities, procedures, dates,
   interpretations, or legal conclusions.
3. If the answer is not supported by the supplied context, say:
   "I couldn't find that information in the uploaded document."
4. Explain legal wording in simple language without changing its meaning.
5. When the context contains a section number or page number, cite it
   naturally, for example: "[Section 21, Page 14]".
6. Never create a citation that is not present in the supplied context.
7. Do not present the response as personal legal advice.
8. For questions asking about punishment or legal consequences, clearly
   distinguish what the document states from any explanation.
9. Keep the answer concise unless the user asks for detail.

RECENT CONVERSATION:
{history_text if history_text else "(No previous conversation.)"}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Now answer using only the retrieved document context.
""".strip()


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    chat_history: List[Dict],
) -> str:
    """Call Groq and return a grounded answer."""
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )

    client = Groq(api_key=api_key)
    context = build_context(retrieved_chunks)
    prompt = build_answer_prompt(question, context, chat_history)

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document-grounded assistant. "
                    "Follow the user's request only when it is supported "
                    "by the retrieved document."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_completion_tokens=900,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The model returned an empty response.")

    return content.strip()


# ============================================================
# DOCUMENT STATE
# ============================================================


def file_hash(uploaded_file) -> str:
    """Create a stable hash for the uploaded document."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def reset_document_state() -> None:
    """Clear document-specific application state."""
    keys_to_remove = [
        "doc_hash",
        "doc_name",
        "pages",
        "chunks",
        "embeddings",
        "messages",
        "last_sources",
    ]

    for key in keys_to_remove:
        st.session_state.pop(key, None)


def initialize_document(uploaded_file) -> None:
    """Extract, chunk, and embed a new document."""
    current_hash = file_hash(uploaded_file)

    if st.session_state.get("doc_hash") == current_hash:
        return

    reset_document_state()

    with st.spinner("Reading and indexing the document..."):
        pages = extract_pdf_pages(uploaded_file)

        if not pages:
            raise ValueError(
                "No readable text was extracted from this PDF. "
                "The document may be scanned/image-only and require OCR."
            )

        chunks = build_chunks(pages)

        if not chunks:
            raise ValueError("The document did not produce any searchable chunks.")

        chunk_texts = tuple(chunk["text"] for chunk in chunks)
        embeddings = embed_chunks(chunk_texts)

    st.session_state.doc_hash = current_hash
    st.session_state.doc_name = uploaded_file.name
    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.messages = []
    st.session_state.last_sources = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🇵🇰 PECA Assistant")

    st.caption(
        "Grounded Q&A over the uploaded Prevention of Electronic Crimes Act, 2016."
    )

    if api_key:
        st.success("Groq API key detected.")
    else:
        st.error("Groq API key not found.")

    st.markdown("---")

    uploaded_pdf = st.file_uploader(
        "Upload official PECA PDF",
        type=["pdf"],
        help="Upload the official text-based PECA document.",
    )

    if st.button("🗑️ Reset document", use_container_width=True):
        reset_document_state()
        st.rerun()

    st.markdown("---")

    st.markdown("### Suggested questions")

    suggested_questions = [
        "What does the document say about cyber harassment?",
        "What punishment is mentioned for this offence?",
        "Which section deals with this offence?",
        "What does the Act say about complaints?",
        "What powers does the Authority have?",
    ]

    selected_question = None

    for question in suggested_questions:
        if st.button(
            question,
            use_container_width=True,
            key=f"suggested_{hash(question)}",
        ):
            selected_question = question

    st.markdown("---")

    st.markdown(
        """
        <div class="disclaimer">
        <b>Important:</b> This application provides information grounded
        in the uploaded document. It is not a substitute for professional
        legal advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    f'<div class="main-header">{APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="sub-header">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)

# ============================================================
# NO DOCUMENT STATE
# ============================================================

if uploaded_pdf is None:
    st.info(
        "Upload the official PECA PDF from the sidebar to start asking questions."
    )

    st.markdown(
        """
        ### How it works

        **1. Upload** the official document  
        **2. Extract** page-aware text  
        **3. Chunk** the document  
        **4. Create embeddings**  
        **5. Retrieve relevant passages**  
        **6. Ask the Groq LLM to answer using only those passages**

        **Core pipeline:**  
        `PDF → Chunks → Embeddings → Retrieval → Grounded Answer`
        """
    )

    st.stop()

# ============================================================
# INITIALIZE DOCUMENT
# ============================================================

try:
    initialize_document(uploaded_pdf)
except Exception as exc:
    st.error(f"Document processing failed: {exc}")
    st.stop()

# ============================================================
# DOCUMENT METRICS
# ============================================================

pages = st.session_state["pages"]
chunks = st.session_state["chunks"]

metric_cols = st.columns(3)

with metric_cols[0]:
    st.metric("Pages", len(pages))

with metric_cols[1]:
    st.metric("Searchable chunks", len(chunks))

with metric_cols[2]:
    st.metric("Embedding model", "MiniLM-L6")

st.caption(f"Document: **{st.session_state['doc_name']}**")

st.markdown("---")

# ============================================================
# DOCUMENT OVERVIEW
# ============================================================

with st.expander("📑 Document information", expanded=False):
    st.write(
        "This application uses the uploaded PDF as the source of truth "
        "for question answering."
    )

    first_page_text = pages[0]["text"]

    st.markdown("**First-page preview**")
    st.write(first_page_text[:2500])

# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# QUESTION INPUT
# ============================================================

default_question = selected_question if "selected_question" in locals() else None

prompt_question = st.chat_input(
    "Ask a question about the PECA document..."
)

question = prompt_question or default_question

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant provisions..."):
                retrieved = retrieve_chunks(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    top_k=TOP_K,
                )

            # Keep retrieval visible for transparency.
            st.session_state.last_sources = retrieved

            best_score = retrieved[0]["score"] if retrieved else 0.0

            if not retrieved or best_score < 0.20:
                answer = (
                    "I couldn't find enough relevant information in the "
                    "uploaded document to answer that reliably."
                )
            else:
                with st.spinner("Generating a grounded answer..."):
                    answer = generate_answer(
                        question,
                        retrieved,
                        st.session_state.messages[:-1],
                    )

            st.markdown(answer)

    except Exception as exc:
        answer = (
            "I couldn't generate the answer because the AI service returned "
            f"an error: `{exc}`"
        )

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

if st.session_state.get("last_sources"):
    st.markdown("---")
    st.subheader("📚 Sources used for the latest answer")

    for source in st.session_state["last_sources"]:
        with st.expander(
            f"Page {source['page']} · {source['section']} · "
            f"relevance {source['score']:.3f}"
        ):
            st.markdown(
                f"""
                <div class="small-muted">
                Semantic similarity: {source['semantic_score']:.3f}<br>
                Lexical match: {source['lexical_score']:.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Retrieved text:**")
            st.write(source["text"])

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "PECA Legal Information Assistant · RAG MVP · "
    "Document-grounded answers only"
)
