# 🇵🇰 PECA Legal Information Assistant

> **A document-grounded RAG MVP for the Prevention of Electronic Crimes Act, 2016**

🌐 **Live / Deployment Target:** Streamlit Community Cloud

---

## 🎯 Project Overview

The **PECA Legal Information Assistant** is a Retrieval-Augmented Generation (RAG) application designed to help users ask questions about an uploaded copy of Pakistan's **Prevention of Electronic Crimes Act, 2016**.

The goal is not to create a general-purpose chatbot.

Instead, the application follows a simple principle:

> **Retrieve relevant information from the official document first, then generate an answer from that retrieved evidence.**

This reduces unsupported answers and makes the underlying document context visible to the user.

---

## ✨ MVP Features

- 📄 Upload the official PECA PDF
- 📑 Preserve page-level document information
- ✂️ Split the document into searchable chunks
- 🧠 Generate semantic embeddings
- 🔎 Hybrid retrieval using semantic + lexical matching
- 🤖 Generate grounded answers using the Groq API
- 📚 Display retrieved source passages
- 📌 Show page and detected section information
- 💬 Conversational follow-up questions
- 🎯 Suggested questions for new users
- 🔐 API key stored through Streamlit Secrets
- ⚠️ Legal-information disclaimer
- 🚫 Grounded fallback when the answer is not sufficiently supported

---

## 🧠 RAG Architecture

```text
                 📄 PECA PDF
                      │
                      ▼
             ┌─────────────────┐
             │  Text Extraction│
             │     PyPDF       │
             └─────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Chunking     │
             │ Page-aware text │
             └─────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   Embeddings    │
             │ Sentence        │
             │ Transformers    │
             └─────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Retrieval    │
             │ Semantic +      │
             │ lexical match   │
             └─────────────────┘
                      │
              Top relevant text
                      │
                      ▼
             ┌─────────────────┐
             │    Groq LLM     │
             │ GPT-OSS 120B    │
             └─────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Grounded Answer │
             │ + Source Text   │
             └─────────────────┘
```

---

## 🔄 Request Flow

When a user asks:

> **"What does the document say about cyber harassment?"**

the application performs:

```text
Question
   ↓
Question Embedding
   ↓
Similarity Search
   ↓
Relevant PECA Chunks
   ↓
Grounded Prompt
   ↓
Groq LLM
   ↓
Answer
   ↓
Page / Section Sources
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Application logic |
| **Streamlit** | Interactive web UI |
| **PyPDF** | PDF text extraction |
| **Sentence Transformers** | Semantic embeddings |
| **NumPy** | Vector similarity |
| **Groq API** | LLM inference |
| **GPT-OSS 120B** | Answer generation |
| **GitHub** | Source control |
| **Streamlit Community Cloud** | Deployment |

---

## 📁 Repository Structure

```text
peca-rag-mvp/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/peca-rag-mvp.git
cd peca-rag-mvp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

For local development, create:

```text
.streamlit/
└── secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### 4. Run

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

Push these files to GitHub:

```text
app.py
requirements.txt
README.md
```

Then create the Streamlit app using `app.py` as the entrypoint.

In the app's **Secrets** settings, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do **not** commit the actual API key to GitHub.

---

## 🔐 Security

The application reads the Groq API key from Streamlit Secrets.

### ✅ Correct

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### ❌ Never do this

```python
api_key = "gsk_your_real_key"
```

Never publish a real API key inside source code or a public repository.

---

## 📚 Why This Is RAG

A normal LLM may answer a question from its learned knowledge.

This application adds a retrieval layer:

```text
User Question
     ↓
Search the PECA document
     ↓
Retrieve relevant passages
     ↓
Give those passages to the LLM
     ↓
Generate answer from that context
```

This makes the application **document-grounded** rather than simply relying on general model knowledge.

---

## 🔎 Retrieved Context

After answering a question, the app shows:

- Page number
- Detected section
- Semantic similarity
- Lexical match
- Actual retrieved PDF text

This makes the retrieval process transparent and easier to verify.

---

## ⚖️ Legal Safety

This project is an **information-retrieval and document-understanding tool**.

It should not be presented as a lawyer, legal representative, or authoritative source of legal advice.

The system is instructed to:

1. Use only retrieved document context.
2. Avoid inventing legal provisions.
3. State when information cannot be found.
4. Show supporting source passages.
5. Explain legal text without claiming to make legal decisions.

---

## ⚠️ Current Limitations

### 1. Text-based PDFs

The current MVP expects a PDF from which text can be extracted.

Scanned/image-only PDFs may require OCR before they can be searched.

### 2. Single-document workflow

The MVP is intentionally focused on one uploaded PECA document.

### 3. Simple chunking

The current implementation uses page-aware overlapping text chunks rather than a full legal-section parser.

### 4. Retrieval confidence

Semantic retrieval is not proof of legal relevance. Users should inspect the retrieved source text.

---

## 🔮 Future Improvements

Possible next versions could add:

- 🇵🇰 Urdu language support
- 📚 Multiple Pakistani cyber-law documents
- 🔎 Section-aware legal parsing
- 🧠 Reranking models
- 📌 Exact section/subsection citations
- 📄 Direct page navigation
- 🗂️ Persistent vector storage
- 🎙️ Voice questions
- 📷 OCR for scanned documents
- 💬 Better multi-turn conversational memory
- 🔗 Source-document download links
- 📊 Retrieval evaluation dashboard

---

## 🧪 Suggested Test Questions

Use these to evaluate the MVP:

```text
What is cyber harassment?

Which section discusses this offence?

What punishment does the document mention?

What does the Act say about complaints?

What powers does the Authority have?

Who is responsible for carrying out this function?

Does the document mention [a term that is not actually present]?
```

The final test is important because the system should respond that it **couldn't find the information** rather than confidently inventing an answer.

---

## 🏆 MVP Pitch

### Problem

Official legal documents can be difficult for ordinary users to navigate and understand.

### Solution

A conversational RAG assistant that searches the official PECA document and explains relevant provisions in simple language while showing the retrieved source text.

### Core Value

> **Ask → Retrieve → Explain → Verify**

---

## 👨‍💻 Author

**Meesum Mukhtar**

Built as an educational and practical RAG MVP for document-grounded AI.

---

## ⭐ Support

If this project is useful, consider giving the repository a ⭐ **Star**.

<p align="center">
  Built with 🐍 Python · 🎈 Streamlit · 🧠 RAG · ⚡ Groq
</p>
