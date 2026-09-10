# 🇵🇰 PECA Legal Assistant

> **A multilingual, document-grounded RAG assistant for the Prevention of Electronic Crimes Act, 2016.**

<p align="center">

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-0d7a46?style=for-the-badge)](https://peca-rag-assistant.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-111827?style=for-the-badge)](https://groq.com/)

</p>

---

## 🚀 Try It Live

### 👉 [Open PECA Legal Assistant](https://peca-rag-assistant.streamlit.app/)

**No PDF upload is required for the default experience.**

The app automatically loads the project's PECA source document, indexes it, and lets visitors ask questions immediately.

---

## 🧠 What Is This?

**PECA Legal Assistant** is a focused **Retrieval-Augmented Generation (RAG)** application built around Pakistan's **Prevention of Electronic Crimes Act, 2016**.

Instead of asking an LLM to answer purely from its general knowledge, the system first retrieves relevant passages from the selected document and then generates an answer grounded in those passages.

### Core idea

> **Retrieve → Ground → Explain → Verify**

---

## ✨ What You Can Do

### 💬 Ask Questions

Ask natural-language questions such as:

```text
What does the Act say about cyber harassment?

What punishment is mentioned for cyber-stalking?

Which section discusses cyber-bullying?

What does the Act say about complaints?

What powers does the Authority have?
```

### 🌍 Choose Your Language

The assistant can explain answers in:

- 🇬🇧 **English**
- 🇵🇰 **Urdu**
- 🇵🇰 **Roman Urdu**

The original document remains the source evidence.

### 📝 Choose Answer Style

- **Simple** — easy-to-understand explanation
- **Detailed** — more complete explanation

### 📚 Inspect the Evidence

Each answer can expose:

- Source number
- Page
- Detected section
- Retrieval score
- Original retrieved passage

This lets you inspect the evidence behind the generated response.

---

## ⚡ How the RAG Pipeline Works

```text
                 📄 PECA PDF
                      │
                      ▼
              Text Extraction
                      │
                      ▼
            Page-Aware Chunking
                      │
                      ▼
                Embeddings
                      │
                      ▼
             Hybrid Retrieval
              ┌───────┴───────┐
              │               │
       Semantic Search   Keyword Match
              │               │
              └───────┬───────┘
                      ▼
             Relevant Passages
                      │
                      ▼
               Grounded Prompt
                      │
                      ▼
                  Groq LLM
                      │
                      ▼
             Answer + Sources
```

---

## 🔍 Why RAG?

A standard LLM can generate an answer from patterns learned during training.

A RAG system adds a retrieval layer:

```text
User Question
      ↓
Search the document
      ↓
Retrieve relevant evidence
      ↓
Give evidence to the LLM
      ↓
Generate grounded answer
```

This is especially useful for document-specific questions where the **source document should control the answer**.

---

## 🛡️ Grounding & Accuracy

The assistant is instructed to:

- Use the retrieved document passages as the source of truth
- Avoid inventing sections or penalties
- Avoid fabricated citations
- State when the document does not contain enough relevant information
- Preserve the meaning of the source while explaining it simply

### Important

This improves reliability, but **no AI system should be treated as infallible**.

For legal decisions, always inspect the cited source document and consult a qualified legal professional where appropriate.

---

## 📄 Built-In PECA Source

The default experience uses the project's PECA PDF hosted on GitHub.

### Source document

**Prevention of Electronic Crimes Act, 2016**

[📥 Open / Download PECA PDF](https://github.com/codewithMeesum/Peca-Rag_Assistant/blob/main/PECA%202026.pdf)

Visitors **do not need to download it manually** to use the default app.

---

## 📤 Upload Another PDF

The application also supports a custom document workflow.

Choose:

> **Upload PDF**

Then the app rebuilds the retrieval index for the uploaded document.

```text
New PDF
   ↓
Extract text
   ↓
Create chunks
   ↓
Create embeddings
   ↓
Retrieve relevant passages
   ↓
Generate answer
```

The conversation resets when the source document changes, preventing the previous document's chat context from being carried into the new document.

### Best results

Use a **text-based PDF** with selectable text.

Scanned/image-only PDFs may require OCR.

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| 🐍 **Python** | Application logic |
| 🎈 **Streamlit** | Web application + UI |
| 📄 **PyPDF** | PDF text extraction |
| 🧠 **Sentence Transformers** | Text embeddings |
| 🔢 **NumPy** | Vector similarity retrieval |
| ⚡ **Groq API** | LLM inference |
| 🤖 **GPT-OSS 120B** | Answer generation |
| ☁️ **Streamlit Community Cloud** | Deployment |
| 🐙 **GitHub** | Source control |

---

## 📁 Project Structure

```text
peca-rag-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

---

## ▶️ Run Locally

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/peca-rag-assistant.git
cd peca-rag-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### 4. Start the app

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit app.
3. Select `app.py`.
4. Add your secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

5. Deploy.

### 🔐 Never commit your real API key

Do **not** put your actual Groq key inside `app.py` or commit it to GitHub.

---

## 🧪 Test the Application

A useful RAG evaluation set should contain both supported and unsupported questions.

### ✅ Supported

```text
What is cyber harassment?

Which section discusses this offence?

What punishment does the document mention?

What does the Act say about complaints?
```

### ⚠️ Unsupported / out-of-document

Ask about a topic that is not present in the selected PDF.

Expected behavior:

> “I couldn't find enough relevant information in the selected document to answer that reliably.”

That test is important because a good RAG system should **refuse to invent an answer** when evidence is missing.

---

## 🎯 Project Goal

The goal is to demonstrate how a practical RAG system can make a difficult official document easier to search and understand.

### Instead of:

> **Read a long legal document → manually find the relevant section**

### The user can:

> **Ask → Retrieve → Understand → Verify**

---

## 🧩 Key RAG Concepts Demonstrated

### Chunking
Large documents are split into smaller searchable passages.

### Embeddings
Text is represented as numerical vectors for semantic comparison.

### Retrieval
Relevant document passages are selected for the user's question.

### Grounded Generation
The LLM receives retrieved evidence before generating the answer.

### Source Transparency
Retrieved passages remain visible so users can inspect the evidence.

---

## ⚖️ Legal Disclaimer

This project is an **informational and educational document assistant**.

It is **not a lawyer**, does not provide professional legal advice, and should not replace advice from a qualified legal professional.

---

## 🔮 Future Improvements

Possible next steps:

- 🔎 Stronger reranking
- 📌 Exact subsection citations
- 🔗 Direct source-page navigation
- 🧾 OCR for scanned documents
- 💾 Persistent vector storage
- 📚 Multiple official legal documents
- 📊 Automated RAG evaluation
- 🌐 Improved Urdu/Roman Urdu retrieval

---

## 👨‍💻 Author

### Meesum Mukhtar

Building practical AI systems with **Generative AI, RAG, Machine Learning, and emerging technologies.**

### 🔗 Project Links

**🚀 Live App**  
https://peca-rag-assistant.streamlit.app/

**📄 PECA Source PDF**  
https://github.com/codewithMeesum/Peca-Rag_Assistant/blob/main/PECA%202026.pdf

---

## ⭐ Support the Project

If you found this project useful, consider giving the repository a ⭐ **Star**.

<p align="center">
  <b>Built with 🐍 Python · 🧠 RAG · ⚡ Groq · 🎈 Streamlit</b>
</p>
