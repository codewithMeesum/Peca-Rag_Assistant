🇵🇰 PECA Legal Assistant

A multilingual, document-grounded Retrieval-Augmented Generation (RAG) assistant for Pakistan's Prevention of Electronic Crimes Act, 2016.






🚀 Live Demo

Open PECA Legal Assistant

The default PECA document is loaded automatically, so visitors do not need to download or upload the PDF before using the app.

🧠 About the Project

PECA Legal Assistant is a focused RAG application built around Pakistan's Prevention of Electronic Crimes Act, 2016 (PECA).

Instead of relying only on an LLM's general knowledge, the application:

Extracts text from the selected PDF.

Splits the document into searchable chunks.

Creates embeddings for those chunks.

Retrieves relevant passages for the user's question.

Sends the retrieved evidence to the LLM.

Generates a grounded answer with source information.

Core idea:

Retrieve → Ground → Explain → Verify

✨ Features

ChatGPT-inspired minimal chat interface

Automatic PECA document loading

Optional custom PDF upload

English, Urdu, and Roman Urdu answers

Simple and Detailed answer styles

Multi-turn conversation

Suggested questions

Page-aware PDF extraction

Section-aware metadata

Overlapping document chunks

Sentence-Transformer embeddings

Hybrid semantic + lexical retrieval

Section-number matching

Page-diverse retrieval

Grounded Groq generation

Structured answer guidance

[Source 1], [Source 2] style citations

Expandable retrieved evidence

Legal-information disclaimer

🌍 Supported Languages

The assistant can explain retrieved information in:

English

Urdu

Roman Urdu

The original document remains the source evidence. The selected language controls the explanation.

📚 Structured Answers

The assistant adapts the response structure to the type of question.

Example: practical question

Summary

Practical Steps
1. ...
2. ...
3. ...

Important Note

Sources
[Source 1]
[Source 2]

Example: offence or punishment question

Summary

Relevant Provision
...

Consequence / Punishment
...

Sources
[Source 1]

The application is designed to avoid inventing legal sections, penalties, or citations that are not supported by the retrieved document.

⚡ RAG Pipeline

PECA PDF / Uploaded PDF
          |
          v
   Text Extraction
          |
          v
 Page + Section Aware
       Chunking
          |
          v
      Embeddings
          |
          v
   Hybrid Retrieval
    /      |       \
   /       |        \
Semantic  Keyword  Section
Search     Match   Matching
    \       |       /
     \      |      /
      +-----+-----+
            |
            v
 Relevant Source Passages
            |
            v
     Grounded Prompt
            |
            v
        Groq LLM
            |
            v
 Answer in Selected Language
            |
            v
      Source Evidence

🔎 Why RAG?

A normal LLM can generate an answer from patterns learned during training.

A RAG system adds a retrieval step:

User Question
      |
      v
Search the document
      |
      v
Retrieve relevant evidence
      |
      v
Give evidence to the LLM
      |
      v
Generate grounded answer

This is useful when the selected document should control the answer.

📄 Default PECA Source

The default source document is the project's PECA PDF hosted on GitHub.

Prevention of Electronic Crimes Act, 2016

Open / Download PECA PDF

The application downloads the source document automatically when the default PECA mode is used.

📤 Custom PDF Mode

The application can also work with another text-based PDF.

Choose Upload PDF, then the app:

New PDF
   |
   v
Extract text
   |
   v
Create chunks
   |
   v
Create embeddings
   |
   v
Search uploaded document
   |
   v
Generate answer from uploaded document

When the source document changes, the conversation is reset so that context from the previous document is not carried into the new one.

Best results

Use a text-based PDF with selectable text.

Scanned or image-only PDFs may require OCR support.

🔎 Source Evidence

Retrieved evidence can be expanded to inspect the material used for the answer.

Each source can include:

Source number

Page

Detected section

Retrieval score

Original retrieved passage

This makes the response easier to verify against the source document.

🛡️ Grounding and Accuracy

The assistant is instructed to:

Use retrieved document passages as the source of truth.

Avoid unsupported legal claims.

Avoid fabricated citations.

State when enough relevant information cannot be found.

Preserve the meaning of the source while explaining it clearly.

This improves reliability, but no AI system is infallible.

For legal decisions, verify the cited source document and consult a qualified legal professional where appropriate.

🛠️ Tech Stack

Technology

Purpose

Python

Application logic

Streamlit

Web application and UI

PyPDF

PDF text extraction

Sentence Transformers

Text embeddings

NumPy

Similarity retrieval

Groq API

LLM inference

GPT-OSS 120B

Answer generation

GitHub

Source control

Streamlit Community Cloud

Deployment

📁 Project Structure

peca-rag-assistant/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- .streamlit/
    `-- config.toml

▶️ Run Locally

1. Clone the repository

Replace YOUR_USERNAME with your GitHub username if your repository name is different.

git clone https://github.com/YOUR_USERNAME/peca-rag-assistant.git
cd peca-rag-assistant

2. Install dependencies

pip install -r requirements.txt

3. Add your Groq API key

Create this file:

.streamlit/secrets.toml

Add:

GROQ_API_KEY = "your_groq_api_key"

4. Start the application

streamlit run app.py

☁️ Deploy on Streamlit Community Cloud

Push the project to GitHub.

Open Streamlit Community Cloud.

Create a new app from your repository.

Set app.py as the main file.

Open Settings → Secrets.

Add:

GROQ_API_KEY = "your_groq_api_key"

Deploy the application.

🔐 Security

Never commit your real Groq API key to GitHub.

Do not place the API key directly inside app.py.

🧪 Testing the RAG System

Test both supported and unsupported questions.

Supported examples

What is cyber harassment?
Which section discusses this offence?
What punishment does the document mention?
What does the Act say about complaints?

Unsupported example

Ask about a topic that is not covered by the selected document.

The expected behavior is:

I couldn't find enough relevant information in the selected document to answer that reliably.

This is an important RAG behavior because the system should not invent an answer when relevant evidence is unavailable.

🎯 Project Goal

The project demonstrates how RAG can make a difficult official document easier to search and understand.

Instead of:

Read a long legal document → manually find the relevant section

The user can:

Ask → Retrieve → Understand → Verify

🧩 RAG Concepts Demonstrated

Chunking

Large documents are divided into smaller searchable passages.

Embeddings

Text is represented as numerical vectors for semantic comparison.

Retrieval

Relevant passages are selected from the document for the user's question.

Grounded Generation

The LLM receives retrieved evidence before generating the response.

Source Transparency

Retrieved passages remain inspectable so users can verify the evidence.

⚖️ Legal Disclaimer

This project is an informational and educational document assistant.

It is not a lawyer, does not provide professional legal advice, and should not replace advice from a qualified legal professional.

🔮 Future Improvements

Stronger reranking

Exact subsection citations

Direct source-page navigation

OCR for scanned PDFs

Persistent vector storage

Multiple official legal documents

Automated RAG evaluation

Improved Urdu and Roman Urdu retrieval

👨‍💻 Author

Mesum Mukhtar

Building practical AI systems with Generative AI, RAG, Machine Learning, and emerging technologies.

Project Links

🚀 Live App:
https://peca-rag-assistant.streamlit.app/

📄 PECA Source PDF:
https://github.com/codewithMeesum/Peca-Rag_Assistant/blob/main/PECA%202026.pdf

⭐ Support

If you find the project useful, consider giving the repository a ⭐ Star.

Built with Python · RAG · Groq · Streamlit
