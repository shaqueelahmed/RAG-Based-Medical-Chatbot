# 🤖 MediBot: RAG-Based Conversational AI

An intelligent, context-aware conversational agent built using **Retrieval-Augmented Generation (RAG)**. This application provides precise, domain-specific answers by retrieving information from a local vector database and generating responses using the LLaMA 3.1 Large Language Model via the Groq API.

## 🚀 Overview
Unlike standard LLMs that rely purely on pre-trained knowledge, MediBot actively searches a customized knowledge base (stored as vector embeddings) to ground its answers in specific context. It is designed to strictly adhere to the provided data, reducing hallucinations and making it highly reliable for specialized domains like medical queries or technical support.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit (Interactive Chat UI)
* **LLM Orchestration:** LangChain
* **Large Language Model:** openai/gpt-oss-20b (via Groq API for ultra-fast inference)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)

## ⚙️ Key Features
* **Custom Knowledge Retrieval:** Uses FAISS to perform similarity searches, retrieving the top 3 most relevant document chunks ($k=3$) for any given user query.
* **Hallucination Prevention:** The custom prompt strictly instructs the LLM to only use the retrieved context and to transparently state if it does not know the answer.
* **Session Memory:** Built with Streamlit's session state to maintain the chat history and provide a seamless back-and-forth conversational experience.
* **High-Speed Inference:** Integrates Groq's LPU (Language Processing Unit) architecture for near-instant text generation.

## 🚀 Deployment notes
This repository intentionally does not commit the source PDF or generated FAISS index.

Before deploying to Streamlit Cloud or another hosted environment:

1. Add your Groq API key to Streamlit secrets or a local `.env` file.
2. Add the legal source PDF to the `data/` folder if your licensing allows it.
3. Run:
   ```bash
   python create_memory_for_llm.py
   ```
4. Launch:
   ```bash
   streamlit run medibot.py
   ```

For local development, copy `.env.example` to `.env` and set your key:

```bash
copy .env.example .env
```

For Streamlit Cloud, add this secret:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## ⚠️ Domain note
This application currently answers questions based on the source material in the project. The current dataset is cancer-focused, so its answers are only as broad as the knowledge base loaded into the FAISS index.
