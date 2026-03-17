# 🤖 MediBot: RAG-Based Conversational AI

An intelligent, context-aware conversational agent built using **Retrieval-Augmented Generation (RAG)**. This application provides precise, domain-specific answers by retrieving information from a local vector database and generating responses using the LLaMA 3.1 Large Language Model via the Groq API.

## 🚀 Overview
Unlike standard LLMs that rely purely on pre-trained knowledge, MediBot actively searches a customized knowledge base (stored as vector embeddings) to ground its answers in specific context. It is designed to strictly adhere to the provided data, reducing hallucinations and making it highly reliable for specialized domains like medical queries or technical support.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit (Interactive Chat UI)
* **LLM Orchestration:** LangChain
* **Large Language Model:** LLaMA-3.1-8b-instant (via Groq API for ultra-fast inference)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)

## ⚙️ Key Features
* **Custom Knowledge Retrieval:** Uses FAISS to perform similarity searches, retrieving the top 3 most relevant document chunks ($k=3$) for any given user query.
* **Hallucination Prevention:** The custom prompt strictly instructs the LLM to only use the retrieved context and to transparently state if it does not know the answer.
* **Session Memory:** Built with Streamlit's session state to maintain the chat history and provide a seamless back-and-forth conversational experience.
* **High-Speed Inference:** Integrates Groq's LPU (Language Processing Unit) architecture for near-instant text generation.
