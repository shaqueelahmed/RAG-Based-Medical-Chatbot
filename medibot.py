import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
DATA_PATH = os.getenv("DATA_PATH", "data/")


def ensure_vectorstore_exists():
    if os.path.exists(DB_FAISS_PATH):
        return

    data_dir = Path(DATA_PATH)
    pdf_files = list(data_dir.glob("*.pdf")) if data_dir.exists() else []

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found in the deployment environment. Place a legal source PDF in the data folder or set DATA_PATH to the folder containing the PDF before running the app."
        )

    from create_memory_for_llm import main as build_index

    build_index()


def build_custom_prompt():
    return PromptTemplate(
        template="""
You are a medical assistant. Use only the information in the provided context.
If the context contains an exact answer, give it directly and clearly.
If the context does not contain a direct answer but includes closely related information, summarize the closest relevant information and clearly label it as related context rather than a direct statement.
If the context is insufficient, say exactly: "I don't have enough information in the provided context to answer this accurately."
Do not invent or assume medical facts.
Do not provide information that is not supported by the context.

Context: {context}
Question: {input}

Answer directly and concisely.
""".strip(),
        input_variables=["context", "input"],
    )


@st.cache_resource
def get_vectorstore():
    ensure_vectorstore_exists()

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource
def get_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            groq_api_key = None

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to .env locally or to Streamlit secrets for deployment.")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
        api_key=groq_api_key,
    )


@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    llm = get_llm()
    custom_prompt = build_custom_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, custom_prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, combine_docs_chain)


def main():
    st.title("Cancer Encyclopedia Assistant")
    st.caption("A RAG-based AI chatbot for retrieving information from The Gale Encyclopedia of Cancer")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            rag_chain = get_rag_chain()
            response = rag_chain.invoke({"input": prompt})
            result = response.get("answer", "I could not generate an answer from the available context.")

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()