import os
import sys

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


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


def build_rag_chain():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to the .env file or environment variables.")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=512,
        api_key=groq_api_key,
    )

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )

    prompt = build_custom_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)


def ask_question(question=None):
    if question is None:
        question = input("Write Query Here: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    rag_chain = build_rag_chain()
    response = rag_chain.invoke({"input": question})
    print("RESULT: ", response["answer"])
    print("\nSOURCE DOCUMENTS: ")
    for doc in response.get("context", []):
        print(f"- {doc.metadata} -> {doc.page_content[:200]}...")


def main():
    if len(sys.argv) > 1:
        ask_question(" ".join(sys.argv[1:]))
        return
    ask_question()


if __name__ == "__main__":
    main()
