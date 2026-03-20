import os
import sys

# MUST be before any torch/sentence-transformers imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["NO_CUDA"] = "1"


import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

# =========================
# LOAD DB
# =========================

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# =========================
# LLM
# =========================

def load_llm():
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

# =========================
# FILTER
# =========================

def filter_docs(query, docs):
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to"}

    query_words = [
        w for w in query.lower().split()
        if w not in stopwords
    ]

    scored = []
    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query_words)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:3]]

# =========================
# APP
# =========================

def main():
    st.title("Health AI")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask something...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.chat.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):

            db = load_db()
            retriever = db.as_retriever(search_kwargs={"k": 5})

            docs = retriever.invoke(query)
            docs = filter_docs(query, docs)

            context = "\n\n".join([doc.page_content for doc in docs])

            llm = load_llm()

            response = llm.invoke(f"""
You are a medical assistant.

Answer ONLY from context.
- Use bullet points
- Be clear
- No guessing
- If incomplete: "The context provides limited information"

Context:
{context}

Question:
{query}

Answer:
""")

            answer = response.content
            
            output = answer

            st.chat_message("assistant").markdown(output)
            st.session_state.chat.append({"role": "assistant", "content": output})

if __name__ == "__main__":
    main()