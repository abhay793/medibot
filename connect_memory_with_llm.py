import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_FAISS_PATH = "vectorstore/db_faiss"

# LLM

def load_llm():
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

# VECTOR STORE

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 5})

# FILTER

def filter_docs(query, docs):
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to"}

    query_words = [
        word for word in query.lower().split()
        if word not in stopwords
    ]

    scored = []
    for doc in docs:
        text = doc.page_content.lower()
        score = sum(word in text for word in query_words)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:3]]

# LOOP

while True:
    query = input("\nAsk (or exit): ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)
    docs = filter_docs(query, docs)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = load_llm()

    response = llm.invoke(f"""
You are a medical assistant.

Answer ONLY from the context.
- Be specific
- Use bullet points
- If incomplete say: "The context provides limited information"

Context:
{context}

Question:
{query}

Answer:
""")

    print("\nANSWER:\n", response.content)

    print("\nSOURCES:")
    for doc in docs:
        print(f"- {doc.metadata.get('source')} -> {doc.page_content[:120]}...")