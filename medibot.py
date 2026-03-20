import os
import sys
import streamlit as st

# === CRITICAL: Set environment variables BEFORE any imports ===
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Now import everything else
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

# =========================
# LOAD DB
# =========================

@st.cache_resource
def load_db():
    try:
        # Use a simpler embedding configuration
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        
        # Load FAISS database
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# =========================
# LLM
# =========================

def load_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        # Try to get from streamlit secrets
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        else:
            st.error("GROQ_API_KEY not found!")
            return None
    
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

# =========================
# FILTER
# =========================

def filter_docs(query, docs):
    if not docs:
        return []
    
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to"}
    
    query_words = [
        w for w in query.lower().split()
        if w not in stopwords
    ]
    
    scored = []
    for doc in docs:
        text = doc.page_content.lower()
        score = sum(word in text for word in query_words)
        scored.append((doc, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:3]]

# =========================
# APP
# =========================

def main():
    st.title("🏥 Health AI Medical Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about medical information..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical database..."):
                try:
                    # Load database
                    db = load_db()
                    if db is None:
                        st.error("Database not available. Please check your vectorstore.")
                        return
                    
                    # Retrieve relevant documents
                    retriever = db.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(prompt)
                    docs = filter_docs(prompt, docs)
                    
                    if not docs:
                        response = "I couldn't find relevant information in the medical database."
                    else:
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        llm = load_llm()
                        if llm is None:
                            return
                        
                        response_obj = llm.invoke(f"""
You are a medical assistant. Answer ONLY from the context provided.
- Be specific and use bullet points
- Do not guess or add information not in context
- If information is incomplete, say: "The context provides limited information"

Context:
{context}

Question:
{prompt}

Answer:
""")
                        response = response_obj.content
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please make sure the vectorstore is properly created.")

if __name__ == "__main__":
    main()
