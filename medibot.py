import os
import streamlit as st
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document

DB_FAISS_PATH = "vectorstore/db_faiss"

# =========================
# EMBEDDINGS USING API
# =========================

class HFAPIEmbeddings:
    """Use HuggingFace Inference API for embeddings"""
    
    def __init__(self, api_token):
        self.api_token = api_token
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def embed_documents(self, texts):
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Get embedding for a single query"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Embedding API error: {response.status_code}")
                return [0.0] * 384  # Return zero vector as fallback
        except Exception as e:
            st.error(f"Error getting embeddings: {e}")
            return [0.0] * 384

# =========================
# LOAD DB WITH API EMBEDDINGS
# =========================

@st.cache_resource
def load_db():
    try:
        # Get HuggingFace token from secrets
        hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
        
        if not hf_token:
            st.warning("HuggingFace token not found. Using fallback mode.")
            return None
        
        # Create embeddings using API
        embeddings = HFAPIEmbeddings(hf_token)
        
        # Load FAISS database (this will use the API for any needed embeddings)
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
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in secrets!")
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
    
    # Check if API keys are set
    if not st.secrets.get("GROQ_API_KEY", ""):
        st.error("Please set GROQ_API_KEY in Streamlit secrets!")
        st.stop()
    
    if not st.secrets.get("HUGGINGFACEHUB_API_TOKEN", ""):
        st.warning("HuggingFace API token not set. Some features may not work.")
    
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
                        # Fallback: use LLM without context
                        llm = load_llm()
                        if llm:
                            response_obj = llm.invoke(f"""
You are a medical assistant. Answer the following question:
{prompt}
""")
                            response = response_obj.content
                        else:
                            response = "Database not available and LLM not configured."
                    else:
                        # Get retriever
                        retriever = db.as_retriever(search_kwargs={"k": 5})
                        
                        # Get relevant documents
                        docs = retriever.invoke(prompt)
                        docs = filter_docs(prompt, docs)
                        
                        if not docs:
                            response = "I couldn't find relevant information in the medical database."
                        else:
                            context = "\n\n".join([doc.page_content for doc in docs])
                            
                            llm = load_llm()
                            if llm is None:
                                response = "LLM not available."
                            else:
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
                    st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
