import os
import streamlit as st
import pickle
import faiss
import numpy as np
from groq import Groq

DB_FAISS_PATH = "vectorstore/db_faiss"

# =========================
# SIMPLE DOCUMENT CLASS
# =========================

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# =========================
# LOAD FAISS INDEX AND DOCUMENTS
# =========================

@st.cache_resource
def load_faiss_index():
    """Load FAISS index and documents without any ML libraries"""
    try:
        # Load FAISS index
        index_path = f"{DB_FAISS_PATH}/index.faiss"
        index = faiss.read_index(index_path)
        
        # Load documents
        documents = []
        
        # Try to load from documents.pkl
        docs_path = f"{DB_FAISS_PATH}/documents.pkl"
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                documents = pickle.load(f)
        else:
            # Try to load from index.pkl
            index_pkl_path = f"{DB_FAISS_PATH}/index.pkl"
            if os.path.exists(index_pkl_path):
                with open(index_pkl_path, "rb") as f:
                    data = pickle.load(f)
                    documents = data.get("documents", [])
        
        if not documents:
            st.warning("No documents found. Using keyword-based search without document retrieval.")
        
        return {"index": index, "documents": documents}
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

# =========================
# SIMPLE KEYWORD RETRIEVAL
# =========================

def get_relevant_docs(faiss_data, query, k=3):
    """Get relevant documents using simple keyword matching"""
    if not faiss_data or not faiss_data["documents"]:
        return []
    
    # Stopwords to filter
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to", "in", "on", 
                 "for", "with", "by", "at", "from", "as", "are", "was", "were", "be"}
    
    # Extract keywords from query
    query_words = [w.lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2]
    
    if not query_words:
        return []
    
    # Score documents based on keyword matches
    scored_docs = []
    for doc in faiss_data["documents"]:
        doc_text = doc.page_content.lower()
        score = sum(1 for word in query_words if word in doc_text)
        if score > 0:
            scored_docs.append((doc, score))
    
    # Sort by score and return top k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:k]]

# =========================
# GROQ API CALL
# =========================

def get_groq_response(prompt, context=None):
    """Call Groq API directly"""
    
    # Get API key from secrets
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets!")
        return None
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Build the prompt
    if context:
        full_prompt = f"""You are a medical assistant. Answer ONLY from the context provided below.

IMPORTANT RULES:
- Answer based ONLY on the given context
- Use bullet points for clarity
- Be specific and concise
- Do NOT guess or add information outside the context
- If the answer is not in the context, say: "The context provides limited information about this topic."

CONTEXT:
{context}

QUESTION:
{prompt}

ANSWER:
"""
    else:
        full_prompt = f"""You are a medical assistant. Answer the following question:
{prompt}"""
    
    try:
        # Make API call
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# =========================
# MAIN APP
# =========================

def main():
    st.set_page_config(
        page_title="Health AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Health AI Medical Assistant")
    st.caption("Your AI-powered medical information assistant")
    
    # Check if API key is set
    if not st.secrets.get("GROQ_API_KEY", ""):
        st.error("""
        ### ⚠️ GROQ_API_KEY not configured!
        
        Please add your Groq API key to continue:
        1. Click on **"Manage app"** in the bottom right
        2. Go to **"Secrets"**
        3. Add: `GROQ_API_KEY = "your_api_key_here"`
        
        [Get a Groq API key](https://console.groq.com/keys)
        """)
        st.stop()
    
    # Load FAISS data
    with st.spinner("📚 Loading medical database..."):
        faiss_data = load_faiss_index()
    
    if faiss_data is None:
        st.warning("⚠️ Medical database not loaded. The app will work without document retrieval.")
    
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
            with st.spinner("🤔 Searching medical database and formulating answer..."):
                # Get relevant documents if available
                context = None
                relevant_docs = []
                
                if faiss_data and faiss_data["documents"]:
                    relevant_docs = get_relevant_docs(faiss_data, prompt, k=3)
                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Show sources in expander
                        with st.expander("📚 Retrieved from medical documents"):
                            for i, doc in enumerate(relevant_docs, 1):
                                source = doc.metadata.get('source', 'Unknown')
                                st.markdown(f"**Source {i}:** `{source}`")
                                st.markdown(f"{doc.page_content[:300]}...")
                                st.divider()
                
                # Get response from Groq
                response = get_groq_response(prompt, context)
                
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_msg = "Sorry, I couldn't process your request. Please check your API key and try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant provides information based on:
        - Medical documents in the database
        - Keyword-based document retrieval
        - Groq's Llama 3.1 model for answers
        """)
        
        if faiss_data and faiss_data["documents"]:
            st.markdown(f"### 📊 Database Stats")
            st.markdown(f"- **Total documents:** {len(faiss_data['documents'])}")
        
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This is an AI assistant for informational purposes only.
        Always consult qualified healthcare professionals for medical advice.
        """)

if __name__ == "__main__":
    main()
