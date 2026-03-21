import os
import streamlit as st
import pickle
from groq import Groq

DB_FAISS_PATH = "vectorstore/db_faiss"

# LOAD DOCUMENTS FROM VECTORSTORE

@st.cache_resource
def load_documents():
    """Load documents from vectorstore"""
    try:
        # Try to load documents.pkl first
        docs_path = f"{DB_FAISS_PATH}/documents.pkl"
        
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'documents' in data:
                    return data['documents']
        
        # If no documents.pkl, try to load from index.pkl
        index_pkl_path = f"{DB_FAISS_PATH}/index.pkl"
        if os.path.exists(index_pkl_path):
            with open(index_pkl_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    for key in ['documents', 'chunks', 'docs']:
                        if key in data and data[key]:
                            return data[key]
        
        return []
                
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# CHECK VECTORSTORE

def check_vectorstore():
    """Check if vectorstore exists"""
    index_faiss_path = f"{DB_FAISS_PATH}/index.faiss"
    
    if not os.path.exists(index_faiss_path):
        st.warning("""
        ### 📚 Medical Database Not Found
        
        Please add PDF files to the `data` folder and push to GitHub.
        """)
        return False
    
    return True

# SIMPLE KEYWORD RETRIEVAL

def get_relevant_docs(documents, query, k=3):
    """Get relevant documents using keyword matching"""
    if not documents:
        return []
    
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to", "in", "on", 
                 "for", "with", "by", "at", "from", "as", "are", "was", "were"}
    
    query_words = [w.lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2]
    
    if not query_words:
        return documents[:k]
    
    scored_docs = []
    for doc in documents:
        if hasattr(doc, 'page_content'):
            doc_text = doc.page_content.lower()
        elif isinstance(doc, dict):
            doc_text = doc.get('page_content', '').lower()
        else:
            continue
        
        score = sum(1 for word in query_words if word in doc_text)
        if score > 0:
            scored_docs.append((doc, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:k]]

def extract_doc_text(doc):
    if hasattr(doc, 'page_content'):
        return doc.page_content
    elif isinstance(doc, dict):
        return doc.get('page_content', '')
    return str(doc)

# GROQ API CALL

def get_groq_response(prompt, context=None):
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found!")
        return None
    
    client = Groq(api_key=api_key)
    
    if context:
        full_prompt = f"""You are a medical assistant. Answer ONLY from the context below.

CONTEXT:
{context}

QUESTION:
{prompt}

ANSWER:"""
    else:
        full_prompt = f"""You are a medical assistant. Answer: {prompt}"""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# MAIN APP

def main():
    st.set_page_config(
        page_title="Health AI Medical Assistant", 
        page_icon="🏥",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .sidebar-content {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Health AI Medical Assistant")
    
    # Check API key
    if not st.secrets.get("GROQ_API_KEY", ""):
        st.error("⚠️ GROQ_API_KEY not configured! Add it in Secrets.")
        st.stop()
    
    # Check vectorstore
    vectorstore_exists = check_vectorstore()
    
    # Load documents if vectorstore exists
    documents = []
    if vectorstore_exists:
        with st.spinner("📚 Loading medical database..."):
            documents = load_documents()
    
    # Sidebar with clean content
    with st.sidebar:
        st.markdown("### 📚 Knowledge Base")
        st.markdown("Powered by trusted medical literature")
        
        st.markdown("---")
        
        st.markdown("### 💡 Features")
        st.markdown("""
        - Evidence-based medical information
        - Quick access to verified sources
        - 24/7 medical information support
        """)
        
        st.markdown("---")
        
        st.markdown("### 📖 About")
        st.markdown("""
        This assistant provides information from:
        - Medical textbooks
        - Clinical guidelines
        - Peer-reviewed medical literature
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚕️ Your Health Partner")
        st.markdown("""
        Get reliable medical information when you need it.
        
        **Note:** Always consult healthcare professionals for medical advice.
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔒 Privacy")
        st.markdown("Your conversations are private and secure.")
        
        st.markdown("---")
        
        st.markdown("### 📞 Emergency")
        st.markdown("**For medical emergencies, call emergency services immediately.**")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about medical information..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical literature..."):
                context = None
                if documents:
                    relevant = get_relevant_docs(documents, prompt, k=3)
                    if relevant:
                        context = "\n\n".join([extract_doc_text(d) for d in relevant])
                
                response = get_groq_response(prompt, context)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()