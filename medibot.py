import os
import streamlit as st
import pickle
from groq import Groq

DB_FAISS_PATH = "vectorstore/db_faiss"

# =========================
# CHECK VECTORSTORE
# =========================

def check_vectorstore():
    """Check if vectorstore exists and show helpful message"""
    
    index_faiss_path = f"{DB_FAISS_PATH}/index.faiss"
    index_pkl_path = f"{DB_FAISS_PATH}/index.pkl"
    
    if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
        st.error("""
        ### ❌ Medical Database Not Found
        
        The vectorstore files are missing. To fix this:
        
        **Option 1: Add PDFs to the data folder**
        1. Create a `data` folder in your repository
        2. Add your medical PDF files to the `data` folder
        3. Push to GitHub and redeploy
        
        **Option 2: Create vectorstore locally and push**
        1. Run `python creat_memory_for_llm.py` locally
        2. Run `python extract_real_documents.py` to create documents.pkl
        3. Push the `vectorstore` folder to GitHub
        
        **Option 3: Use the app without documents**
        The app will work but will answer from general knowledge only.
        """)
        return False
    
    return True

# =========================
# LOAD DOCUMENTS
# =========================

@st.cache_resource
def load_documents():
    """Load documents from vectorstore"""
    try:
        # Check if documents.pkl exists
        docs_path = f"{DB_FAISS_PATH}/documents.pkl"
        
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    st.success(f"✅ Loaded {len(data)} medical documents")
                    return data
        
        # If no documents.pkl, check if index exists
        if os.path.exists(f"{DB_FAISS_PATH}/index.faiss"):
            st.info("📚 Vectorstore found but no documents.pkl. The app will work with basic functionality.")
        
        return []
        
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# =========================
# GROQ API CALL
# =========================

def get_groq_response(prompt, context=None):
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets!")
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
        st.error(f"Error: {e}")
        return None

# =========================
# SIMPLE RETRIEVAL
# =========================

def get_relevant_docs(documents, query, k=3):
    """Simple keyword-based retrieval"""
    if not documents:
        return []
    
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to", "in", "on"}
    query_words = [w.lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2]
    
    if not query_words:
        return documents[:k]
    
    scored = []
    for doc in documents:
        text = doc.page_content.lower() if hasattr(doc, 'page_content') else str(doc).lower()
        score = sum(1 for word in query_words if word in text)
        if score > 0:
            scored.append((doc, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]

def extract_text(doc):
    if hasattr(doc, 'page_content'):
        return doc.page_content
    return str(doc)

# =========================
# MAIN APP
# =========================

def main():
    st.set_page_config(page_title="Health AI Medical Assistant", page_icon="🏥")
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("Medical AI Assistant using Groq's Llama 3.1")
        
        if documents:
            st.markdown(f"**Documents Loaded:** {len(documents)}")
            st.success("✅ Medical database active")
        elif vectorstore_exists:
            st.info("📚 Vectorstore found, processing...")
        else:
            st.warning("⚠️ No medical database loaded")
            st.markdown("**The app will answer from general knowledge only.**")
        
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("For informational purposes only. Consult healthcare professionals.")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about medical information..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = None
                if documents:
                    relevant = get_relevant_docs(documents, prompt, k=3)
                    if relevant:
                        context = "\n\n".join([extract_text(d) for d in relevant])
                        
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(relevant, 1):
                                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                                st.markdown(f"**{i}.** {source}")
                                st.caption(extract_text(doc)[:150] + "...")
                                st.divider()
                
                response = get_groq_response(prompt, context)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
