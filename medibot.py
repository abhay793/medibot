import os
import streamlit as st
import pickle
from groq import Groq

DB_FAISS_PATH = "vectorstore/db_faiss"

# =========================
# LOAD DOCUMENTS FROM VECTORSTORE
# =========================

@st.cache_resource
def load_documents():
    """Load documents from vectorstore"""
    try:
        # First try to load documents.pkl
        docs_path = f"{DB_FAISS_PATH}/documents.pkl"
        
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    st.success(f"✅ Loaded {len(data)} documents from documents.pkl")
                    return data
                elif isinstance(data, dict) and 'documents' in data:
                    st.success(f"✅ Loaded {len(data['documents'])} documents from documents.pkl")
                    return data['documents']
        
        # If no documents.pkl, try to load from index.pkl
        index_pkl_path = f"{DB_FAISS_PATH}/index.pkl"
        if os.path.exists(index_pkl_path):
            with open(index_pkl_path, "rb") as f:
                data = pickle.load(f)
                
                # Check if it contains documents
                if isinstance(data, dict):
                    # Try different keys
                    for key in ['documents', 'chunks', 'docs', 'texts']:
                        if key in data and data[key]:
                            st.success(f"✅ Loaded {len(data[key])} documents from index.pkl ({key})")
                            return data[key]
                    
                    # Check if it's a FAISS docstore
                    if 'docstore' in data and hasattr(data['docstore'], '_dict'):
                        docs = list(data['docstore']._dict.values())
                        if docs:
                            st.success(f"✅ Loaded {len(docs)} documents from FAISS docstore")
                            return docs
        
        # If still no documents, check if FAISS index exists
        index_faiss_path = f"{DB_FAISS_PATH}/index.faiss"
        if os.path.exists(index_faiss_path):
            st.info("FAISS index found but no documents. The app will work with limited functionality.")
            
        return []
                
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# =========================
# CREATE SIMPLE DOCUMENTS FROM FAISS
# =========================

@st.cache_resource
def create_simple_documents():
    """Create simple documents if none found"""
    try:
        import faiss
        
        index_faiss_path = f"{DB_FAISS_PATH}/index.faiss"
        if not os.path.exists(index_faiss_path):
            return []
        
        # Load FAISS index to get number of vectors
        index = faiss.read_index(index_faiss_path)
        num_vectors = index.ntotal
        
        # Create simple document objects
        documents = []
        
        class SimpleDocument:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        for i in range(min(num_vectors, 100)):
            doc = SimpleDocument(
                content=f"Medical document chunk {i+1} from the database",
                metadata={"source": f"chunk_{i+1}", "chunk_id": i}
            )
            documents.append(doc)
        
        if documents:
            st.info(f"Created {len(documents)} placeholder documents from FAISS index")
        
        return documents
    except Exception as e:
        st.error(f"Error creating documents: {e}")
        return []

# =========================
# SIMPLE KEYWORD RETRIEVAL
# =========================

def get_relevant_docs(documents, query, k=3):
    """Get relevant documents using simple keyword matching"""
    if not documents:
        return []
    
    # Stopwords to filter out common words
    stopwords = {"what", "is", "the", "how", "a", "an", "of", "to", "in", "on", 
                 "for", "with", "by", "at", "from", "as", "are", "was", "were", "be",
                 "this", "that", "these", "those", "it", "they", "we", "you", "i"}
    
    # Extract keywords from query
    query_words = [w.lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2]
    
    if not query_words:
        return documents[:k]
    
    # Score documents based on keyword matches
    scored_docs = []
    for doc in documents:
        # Extract text from document
        if hasattr(doc, 'page_content'):
            doc_text = doc.page_content.lower()
        elif isinstance(doc, dict):
            doc_text = doc.get('page_content', '').lower()
        else:
            continue
        
        # Calculate score
        score = sum(1 for word in query_words if word in doc_text)
        if score > 0:
            scored_docs.append((doc, score))
    
    # Sort by score and return top k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:k]]

# =========================
# EXTRACT DOCUMENT TEXT
# =========================

def extract_doc_text(doc):
    """Extract text from document object"""
    if hasattr(doc, 'page_content'):
        return doc.page_content
    elif isinstance(doc, dict):
        return doc.get('page_content', '')
    else:
        return str(doc)

def extract_doc_metadata(doc):
    """Extract metadata from document object"""
    if hasattr(doc, 'metadata'):
        return doc.metadata
    elif isinstance(doc, dict):
        return doc.get('metadata', {})
    else:
        return {}

# =========================
# GROQ API CALL
# =========================

def get_groq_response(prompt, context=None):
    """Call Groq API directly"""
    
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets!")
        return None
    
    client = Groq(api_key=api_key)
    
    if context:
        full_prompt = f"""You are a medical assistant. Answer ONLY from the context provided below.

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
# MAIN APP
# =========================

def main():
    st.set_page_config(page_title="Health AI Medical Assistant", page_icon="🏥", layout="wide")
    st.title("🏥 Health AI Medical Assistant")
    
    # Check API key
    if not st.secrets.get("GROQ_API_KEY", ""):
        st.error("⚠️ GROQ_API_KEY not configured! Add it in Secrets.")
        st.stop()
    
    # Check vectorstore files
    index_faiss_path = f"{DB_FAISS_PATH}/index.faiss"
    index_pkl_path = f"{DB_FAISS_PATH}/index.pkl"
    
    if not os.path.exists(index_faiss_path):
        st.error(f"❌ Vectorstore not found at {DB_FAISS_PATH}/")
        st.stop()
    
    # Load documents
    with st.spinner("📚 Loading medical database..."):
        documents = load_documents()
        
        if not documents:
            documents = create_simple_documents()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("Medical AI Assistant using Groq's Llama 3.1")
        
        if documents:
            st.markdown(f"**Documents:** {len(documents)}")
        
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
                        context = "\n\n".join([extract_doc_text(d) for d in relevant])
                        
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(relevant, 1):
                                meta = extract_doc_metadata(doc)
                                st.markdown(f"**{i}.** {meta.get('source', 'Unknown')}")
                
                response = get_groq_response(prompt, context)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
