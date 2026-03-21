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
    """Load documents from vectorstore without FAISS index"""
    try:
        # Try to load documents.pkl first
        docs_path = f"{DB_FAISS_PATH}/documents.pkl"
        
        if not os.path.exists(docs_path):
            st.warning(f"No documents.pkl found at {docs_path}")
            
            # Try to load from index.pkl as fallback
            index_pkl_path = f"{DB_FAISS_PATH}/index.pkl"
            if os.path.exists(index_pkl_path):
                with open(index_pkl_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        documents = data.get('documents', data.get('chunks', []))
                        if documents:
                            return documents
            return []
        
        # Load documents.pkl
        with open(docs_path, "rb") as f:
            data = pickle.load(f)
            
            # Handle different data structures
            if isinstance(data, list):
                documents = data
                return documents
            elif isinstance(data, dict):
                documents = data.get('documents', data.get('chunks', []))
                if documents:
                    return documents
            elif isinstance(data, tuple):
                # Try to extract documents from tuple
                documents = []
                for item in data:
                    if hasattr(item, 'page_content'):
                        documents.append(item)
                if documents:
                    return documents
            else:
                return []
                
    except Exception as e:
        st.error(f"Error loading documents: {e}")
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
        # If no meaningful keywords, return first k documents
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

ANSWER:"""
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
    # Page configuration
    st.set_page_config(
        page_title="Health AI Medical Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title
    st.title("🏥 Health AI Medical Assistant")
    
    # Check API key first
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
    
    # Load documents
    with st.spinner("📚 Loading medical database..."):
        documents = load_documents()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant provides information based on:
        - Medical documents in the database
        - Keyword-based document retrieval
        - Groq's Llama 3.1 model for answers
        """)
        
        if documents:
            st.markdown(f"### 📊 Database Stats")
            st.markdown(f"- **Documents loaded:** {len(documents)}")
            
            # Show document sources
            sources = set()
            for doc in documents:
                metadata = extract_doc_metadata(doc)
                source = metadata.get('source', 'Unknown')
                if source != 'Unknown':
                    sources.add(source.split('/')[-1])
            
            if sources:
                st.markdown("### 📄 Documents")
                for source in list(sources)[:5]:
                    st.markdown(f"- {source}")
        
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This is an AI assistant for informational purposes only.
        Always consult qualified healthcare professionals for medical advice.
        """)
    
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
                context = None
                relevant_docs = []
                
                # Get relevant documents if available
                if documents:
                    relevant_docs = get_relevant_docs(documents, prompt, k=3)
                    if relevant_docs:
                        # Build context from relevant documents
                        contexts = []
                        for doc in relevant_docs:
                            doc_text = extract_doc_text(doc)
                            if doc_text:
                                contexts.append(doc_text)
                        context = "\n\n".join(contexts)
                        
                        # Show sources in expander
                        with st.expander("📚 Retrieved from medical documents"):
                            for i, doc in enumerate(relevant_docs, 1):
                                metadata = extract_doc_metadata(doc)
                                source = metadata.get('source', 'Unknown')
                                chunk_id = metadata.get('chunk_id', '')
                                
                                st.markdown(f"**Source {i}:** `{source}`")
                                if chunk_id:
                                    st.markdown(f"**Chunk ID:** {chunk_id}")
                                
                                doc_text = extract_doc_text(doc)
                                if len(doc_text) > 300:
                                    st.markdown(f"{doc_text[:300]}...")
                                else:
                                    st.markdown(doc_text)
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

if __name__ == "__main__":
    main()
