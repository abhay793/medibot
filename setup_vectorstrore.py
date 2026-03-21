import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

def setup_vectorstore():
    """Setup vectorstore from PDFs in data folder"""
    
    st.info("📚 Setting up medical database from PDFs...")
    
    # Check if data folder exists
    if not os.path.exists(DATA_PATH):
        st.error(f"Data folder '{DATA_PATH}' not found!")
        return False
    
    # Load PDFs
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        st.error("No PDF files found in data folder!")
        return False
    
    # Process documents
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.replace("\n", " ").split())
    
    # Create chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    # Create embeddings and FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    
    # Save
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    
    # Save documents
    with open(f"{DB_FAISS_PATH}/documents.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    st.success(f"✅ Vectorstore created with {len(chunks)} chunks!")
    return True

if __name__ == "__main__":
    setup_vectorstore()