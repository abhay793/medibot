import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

def extract_real_documents():
    """Extract real documents from PDFs and save them"""
    
    print("📚 Loading PDFs from data/ directory...")
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data directory not found: {DATA_PATH}")
        print("Please make sure your PDFs are in the 'data' folder")
        return False
    
    # Load all PDFs
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")
        return False
    
    if not documents:
        print(f"❌ No PDF files found in {DATA_PATH}/")
        print("Please add your medical PDF files to the 'data' folder")
        return False
    
    print(f"✅ Loaded {len(documents)} pages from PDFs")
    
    # Clean text
    for doc in documents:
        text = doc.page_content
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        doc.page_content = text
    
    # Create chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = chunk.metadata.get("source", "Unknown")
    
    # Save chunks to documents.pkl
    docs_path = os.path.join(DB_FAISS_PATH, "documents.pkl")
    with open(docs_path, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"✅ Saved {len(chunks)} real documents to documents.pkl")
    
    # Also save a text preview
    preview_path = os.path.join(DB_FAISS_PATH, "documents_preview.txt")
    with open(preview_path, "w") as f:
        f.write(f"Total documents: {len(chunks)}\n\n")
        f.write("=== First 3 Documents Preview ===\n\n")
        for i, chunk in enumerate(chunks[:3]):
            f.write(f"Document {i+1}:\n")
            f.write(f"Source: {chunk.metadata.get('source', 'Unknown')}\n")
            f.write(f"Content: {chunk.page_content[:300]}...\n\n")
    
    print(f"✅ Saved preview to documents_preview.txt")
    
    return True

if __name__ == "__main__":
    extract_real_documents()