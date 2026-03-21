import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# LOAD PDFs

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Add filename metadata
    for doc in documents:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown")

    print(f"Loaded {len(documents)} pages")
    return documents

# CLEAN TEXT

def clean_text(documents):
    for doc in documents:
        text = doc.page_content
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        doc.page_content = text
    return documents


# CHUNKING

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = chunk.metadata.get("source", "Unknown")

    print(f"Created {len(chunks)} chunks")
    return chunks

# EMBEDDINGS 

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# SAVE FAISS

def create_faiss_db(chunks, embedding_model):
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS DB created!")

# MAIN

if __name__ == "__main__":
    docs = load_pdf_files(DATA_PATH)
    docs = clean_text(docs)
    chunks = create_chunks(docs)
    embedding_model = get_embedding_model()
    create_faiss_db(chunks, embedding_model)