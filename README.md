# 🏥 Health AI Medical Assistant

An intelligent medical information system that provides evidence-based answers from trusted medical literature using Retrieval-Augmented Generation (RAG).

## 📋 Overview

Health AI Medical Assistant combines the power of large language models with curated medical literature to deliver accurate, context-aware medical information. The system processes medical PDF documents, creates searchable vector embeddings, and uses Groq's Llama 3.1 model to generate precise answers based on the retrieved context.

### Key Features
- 🔍 Intelligent Document Retrieval: Searches through medical literature to find relevant information
- 📚 Evidence-Based Answers: Responses grounded in actual medical documents
- 💬 Interactive Chat Interface: User-friendly conversation flow
- 🔒 Private & Secure: Conversations remain confidential
- ⚡ Fast Response: Powered by Groq's high-performance inference
- 📱 Responsive Design: Works on desktop and mobile devices

## 🚀 Live Demo

Deployed on Streamlit Cloud: https://medibot.streamlit.app

## 📁 Project Structure

medical_chatbot/
├── data/                           # Medical PDF documents
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── vectorstore/                    # FAISS vector database
│   └── db_faiss/
│       ├── index.faiss            # FAISS index file
│       ├── index.pkl              # Index metadata
│       └── documents.pkl          # Processed documents
├── medibot.py                      # Main Streamlit application
├── create_memory_for_llm.py       # Vector database creator
├── extract_real_documents.py      # Document extraction utility
├── requirements.txt               # Python dependencies for non-pipenv users
├── Pipfile                         # Pipenv dependencies
├── Pipfile.lock                    # Locked dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation

## 🛠️ Technology Stack

- Frontend: Streamlit
- LLM: Groq (Llama 3.1 8B)
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector Database: FAISS
- Document Processing: PyPDF, LangChain
- Package Manager: Pipenv
- Language: Python 3.9+

## 📦 Installation

Prerequisites:
- Python 3.9 or higher
- Git
- Pipenv (pip install pipenv)
- Groq API key

Step 1: Clone the Repository
git clone https://github.com/abhay793/medibot.git
cd medibot

Step 2: Install Dependencies with Pipenv
pipenv install
pipenv shell

Step 3: Set Up Environment Variables
cp .env.example .env
Edit .env and add your Groq API key

Step 4: Add Medical Documents
mkdir -p data
cp /path/to/your/medical.pdf data/

Step 5: Create Vector Database
python create_memory_for_llm.py

Step 6: Extract Documents
python extract_real_documents.py

Step 7: Run the Application
streamlit run medibot.py

## 🚀 Deployment to Streamlit Cloud

1. Push Code to GitHub
git add .
git commit -m "Initial commit"
git push origin main

2. Deploy on Streamlit Cloud
- Go to share.streamlit.io
- Click New app
- Connect your GitHub repository
- Set main file path: medibot.py
- Add secrets: GROQ_API_KEY = your_groq_api_key_here
- Click Deploy

Note: Streamlit Cloud will automatically detect and install dependencies from requirements.txt or Pipfile.

## 💡 Usage Guide

Asking Questions:
Simply type your medical question in the chat input:
- What is cancer?
- How to manage diabetes?
- Symptoms of heart disease?
- Treatment options for hypertension?

How It Works:
1. Query Processing: Your question is analyzed for keywords
2. Document Retrieval: Relevant sections from medical literature are found
3. Context Building: Retrieved text is combined into context
4. Answer Generation: Llama 3.1 generates a response based on the context

## ⚠️ Disclaimer

This application is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any medical questions.

- The AI may not always provide accurate or complete information
- Medical knowledge is constantly evolving
- Always consult healthcare professionals for medical decisions
- In emergencies, contact emergency services immediately

## 🙏 Acknowledgments

- Groq for high-performance LLM inference
- LangChain for document processing
- Streamlit for the web framework
- Hugging Face for sentence transformers
- FAISS for vector similarity search
- Pipenv for dependency management

---

Built with ❤️ for better healthcare information access
