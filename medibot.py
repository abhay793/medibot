import os
import streamlit as st
from groq import Groq

def get_groq_response(prompt):
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    
    client = Groq(api_key=api_key)
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"You are a medical assistant. Answer: {prompt}"}],
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    st.title("🏥 Health AI Medical Assistant")
    
    if not st.secrets.get("GROQ_API_KEY", ""):
        st.error("⚠️ GROQ_API_KEY not configured!")
        st.stop()
    
    with st.sidebar:
        st.info("💡 To add medical documents, create a 'data' folder with PDF files")
    
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
                response = get_groq_response(prompt)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()