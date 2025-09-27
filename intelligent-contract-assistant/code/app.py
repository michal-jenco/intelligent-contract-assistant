# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


# ============================================================
#  Project:     Intelligent Contract Assistant - Streamlit App
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo
# ============================================================


import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pdf_ingest import PDFIngest
from text_splitter import TextSplitter
from vector_store import VectorStoreMaker


if __name__ == '__main__':
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Streamlit UI setup
    st.set_page_config(page_title="Intelligent Contract Assistant", layout="wide")
    st.title("Intelligent Contract Assistant")
    st.markdown("Ask questions about your PDF in natural language.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    input_file_path = ""
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        input_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

    with st.spinner("Processing document..."):
        pdf_ingest = PDFIngest()
        text_splitter = TextSplitter()
        vector_store_maker = VectorStoreMaker()

        pages = pdf_ingest.get_pages(input_file_path)
        st.success(f"The document contains {len(pages)} pages.")

        # Join full text
        full_text = "".join(pages)

        # Split into chunks
        chunks = text_splitter.get_chunks(full_text)
        st.success(f"Split document into {len(chunks)} chunks.")

        # Build vector store
        vector_store = vector_store_maker.get_vector_store(chunks)
        if not vector_store:
            st.error("Failed to create vector store.")
            st.stop()

        retriever = vector_store.as_retriever()

        # Define LLM + retrieval chain
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

    # Query input
    st.subheader("Ask questions about the document:")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input box
    user_query = st.text_input("Your question", placeholder="Type your question here and press Enter")

    # Handle query
    if user_query:
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": user_query})
        answer = response["answer"]

        # Save to history
        st.session_state.chat_history.append({"question": user_query, "answer": answer})

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for i, entry in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {entry['question']}")
            st.markdown(f"**A{i}:** {entry['answer']}")
            st.markdown("---")
