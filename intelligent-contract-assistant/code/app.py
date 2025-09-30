# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
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



@st.cache_resource
def init_ai(chunks: list):
    """Initialize AI components (LLM, retriever, chain)."""

    vector_store_maker = VectorStoreMaker()

    # Build vector store
    vector_store = vector_store_maker.get_vector_store(chunks)
    retriever = vector_store.as_retriever()

    # Define LLM + retrieval chain
    llm = ChatOpenAI(model="gpt-4o-mini")

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

    return chain


@st.cache_resource
def generate_summary() -> str:
    response = chain.invoke({"input": intro_query})
    answer = response["answer"]

    return answer


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

    # --- UI handling outside cache ---

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        input_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Show the PDF widget (not cached!)
        binary_data = uploaded_file.getvalue()
        pdf_viewer(input=binary_data, width=1000, height=700)

        # Process document (non-UI logic)
        pdf_ingest = PDFIngest()
        text_splitter = TextSplitter()

        pages = pdf_ingest.get_pages(input_file_path)
        full_text = "".join(pages)
        chunks = text_splitter.get_chunks(full_text)

        # Initialize AI only once (cached)
        chain = init_ai(chunks)

    else:
        st.warning("Please upload a PDF to get started.")
        chain = None

    # PDF displayer
    container_pdf, container_chat = st.columns([50, 50])

    # Query input
    st.subheader("Ask questions about the document:")

    intro_query = f"Summarize for me what the provided document talks about."

    if chain:
        with st.spinner("Generating summary..."):
            summary = generate_summary()
        st.subheader(summary)

    # User input box
    user_query = st.text_input("Your question", placeholder="Type your question here and press Enter")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
