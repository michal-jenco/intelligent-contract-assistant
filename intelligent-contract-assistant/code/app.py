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
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pdf_ingest import PDFIngest
from text_splitter import TextSplitter
from vector_store import VectorStoreMaker



@st.cache_resource
def init_ai(chunks: list[str]):
    """Initialize AI components (LLM, retriever, chain)."""

    vector_store_maker = VectorStoreMaker()

    # Build vector store
    vector_store = vector_store_maker.get_vector_store(chunks)
    retriever = vector_store.as_retriever()

    # Define LLM + retrieval chain
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Optional: tweak your prompt
    prompt_template = """You are an assistant. Answer the question using the context below:

    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # or "map_reduce", "refine"
        retriever=retriever,
        return_source_documents=True  # this is what gives us the source docs
    )

    return chain


@st.cache_resource
def generate_summary(_chain) -> str:
    response = _chain.invoke({"query": intro_query})
    answer = response["result"]

    return answer


def ingest_pdf() -> list[str]:
    temp_dir = tempfile.mkdtemp()
    input_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Show the PDF widget
    binary_data = uploaded_file.getvalue()
    pdf_viewer(input=binary_data, width=1000, height=700)

    # Process document (non-UI logic)
    pdf_ingest = PDFIngest()
    text_splitter = TextSplitter()

    pages = pdf_ingest.get_pages(input_file_path)
    full_text = "".join(pages)
    chunks = text_splitter.get_chunks(full_text)

    return chunks


if __name__ == "__main__":
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Streamlit UI setup
    st.set_page_config(page_title="Intelligent Contract Assistant", layout="wide")
    st.title("Intelligent Contract Assistant")
    st.markdown("Ask questions about your PDF in natural language.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        chunks = ingest_pdf()

        # Initialize AI only once (cached)
        chain = init_ai(chunks)

    else:
        st.warning("Please upload a PDF to get started.")
        chain = None
        exit()

    # PDF displayer
    container_pdf, container_chat = st.columns([50, 50])

    # Query input
    st.subheader("Ask questions about the document:")

    intro_query = f"Summarize for me what the provided document talks about."

    if chain:
        with st.spinner("Generating summary..."):
            summary = generate_summary(chain)
        st.subheader(summary)

    # User input box
    user_query = st.text_input(
        "Your question",
        placeholder="Type your question here and press Enter",
        key="input_box",
    )

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    sources = []

    # Handle query
    if user_query:
        with st.spinner("Thinking..."):
            response = chain.invoke({"query": user_query})
        answer = response["result"]
        sources = response["source_documents"]

        # Save to history
        st.session_state.chat_history.append({"question": user_query, "result": answer})

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for i, entry in enumerate(st.session_state.chat_history[::-1], 1):
            st.markdown(f"**Q{i}:** {entry["question"]}")
            st.markdown(f"**A{i}:** {entry["result"]}")
            st.markdown("---")

            st.write("### Sources")
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:500])  # show only first 500 chars
                if "source" in doc.metadata:
                    st.caption(f"From: {doc.metadata["source"]}")
