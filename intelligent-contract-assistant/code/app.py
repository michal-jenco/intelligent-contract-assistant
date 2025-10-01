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
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

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

    system_prompt = """
        You are an assistant. Answer the question using the context below.
        If you don't know the answer, say you don't know.
        Use three sentences maximum and keep the answer concise.

        Context: {context}
        Question: {question}

        Answer:
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain

@st.cache_resource
def generate_summary(_chain) -> str:
    response = _chain.invoke({"query": intro_query})
    answer = response["result"]

    return answer

@st.cache_resource
def ask_ai(user_query: str) -> dict:
    response = chain.invoke({"query": user_query})

    return response


def ingest_pdf() -> list[str]:
    temp_dir = tempfile.mkdtemp()
    input_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Show the PDF widget
    binary_data = uploaded_file.getvalue()
    pdf_viewer(input=binary_data, width=1000, height=700)

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

    # Sidebar
    st.sidebar.header("Settings")
    num_sources = st.sidebar.slider("Number of Sources", min_value=1, max_value=5, value=1)
    source_max_length = st.sidebar.slider("Source length trim", min_value=10, max_value=500, value=100)

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        chunks = ingest_pdf()
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
    user_question = st.text_input(
        "Your question",
        placeholder="Type your question here and press Enter",
        key="input_box",
    )

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "question_history" not in st.session_state:
        st.session_state.question_history = []

    # Handle query
    if user_question and user_question not in st.session_state.question_history:
        with st.spinner("Thinking..."):
            response = ask_ai(user_question)
        answer = response["result"]
        sources = response["source_documents"]

        # Save to history
        st.session_state.chat_history.append(
            {"question": user_question,
             "result": answer,
             "source_documents": sources,
             }
        )
        st.session_state.question_history.append(user_question)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for qa_idx, entry in enumerate(st.session_state.chat_history[::-1], 1):
            st.markdown(f"**Q{qa_idx}:** {entry["question"]}")
            st.markdown(f"**A{qa_idx}:** {entry["result"]}")

            sources = entry["source_documents"]

            st.write("### Sources")
            for src_idx, doc in enumerate(sources[0:num_sources], 1):
                st.markdown(f"*Source {src_idx}:*")
                st.write(doc.page_content[:source_max_length])

            st.markdown("---")
