# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import streamlit as st
import random

from streamlit import feedback
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
from feedback_handler import FeedbackHandler



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

def create_tempfile(uploaded_file) -> str:
    temp_dir = tempfile.mkdtemp()
    input_file_path = os.path.join(temp_dir, uploaded_file.name)

    return input_file_path

def ingest_pdf(input_file_path) -> list[str]:
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

def get_random_key() -> int:
    return random.getrandbits(64)


if __name__ == "__main__":
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Streamlit UI setup
    st.set_page_config(page_title="Intelligent Contract Assistant", layout="wide")
    st.title("Intelligent Contract Assistant")
    st.markdown("Ask questions about your PDF in natural language.")
    # PDF displayer
    container_pdf, container_chat = st.columns([50, 50])
    # Query input
    st.subheader("Ask questions about the document:")

    # Sidebar
    st.sidebar.header("Settings")
    num_sources = st.sidebar.slider("Number of Sources", min_value=1, max_value=5, value=1)
    source_max_length = st.sidebar.slider("Source length trim", min_value=10, max_value=500, value=100)

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        input_file_path = create_tempfile(uploaded_file)
        chunks = ingest_pdf(input_file_path)
        feedback_handler = FeedbackHandler(input_file_path)
        chain = init_ai(chunks)

    else:
        st.warning("Please upload a PDF to get started.")
        chain = None
        exit()

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
    if "feedback" not in st.session_state:
        st.session_state.feedback = []

    # Handle query
    if user_question and user_question not in st.session_state.question_history:
        # Check corrections before retrieval
        correction_entry = feedback_handler.check_corrections(user_question)

        if correction_entry:
            st.success("✅ Using corrected answer from feedback memory")
            answer = correction_entry["correction"]
            sources = ["Manual user input from a previous session"]
        else:
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
        for qa_idx, entry in enumerate(st.session_state.chat_history, 1):
            question = entry["question"]
            answer = entry["result"]
            sources = entry["source_documents"]

            st.markdown(f"**Q{qa_idx}:** {question}")
            st.markdown(f"**A{qa_idx}:** {answer}")

            sources_full_string = ""

            st.write("### Sources")
            for src_idx, doc in enumerate(sources[0:num_sources], 1):
                st.markdown(f"*Source {src_idx}:*")
                source_text = sources[0] if isinstance(sources, list) else doc.page_content[:source_max_length]
                sources_full_string += f"{source_text}\n"
                st.write(source_text)

            st.write("### Feedback")

            # Stable keys per QA entry
            feedback_key = f"feedback_{qa_idx}"
            correction_key = f"correction_{qa_idx}"
            submit_key = f"submit_{qa_idx}"

            feedback_radio = st.radio(
                "Was this answer helpful?",
                ["👍 Yes", "👎 No"],
                key=feedback_key,
                horizontal=True,
            )

            # Show text area only if "No" is selected
            if feedback_radio == "👎 No":
                correction_text = st.text_area(
                    "Provide the correct answer:",
                    key=correction_key,
                )
            else:
                correction_text = ""

            # Submit button per QA
            if st.button("Submit Feedback", key=submit_key):
                feedback_handler.log_feedback(question, answer, sources_full_string, feedback_radio, correction_text)

                if feedback_radio == "👎 No" and correction_text:
                    feedback_handler.save_correction(question, correction_text)
                st.success("✅ Feedback saved!")

            st.markdown("---")
