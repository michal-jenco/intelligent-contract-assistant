## 🧠 Developer Task: "Intelligent Contract Assistant" using LangChain and Python
Sample task for Blocshop/Erste

### 📋 Objective:
Build a contract assistant tool that takes in a PDF contract (e.g. an NDA or service agreement), extracts key clauses, and answers user questions about the contract using LangChain and an LLM (OpenAI or similar).

### 🧱 Requirements:
#### Document Ingestion

✅ Accept a PDF upload via CLI or basic web interface.

✅ Use PyPDF2 or pdfplumber to extract text.

✅ Chunk the document into manageable pieces using LangChain's TextSplitter.

#### Vector Store Setup

✅ Use FAISS or Chroma to create an embedding store from the chunks.

✅ Use OpenAIEmbeddings or similar (can be mocked if API is not available).

#### QA Interface

✅ Implement a basic interface (CLI or Streamlit app) where the user can:

✅ Ask questions like "What is the termination clause?" or "Who owns the IP?"

✅ Get responses grounded in the uploaded document using RetrievalQA.

#### Bonus (for higher difficulty)

Add a named entity recognition (NER) step to extract key fields like parties, dates, and monetary values.

Implement a feedback mechanism so the assistant can improve answers over multiple interactions.

✅ Show source text excerpts alongside answers.

### 🎯 Deliverables:
✅ Python code (ideally in a GitHub repo).

README.md with setup instructions.

✅ Sample contract PDF and example queries.

### ✅ Optional: a Streamlit or Flask app demoing the functionality.


MUST run python -m spacy download en_core_web_sm


1️⃣ Feedback Log (CSV)

Purpose: Keep a complete record of all user interactions, whether positive or negative.

Contents: Question, answer, sources, user rating, corrections (if any), file hash, timestamp.

Use cases:

Analyze trends (e.g., which questions often get negative feedback).

Audit for quality or debugging.

Build datasets for future fine-tuning or improving retrieval.

Essentially, it’s a historical dataset — never “overwrites” anything.

2️⃣ Corrections Store (JSON)

Purpose: Keep a current memory of corrections that can directly improve answers in future interactions.

Contents: Mapping of normalized question → corrected answer, file hash, timestamp.

Use cases:

When the same question comes up again, the assistant can override the model output or inject the correction into retrieval.

Acts as real-time learning memory, so the assistant “remembers” user fixes.

This is a live store, always read before generating an answer.