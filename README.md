## 🧠 Developer Task: "Intelligent Contract Assistant" using LangChain and Python - Sample task for Blocshop/Erste

## Installation and instructions
1. run 'pip install -r requirements.txt' to install all the packages
2. run 'streamlit run app.py' in the command line to start the web app
3. in case of missing "en_core_web_sm", run this manually: 'python -m spacy download en_core_web_sm'  

## Example queries and answers for the provided document    
- #### Can you name the parties between which the contract was signed?
  - The parties involved in the contract are the principal and the service provider, which is represented by Befree Pty Ltd. James Parker is the sole director and secretary of Befree Pty Ltd.
- #### What is the termination clause?
  - The termination clause allows either party to terminate the agreement by giving written notice of their intention to terminate, effective after 7 days from the date the notice is given. The agreement will then terminate at the end of that 7-day period.
- #### What is the tax invoice payment time frame?
  - The principal must pay tax invoices within 7 days after receiving them.

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

✅ Add a named entity recognition (NER) step to extract key fields like parties, dates, and monetary values.

✅ Implement a feedback mechanism so the assistant can improve answers over multiple interactions.

✅ Show source text excerpts alongside answers.

### 🎯 Deliverables:
✅ Python code (ideally in a GitHub repo).

✅ README.md with setup instructions.

✅ Sample contract PDF and example queries.

### ✅ Optional: a Streamlit or Flask app demoing the functionality.

## Explanation of using both a Feedback Log and Corrections store

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