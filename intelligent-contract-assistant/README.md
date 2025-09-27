# intelligent-contract-assistant
Sample task for Blocshop/Erste


🧠 Developer Task: "Intelligent Contract Assistant" using LangChain and Python
📋 Objective:
Build a contract assistant tool that takes in a PDF contract (e.g. an NDA or service agreement), extracts key clauses, and answers user questions about the contract using LangChain and an LLM (OpenAI or similar).

🧱 Requirements:
Document Ingestion

Accept a PDF upload via CLI or basic web interface.

✅ Use PyPDF2 or pdfplumber to extract text.

✅ Chunk the document into manageable pieces using LangChain's TextSplitter.

Vector Store Setup

✅ Use FAISS or Chroma to create an embedding store from the chunks.

✅ Use OpenAIEmbeddings or similar (can be mocked if API is not available).

QA Interface

Implement a basic interface (CLI or Streamlit app) where the user can:

Ask questions like "What is the termination clause?" or "Who owns the IP?"

Get responses grounded in the uploaded document using RetrievalQA.

Bonus (for higher difficulty)

Add a named entity recognition (NER) step to extract key fields like parties, dates, and monetary values.

Implement a feedback mechanism so the assistant can improve answers over multiple interactions.

Show source text excerpts alongside answers.

🎯 Deliverables:
Python code (ideally in a GitHub repo).

README.md with setup instructions.

Sample contract PDF and example queries.

Optional: a Streamlit or Flask app demoing the functionality.
