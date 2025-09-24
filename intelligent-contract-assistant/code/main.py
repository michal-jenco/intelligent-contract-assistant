# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


from dotenv import load_dotenv
import os
import openai

from pdf_ingest import PDFIngest
from text_splitter import TextSplitter
from vector_store import VectorStoreMaker


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    pdf_ingest = PDFIngest()
    text_splitter = TextSplitter()
    vector_store_maker = VectorStoreMaker()

    pages = pdf_ingest.get_pages()

    print(f"The document contains {len(pages)} pages.")

    full_text = ""
    for page in pages:
        full_text += page

    chunks = text_splitter.get_chunks(full_text)

    print(f"Split document into {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} contains {len(chunk)} characters.")

    vector_store = vector_store_maker.get_vector_store(chunks)

    if not vector_store:
        exit()

    query = "What is this document about?"
    results = vector_store.similarity_search(query, k=3)

    for i, r in enumerate(results):
        print(f"Result {i + 1}: {r.page_content}\n")