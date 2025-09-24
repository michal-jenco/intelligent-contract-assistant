# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


from pdf_ingest import PDFIngest
from text_splitter import TextSplitter


if __name__ == '__main__':
    pdf_ingest = PDFIngest()
    text_splitter = TextSplitter()

    pages = pdf_ingest.get_pages()

    print(f"The document contains {len(pages)} pages.")

    full_text = ""
    for page in pages:
        full_text += page

    chunks = text_splitter.get_chunks(full_text)

    print(f"Split document into {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} contains {len(chunk)} characters.")
