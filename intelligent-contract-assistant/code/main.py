# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


from pdf_ingest import PDFIngest


if __name__ == '__main__':
    pdf_ingest = PDFIngest()

    pages = pdf_ingest.get_pages()

    print(f"The document contains {len(pages)} pages.")
    print(pages)
