# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import PyPDF2


class PDFIngest:
    def __init__(self):
        self.default_pdf_path = f"../documents/NDA.pdf"

    def get_pages(self, file_path: str = None) -> list[str]:
        if not file_path:
            file_path = self.default_pdf_path

        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)

            pages = []

            for page in reader.pages:
                pages.append(page.extract_text())
            return pages
