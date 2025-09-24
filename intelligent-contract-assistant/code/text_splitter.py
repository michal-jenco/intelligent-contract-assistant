# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self):
        pass

    def get_chunks(self, text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            # max characters per chunk
            chunk_size=500,
            # overlap between chunks to keep context
            chunk_overlap=50,
            # how to measure chunk size
            length_function=len,
            # split priority
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(text)

        return chunks
