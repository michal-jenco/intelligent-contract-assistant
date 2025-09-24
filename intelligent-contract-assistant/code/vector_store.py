# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class VectorStoreMaker:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def get_vector_store(self, chunks: list[str]) -> FAISS | None:
        # Store chunks and embeddings in FAISS

        try:
            vectorstore = FAISS.from_texts(chunks, self.embeddings)
        except Exception as e:
            print(f"Unable to create a Vector Store, reason: {e}")
            return None

        return vectorstore
