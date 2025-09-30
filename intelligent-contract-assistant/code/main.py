# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


from dotenv import load_dotenv
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pdf_ingest import PDFIngest
from text_splitter import TextSplitter
from vector_store import VectorStoreMaker
from ner import NamedEntityRecognizer


# I used this file to implement and debug the functionality before porting it into streamlit


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    pdf_ingest = PDFIngest()
    text_splitter = TextSplitter()
    vector_store_maker = VectorStoreMaker()
    ner = NamedEntityRecognizer()

    pages = pdf_ingest.get_pages()
    fields = pdf_ingest.get_fields()

    if fields:
        for field_name, value in fields.items():
            field_value = value.get('/V', None)
            print(field_name, ':', field_value)

    print(f"The document contains {len(pages)} pages.")

    full_text = ""
    for page in pages:
        full_text += page

    # print(f"\n\n{full_text}\n\n")

    chunks = text_splitter.get_chunks(full_text)

    print(f"Split document into {len(chunks)} chunks.\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} contains {len(chunk)} characters.")
    print()

    vector_store = vector_store_maker.get_vector_store(chunks)

    if not vector_store:
        exit()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vector_store.as_retriever()


    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    query = f"Summarize for me what the provided document talks about."

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    response = chain.invoke({"query": query})

    print(response["result"])
    print()

    entities = ner.get_entities(text=full_text)

    for entity in entities:
        print(f"{entity.text}, {entity.label_}")
