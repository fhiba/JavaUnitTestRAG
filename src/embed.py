import json
import re
from dotenv import load_dotenv
from vector_store import get_vector_store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

import logging

load_dotenv()

with open("../datasets/dataset.json", "r") as f:
    data = json.load(f)


embeddings = HuggingFaceEmbeddings(  # embedding=OpenAIEmbeddings() rate limit
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cuda'}  # TODO CHANGE IF NOT USING GPU
)


def main():

    index, index_name = get_vector_store()



    records = []

    for item in data:
        m = re.search(r"class\s+(\w+)", item["class"])
        doc_id = m.group(1) if m else str(data.index(item))

        text = item["class"] + "\n" + item.get("description", "")
        chunks = get_chunks(text)
        for chunk in chunks:
            embedding = embeddings.get_embedding(chunks)

            records.append({
                "id":       doc_id,
                "value":     embedding,
                "class":    item["class"],
                "test":     item.get("tests", ""),
                "description": item.get("description", "")
            })

    index.upsert_records(records=records, namespace="default")
    print(f"Upserted {len(records)} records into '{index_name}'")

def get_chunks(docs, chunk_size=750, chunk_overlap=150):
    """
    Get chunks from docs. Our loaded doc may be too long for most models, and even if it fits it can struggle to find relevant context. So we generate chunks
    :param docs: docs to be splitted

    :return: chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        # recommended splitter for generic text. split documents recursively by different characters - starting with "\n\n", then "\n", then " "
        chunk_size=chunk_size,  # max size (in terms of number of characters) of the final documents
        chunk_overlap=chunk_overlap,  # how much overlap there should be between chunks
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks


logger = logging.getLogger(__name__)
logger.info("Langchain Demo Initialized")


if __name__ == "__main__":
    main()
