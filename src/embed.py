import json
import re
from dotenv import load_dotenv
from vector_store import get_vector_store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter


import logging

load_dotenv()

with open("../datasets/dataset.json", "r") as f:
    data = json.load(f)


embeddings = HuggingFaceEmbeddings(  # embedding=OpenAIEmbeddings() rate limit
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}  # TODO CHANGE IF NOT USING GPU
)


def main():

    index, index_name = get_vector_store()



    records = []

    for item in data:
        m = re.search(r"class\s+(\w+)", item["class"])
        doc_id = m.group(1) if m else str(data.index(item))

        text = item["class"] + "\n" + item.get("description", "")

        json_str = json.dumps(text,ensure_ascii=False)
        chunks = get_chunks(json_str,750,150)
        for i,chunk in enumerate(chunks):
            embedding = embeddings.embed_query(chunk)
            if embedding is None:
                continue
            records.append({
                "id":       f"{doc_id}__{i}",
                "values":     embedding,
                "metadata": {
                    "text":     text,
                    "class":    item["class"],
                    "test":     item.get("tests", ""),
                    "description": item.get("description", "")
                }
            })

    index.upsert(vectors=records, namespace="default")
    print(f"Upserted {len(records)} records into '{index_name}'")

def get_chunks(text: str, chunk_size: int, chunk_overlap: int):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    # <-- use split_text() on a raw string
    chunks = splitter.split_text(text)
    logger.info(f"Split text of length {len(text)} into {len(chunks)} chunks.")
    return chunks


logger = logging.getLogger(__name__)
logger.info("Langchain Demo Initialized")


if __name__ == "__main__":
    main()
