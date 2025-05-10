import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def get_vector_store():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    target_dimension = 384

    existing = pc.list_indexes()
    existing_names = [index.name for index in existing]

    if index_name not in existing_names:
        pc.create_index(
            name=index_name,
            dimension=target_dimension,
            metric="cosine"
        )
    else:
        desc = pc.describe_index(index_name)
        if desc.dimension != target_dimension:
            raise ValueError(
                f"Index '{index_name}' has dimension {desc.dimension}, expected {target_dimension}"
            )

    return pc.Index(index_name), index_name
