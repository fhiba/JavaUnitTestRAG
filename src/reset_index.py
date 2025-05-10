from pinecone import Pinecone,ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name in [i.name for i in pc.list_indexes()]:
    print(f"Eliminando índice '{index_name}'...")
    pc.delete_index(index_name)

print(f"Creando índice '{index_name}' con dimensión 384...")

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
print("Índice creado exitosamente.")
