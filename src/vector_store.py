import os
from dotenv import load_dotenv
from pinecone import Pinecone, CloudProvider, AwsRegion, EmbedModel, IndexEmbed

load_dotenv()

def get_vector_store():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    existing = pc.list_indexes()
    existing_names = [index.name for index in existing]

    if index_name not in existing_names:
        pc.create_index_for_model(
            name=index_name,
            cloud= CloudProvider.AWS,
            region= AwsRegion.US_EAST_1,
            embed= IndexEmbed(
                model= EmbedModel.Multilingual_E5_Large,
                field_map={"text": "text"},
                metric="cosine"
            )
        )

    return pc.Index(index_name), index_name