import pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone


def get_vector_store(index_name, embeddings, embedding_size=384):
    """Creates vector store from Pinecone for storing and managing embeddings.

    :param str index_name: The name of the index to create or retrieve from Pinecone.
    :param str embeddings: The embedding function to be used to generate embeddings
    :param int embedding_size: The size (dimension) of the embeddings. Defaults to 384 (e.g., for sentence-transformers/all-MiniLM-L6-v2).

    :return: PineconeVectorStore: An object representing the vector store in Pinecone for managing embeddings.

    :raise: ValueError: If the index creation fails due to invalid parameters or connection issues.
    """

    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )  # Pinecone is initialized using an API key stored in the environment variable

    if (
        INDEX_NAME not in pc.list_indexes().names()
    ):  # Check whether an index with the given index_name already exists
        pc.create_index(
            name=INDEX_NAME,  # Name of the index
            dimension=embedding_size,  # Size of the vectors (embeddings)
            metric="cosine",  # Distance metric used to compare vectors
            spec=ServerlessSpec(  # Determines the infrastructure used
                cloud="aws",  # Specifies that the Pinecone index is hosted on AWS
                region="us-east-1",  # Specifies the region of the cloud provider
            ),
        )

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embeddings
    )  # initializes a PineconeVectorStore object using the index_name and the provided embeddings model or function

    return vectorstore
