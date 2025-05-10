import os
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def get_pinecone_vectorstore():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={"device": "cpu"}
    )

    return PineconeVectorStore(index=index, embedding=embeddings, namespace="default", text_key="text")

def generate_response(db, prompt):
    """
    Generate a response with a LLM based on previous custom context
    :return: chatbot response
    """
    
    hf_llm = HuggingFaceEndpoint(
        model="google/flan-t5-base",
        task="text2text-generation",
        temperature=0.3,
        top_k=30,
        repetition_penalty=1.2,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    chain = RetrievalQA.from_chain_type(
        llm=hf_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        verbose=False
    )

    return chain.invoke(prompt)

def main():
    db = get_pinecone_vectorstore()
    print("Ask anything (Ctrl+C to exit):")
    while True:
        try:
            user_input = input("> ")
            if user_input.strip() == "":
                continue
            response = generate_response(db, user_input)
            print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()