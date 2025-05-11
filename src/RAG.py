import os
import traceback
import sys
from dotenv import load_dotenv
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.llms import HuggingFacePipeline
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama


load_dotenv()



def create_local_llm():
    return Ollama(model="mistral", temperature=0.3)

def get_pinecone_vectorstore():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY no está configurado en el archivo .env")

        pc = Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME no está configurado en el archivo .env")

        index = pc.Index(index_name)

        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={"device": "cpu"}
        )

        return PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace="default",
            text_key="text"
        )
    except Exception as e:
        traceback.print_exc()
        raise


def retrieve_k_similar_docs(db,query, k=5):
    results = db.similarity_search_with_score(
        query=query,
        k=k
    )

    retrieved_documents = [doc.page_content for doc, score in results]

    return retrieved_documents, results

def generate_response(db, prompt):
    try:
        hf_llm = create_local_llm()

        chain = RetrievalQA.from_chain_type(
            llm=hf_llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True,
            verbose=True
        )
        retrieved_documents, results = retrieve_k_similar_docs(db,prompt)
        information = "\n\n".join(retrieved_documents)
        message_content = f"""You are an autonomous code generation agent. Your task is to write **new** unit tests using **JUnit 5** for a Java class provided.
                            IMPORTANT RULES:
                            1. You MUST return only valid Java test code using `@Test` and assertion methods such as `assertEquals`, `assertTrue`, `assertThrows`.
                            2. You MUST write tests that are relevant to the user's request.
                            3. DO NOT include `import` statements, comments, class headers, or method explanations.
                            4. NEVER repeat code from the original class or documentation.
                            5. You MUST generate at least one passing test and one failing test **if applicable**.
                            6. You may assume the test class is already defined and has access to the class under test.
                            
                            RESPONSE FORMAT:
                            Only raw Java methods
                            
                            CONTEXT INFORMATION (retrieved from knowledge base):
                            {information}
                            
                            USER REQUEST:
                            {prompt}
                            
                            Generate the test methods now."""

        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=20, tokens_per_chunk=256)

        token_split_texts = []
        token_split_texts += token_splitter.split_text(message_content)

        result = chain.invoke({"query": message_content})

        if isinstance(result, dict):
            if "result" in result:
                return result["result"]
            elif "answer" in result:
                return result["answer"]
            else:
                return str(result)
        return result
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}"


def main():
    try:
        db = get_pinecone_vectorstore()
        print("Ask anything (Ctrl+C to exit or type 'exit'):")
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() == "exit":
                    sys.exit(0)
                if user_input == "":
                    continue
                print("Generating response...")
                response = generate_response(db, user_input)
                print(response)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            except Exception as e:
                traceback.print_exc()
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    main()
