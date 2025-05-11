import os
import traceback
import sys
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.llms import HuggingFacePipeline
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def create_local_llm():
    model_id = "Salesforce/codet5p-770m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        top_k=30,
        repetition_penalty=1.2,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=pipe)


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

        result = chain.invoke({"query": prompt})

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
