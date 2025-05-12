import os
import sys
import re
import traceback
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
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
            model_kwargs={"device": "cuda"}
        )

        return PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace="default",
            text_key="text"
        )
    except Exception:
        traceback.print_exc()
        raise

def retrieve_k_similar_docs(db, query, k=2):
    results = db.similarity_search_with_score(
        query=query,
        k=k
    )
    return [doc.page_content for doc, _ in results], results

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
        docs, _ = retrieve_k_similar_docs(db, prompt)
        info = "\n\n".join(docs)
        full_prompt = f"""
You are an autonomous code generation agent. Your task is to write **new** unit tests using **JUnit 5** for a Java class provided.
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
{info}

USER REQUEST:
{prompt}

Generate the test methods now.
"""
        result = chain.invoke({"query": full_prompt})
        if isinstance(result, dict):
            return result.get("result") or result.get("answer") or str(result)
        return result
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}"

def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: python RAG.py <path/to/YourClass.java>", file=sys.stderr)
            sys.exit(1)

        java_path = sys.argv[1]
        if not os.path.isfile(java_path) or not java_path.endswith(".java"):
            print(f"Error: '{java_path}' is not a valid .java file path.", file=sys.stderr)
            sys.exit(1)

        with open(java_path, "r", encoding="utf-8") as f:
            class_content = f.read().strip()

        print(f"DEBUG ── user_input repr: {repr(class_content[:80] + '...' if len(class_content) > 80 else class_content)}", file=sys.stderr)
        if not class_content:
            print("Error: Java file is empty.", file=sys.stderr)
            sys.exit(1)

        m = re.search(r'public\s+class\s+(\w+)', class_content)
        class_name = m.group(1) if m else 'Output'
        test_filename = f"{class_name}Test.java"

        db = get_pinecone_vectorstore()
        print("Generating response.")
        response = generate_response(db, class_content)

        with open(test_filename, 'w', encoding='utf-8') as out:
            out.write(response)
        print(f"Test methods written to {test_filename}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
