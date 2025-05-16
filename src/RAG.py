from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from plot_embeddings import (
    load_and_fit_embeddings,
    project_embeddings,
    plot_relevant_docs,
)

load_dotenv()

def _ensure_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Environment variable '{var}' is not set.")
    return val


def create_local_llm():
    return Ollama(model="mistral", temperature=0.3)


def get_pinecone_vectorstore() -> PineconeVectorStore:
    api_key = _ensure_env("PINECONE_API_KEY")
    index_name = _ensure_env("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # switch to "cuda" if GPU available
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="default",
        text_key="text",
    )


def retrieve_k_similar_docs(db: PineconeVectorStore, query: str, k: int = 2):
    results = db.similarity_search_with_score(query=query, k=k)
    return [doc.page_content for doc, _ in results], results


def generate_response(db: PineconeVectorStore, prompt: str) -> str:
    hf_llm = create_local_llm()
    chain = RetrievalQA.from_chain_type(
        llm=hf_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        return_source_documents=True,
        verbose=False,
    )
    docs, _ = retrieve_k_similar_docs(db, prompt)
    info_block = "\n\n".join(docs)

    full_prompt = f"""
You are an autonomous code generation agent. Your task is to write **new** unit tests using **JUnit 5** for a Java class provided.
IMPORTANT RULES:
1. Return **only** valid Java test code using `@Test` and assertion methods such as `assertEquals`, `assertTrue`, `assertThrows`.
2. Tests **must** be relevant to the user's request.
3. **Do not** include `import` statements, comments, class headers, or method explanations.
4. **Never** repeat code from the original class or documentation.
5. Generate at least one passing test and one failing test *if applicable*.
6. Assume the test class is already defined and has access to the class under test.

CONTEXT INFORMATION (retrieved from knowledge base):
{info_block}

USER REQUEST:
{prompt}

Generate the test methods now.
"""

    result = chain.invoke({"query": full_prompt})
    if isinstance(result, dict):
        return result.get("result") or result.get("answer") or str(result)
    return str(result)

def _save_plot_filename(class_name: str) -> str:
    Path("plots").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"plots/{class_name}_embedding_{ts}.png"


def main() -> None:
    try:
        if len(sys.argv) != 2:
            print("Usage: python RAG.py <path/to/YourClass.java>", file=sys.stderr)
            sys.exit(1)

        java_path = Path(sys.argv[1])
        if not java_path.is_file() or java_path.suffix != ".java":
            print(f"Error: '{java_path}' is not a valid .java file path.", file=sys.stderr)
            sys.exit(1)

        class_source = java_path.read_text(encoding="utf-8").strip()
        if not class_source:
            print("Error: Java file is empty.", file=sys.stderr)
            sys.exit(1)


        m = re.search(r"public\s+class\s+(\w+)", class_source)
        class_name = m.group(1) if m else "Output"
        test_filename = f"{class_name}Test.java"

        db = get_pinecone_vectorstore()
        print("[RAG] Generating JUnit‑5 tests…", file=sys.stderr)
        test_methods = generate_response(db, class_source)

        Path(test_filename).write_text(test_methods, encoding="utf-8")
        print(f"[RAG] Test methods written to {test_filename}")

        query_vec = db.embedding.embed_query(class_source)

        # Load a sample of dataset embeddings & fit UMAP
        emb_matrix, umap_model = load_and_fit_embeddings(
            seed_vec=query_vec,
            top_k=1000,
            namespace="default",
        )

        projected_dataset = project_embeddings(emb_matrix, umap_model)
        projected_query = project_embeddings(np.array(query_vec), umap_model)
        retrieved_docs, _ = retrieve_k_similar_docs(db, class_source, k=5)
        retrieved_vecs = db.embedding.embed_documents(retrieved_docs)
        retrieved_matrix = np.vstack(retrieved_vecs)
        projected_retrieved = project_embeddings(retrieved_matrix, umap_model)

        plot_path = plot_relevant_docs(
            projected_dataset,
            projected_query,
            projected_retrieved,
            title=f"{class_name} RAG Projection",
            save_path=_save_plot_filename(class_name),
        )
        print(f"[RAG] Plot saved to {plot_path}")

    except Exception:  # pragma: no cover
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
