from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from datetime import datetime

import matplotlib
import numpy as np
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from datasets import Dataset, Features, Sequence, Value
from ragas.metrics import faithfulness
from ragas import evaluate
from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from plot_embeddings import (
    load_and_fit_embeddings,
    project_embeddings,
    plot_relevant_docs,
)

RERANKING = False
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
        model_kwargs={"device": "cuda"},
    )
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="default",
        text_key="text",
    )


def retrieve_k_similar_docs(db: PineconeVectorStore, query: str, k: int = 2):
    results = db.similarity_search_with_score(query=query, k=k)
    docs_list = [doc for doc, _ in results]
    orig_scores = [score for _, score in results]
    if RERANKING:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, d.page_content] for d in docs_list]
        rerank_scores = reranker.predict(pairs)
        order = np.argsort(rerank_scores)[::-1]
        docs_list = [docs_list[i] for i in order]
        orig_scores = [orig_scores[i] for i in order]
    reranked_results = list(zip(docs_list, orig_scores))
    return [d.page_content for d in docs_list], reranked_results


def generate_response(
        db: PineconeVectorStore | None,
        prompt: str,
        model,
        use_rag: bool
) -> str:
    base_instructions = """
You are an autonomous code generation agent. Your task is to write **new** unit tests using **JUnit 5** for a Java class provided.
IMPORTANT RULES:
1. Return **only** valid Java test code using `@Test` and assertions such as `assertEquals`, `assertTrue`, `assertThrows`.
2. Tests **must** be relevant to the user's request.
3. **Do not** include `import` statements, comments, class headers, or method explanations.
4. **Never** repeat code from the original class or documentation.
5. Generate at least one passing test and one failing test *if applicable*.
6. Assume the test class is already defined and has access to the class under test.
"""
    if use_rag and db is not None:
        docs, _ = retrieve_k_similar_docs(db, prompt)
        info_block = "\n\n".join(docs)
        full_prompt = f"""
{base_instructions}

CONTEXT INFORMATION (retrieved from knowledge base):
{info_block}

USER REQUEST:
{prompt}

Generate the test methods now.
"""
        chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True,
            verbose=False,
        )
        result = chain.invoke({"query": full_prompt})
    else:
        template_str = f"""{base_instructions}

USER REQUEST:
{{user_request}}

Generate the test methods now.
"""
        prompt_template = PromptTemplate(
            input_variables=["user_request"],
            template=template_str
        )
        llm_chain = LLMChain(llm=model, prompt=prompt_template)
        result = llm_chain.run(user_request=prompt)

    if isinstance(result, dict):
        result = result.get("result") or result.get("answer") or ""
    return str(result).strip()


def _save_plot_filename(class_name: str) -> str:
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{class_name}_embedding_{ts}.png"
    return os.path.join(plots_dir, filename)


def main() -> None:
    global RERANKING

    parser = argparse.ArgumentParser(
        description="Generate JUnit-5 tests via RAG or direct LLM, with optional reranking."
    )
    parser.add_argument("java_path", help="Path to the Java class file")
    parser.add_argument("--rerank", action="store_true", dest="reranking",
                        help="Enable cross-encoder reranking")
    parser.add_argument("--use_rag", action="store_true", dest="use_rag",
                        help="Enable RAG retrieval; if not set, fallback to direct LLM")
    args = parser.parse_args()

    RERANKING = args.reranking
    use_rag = args.use_rag

    try:
        path_str = args.java_path
        if not os.path.isfile(path_str) or not path_str.endswith(".java"):
            sys.exit(1)

        with open(path_str, 'r', encoding='utf-8') as f:
            class_source = f.read().strip()
        if not class_source:
            sys.exit(1)

        m = re.search(r"public\s+class\s+(\w+)", class_source)
        class_name = m.group(1) if m else "Output"
        suffix = "RAG" if use_rag else "no_rag"
        test_filename = f"{class_name}_{suffix}Test.java"

        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        db = get_pinecone_vectorstore() if use_rag else None
        llm = create_local_llm()

        test_methods = generate_response(db, class_source, llm, use_rag)
        out_path = os.path.join(output_dir, test_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(test_methods)
        label = "rag" if use_rag else "no_rag"
        print(f"[{label}] Test methods written to {out_path}", file=sys.stderr)

        if use_rag and db is not None:
            retrieved_docs, _ = retrieve_k_similar_docs(db, class_source, k=5)

            if not retrieved_docs or all(not doc.strip() for doc in retrieved_docs):
                print("[WARNING] Retrieved contexts are empty or invalid. Skipping faithfulness evaluation.",
                      file=sys.stderr)
            else:
                if not isinstance(test_methods, str):
                    test_methods = str(test_methods)

                print(f"\n[DEBUG] Number of contexts: {len(retrieved_docs)}")
                print(f"[DEBUG] First context length: {len(retrieved_docs[0]) if retrieved_docs else 0}")

                data_samples = {
                    'question': [class_source],
                    'answer': [test_methods],
                    'contexts': [retrieved_docs],
                }

                features = Features({
                    "question": Value("string"),
                    "answer": Value("string"),
                    "contexts": Sequence(Value("string")),
                })

                dataset = Dataset.from_dict(data_samples, features=features)

                try:
                    score = evaluate(dataset, metrics=[faithfulness], raise_exceptions=True)
                    df_scores = score.to_pandas()
                    print("\n=== RAG Metrics ===")
                    print(df_scores.to_string(index=False))
                except Exception as e:
                    print(f"\n[ERROR] Faithfulness evaluation failed: {str(e)}")
                    traceback.print_exc()

            query_vec = db.embeddings.embed_query(class_source)
            emb_matrix, umap_model = load_and_fit_embeddings(
                seed_vec=query_vec,
                top_k=1000,
                namespace="default",
            )
            projected_dataset = project_embeddings(emb_matrix, umap_model)
            projected_query = project_embeddings(np.array(query_vec), umap_model)
            retrieved_vecs = db.embeddings.embed_documents(retrieved_docs)
            retrieved_matrix = np.vstack(retrieved_vecs)
            projected_retrieved = project_embeddings(retrieved_matrix, umap_model)

            plt.figure(figsize=(8, 6))
            fig = plot_relevant_docs(
                projected_dataset,
                projected_query,
                projected_retrieved,
                title=f"{class_name} RAG Projection",
                save_path=None,
            )

            fig_to_save = fig if isinstance(fig, plt.Figure) else plt.gcf()
            save_path = _save_plot_filename(class_name)
            fig_to_save.savefig(save_path, bbox_inches="tight")
            plt.show()
            plt.close(fig_to_save)
            print(f"[RAG] Plot saved to {save_path}", file=sys.stderr)

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()