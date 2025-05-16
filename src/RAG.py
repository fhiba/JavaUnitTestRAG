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
import matplotlib.pyplot as plt

# Import for custom RAGAS evaluation with Ollama
from typing import List, Dict, Any

matplotlib.use('Agg')

# Import your embedding visualization module
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


def create_local_llm(model_name="mistral"):
    """Create a local LLM using Ollama."""
    return Ollama(model=model_name, temperature=0.3)


def get_pinecone_vectorstore() -> PineconeVectorStore:
    """Initialize and return a PineconeVectorStore with HuggingFace embeddings."""
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
    """Retrieve k similar documents from the vector store, with optional reranking."""
    results = db.similarity_search_with_score(query=query, k=k)
    docs_list = [doc for doc, _ in results]
    orig_scores = [score for _, score in results]

    if RERANKING:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [[query, d.page_content] for d in docs_list]
            rerank_scores = reranker.predict(pairs)
            order = np.argsort(rerank_scores)[::-1]
            docs_list = [docs_list[i] for i in order]
            orig_scores = [orig_scores[i] for i in order]
        except ImportError:
            print("Warning: sentence-transformers not available. Skipping reranking.")

    reranked_results = list(zip(docs_list, orig_scores))
    return [d.page_content for d in docs_list], reranked_results


def generate_response(
        db: PineconeVectorStore | None,
        prompt: str,
        model,
        use_rag: bool
) -> str:
    """Generate a response using either RAG or direct LLM."""
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


# Custom RAGAS-like evaluation using Ollama
def evaluate_faithfulness(
        question: str,
        answer: str,
        contexts: List[str],
        llm_model: Any
) -> Dict[str, float]:
    """
    Evaluate faithfulness of the answer based on contexts using Ollama.
    This function mimics RAGAS faithfulness evaluation without requiring OpenAI.
    """
    # Prepare the evaluation prompt
    eval_prompt = f"""
Task: Evaluate the faithfulness of the generated answer based on the provided contexts.
Faithfulness measures if the answer contains information that is not present in the contexts.

Question: 
{question}

Answer:
{answer}

Contexts:
{' '.join(contexts)}

Evaluate the faithfulness of the answer on a scale of 0 to 1, where:
- 1 means the answer is completely faithful to the contexts (all information comes from contexts)
- 0 means the answer contains significant information not found in contexts

First, list any claims in the answer that aren't supported by the contexts.
Then provide a final score between 0 and 1.
Output only the final score as a number at the end.
"""

    # Get evaluation from Ollama
    try:
        evaluation_result = llm_model.invoke(eval_prompt)

        # Extract score - look for the last number in the text
        score_matches = re.findall(r'(\d+\.\d+|\d+)', evaluation_result)
        if score_matches:
            score = float(score_matches[-1])
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
        else:
            print("Warning: Could not extract faithfulness score. Using default 0.5")
            score = 0.5

        return {"faithfulness": score}
    except Exception as e:
        print(f"Error in faithfulness evaluation: {str(e)}")
        return {"faithfulness": 0.5}  # Default neutral score


def _save_plot_filename(class_name: str) -> str:
    """Generate a filename for saving the embedding plot."""
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
    parser.add_argument("--model", default="mistral", help="Ollama model to use (default: mistral)")
    args = parser.parse_args()

    RERANKING = args.reranking
    use_rag = args.use_rag
    model_name = args.model

    try:
        # Import torch here to avoid issues if not installed
        try:
            import torch
        except ImportError:
            print("Warning: PyTorch not found. Using CPU for embeddings.")

        path_str = args.java_path
        if not os.path.isfile(path_str) or not path_str.endswith(".java"):
            print(f"Error: {path_str} is not a valid Java file.", file=sys.stderr)
            sys.exit(1)

        with open(path_str, 'r', encoding='utf-8') as f:
            class_source = f.read().strip()
        if not class_source:
            print("Error: Java file is empty.", file=sys.stderr)
            sys.exit(1)

        m = re.search(r"public\s+class\s+(\w+)", class_source)
        class_name = m.group(1) if m else "Output"
        suffix = "RAG" if use_rag else "no_rag"
        test_filename = f"{class_name}_{suffix}Test.java"

        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the database and LLM
        db = get_pinecone_vectorstore() if use_rag else None
        llm = create_local_llm(model_name=model_name)

        # Generate test methods
        test_methods = generate_response(db, class_source, llm, use_rag)
        out_path = os.path.join(output_dir, test_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(test_methods)
        label = "rag" if use_rag else "no_rag"
        print(f"[{label}] Test methods written to {out_path}", file=sys.stderr)

        # If using RAG, perform evaluation and visualization
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

                # Use custom Ollama-based evaluation instead of RAGAS
                faithfulness_score = evaluate_faithfulness(
                    question=class_source,
                    answer=test_methods,
                    contexts=retrieved_docs,
                    llm_model=llm
                )

                print("\n=== RAG Metrics ===")
                print(f"Faithfulness: {faithfulness_score['faithfulness']:.4f}")

                # Create a simple visualization of the scores
                try:
                    plt.figure(figsize=(6, 4))
                    plt.bar(['Faithfulness'], [faithfulness_score['faithfulness']], color='blue')
                    plt.ylim(0, 1)
                    plt.ylabel('Score')
                    plt.title('RAG Evaluation Metrics')
                    metrics_plot_path = os.path.join("plots",
                                                     f"{class_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(metrics_plot_path)
                    plt.close()
                    print(f"[RAG] Metrics plot saved to {metrics_plot_path}", file=sys.stderr)
                except Exception as e:
                    print(f"[WARNING] Failed to create metrics plot: {str(e)}", file=sys.stderr)

            # Create embedding visualization
            try:
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
                plt.close(fig_to_save)
                print(f"[RAG] Plot saved to {save_path}", file=sys.stderr)
            except Exception as e:
                print(f"[WARNING] Failed to create embedding visualization: {str(e)}", file=sys.stderr)
                traceback.print_exc()

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
