import os
from datetime import datetime
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import umap
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


def _default_save_path(prefix: str) -> str:
    """
    Generate a default filepath under 'plots/' for saving images.
    """
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join("plots", f"{prefix}_{timestamp}.png")


def load_and_fit_embeddings(
    seed_vec: list[float] | np.ndarray,
    *,
    top_k: int = 1000,
    namespace: str = "default",
    dim: int = 384,
) -> Tuple[np.ndarray, "umap.UMAP"]:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    resp = index.query(
        vector=seed_vec,
        top_k=top_k,
        namespace=namespace,
        include_values=True,
    )
    emb_list = [m.values for m in resp.matches]
    if not emb_list:
        raise RuntimeError("No embeddings returned from Pinecone query.")

    emb_matrix = np.vstack(emb_list)
    reducer = umap.UMAP(n_components=2, metric="cosine")
    reducer.fit(emb_matrix)
    return emb_matrix, reducer


def project_embeddings(
    emb: np.ndarray | list[float], reducer: "umap.UMAP"
) -> np.ndarray:
    arr = np.array(emb)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return reducer.transform(arr)


def plot_relevant_docs(
    projected_dataset: np.ndarray,
    projected_query: np.ndarray,
    projected_retrieved: np.ndarray,
    title: str = "RAG Embeddings",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the full embedding space, highlight the query point and retrieved points.
    Returns a matplotlib Figure.
    """
    if save_path is None:
        save_path = _default_save_path(prefix=title.replace(" ", "_"))

    fig, ax = plt.subplots(figsize=(8, 6))
    # Corpus
    ax.scatter(
        projected_dataset[:, 0],
        projected_dataset[:, 1],
        s=20,
        alpha=0.3,
        label="Corpus"
    )
    # Query
    ax.scatter(
        projected_query[:, 0],
        projected_query[:, 1],
        s=100,
        marker="x",
        label="Query"
    )
    # Retrieved
    ax.scatter(
        projected_retrieved[:, 0],
        projected_retrieved[:, 1],
        s=80,
        facecolors="none",
        edgecolors="green",
        label="Retrieved"
    )

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig
