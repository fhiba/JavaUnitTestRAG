from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

load_dotenv()

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
        raise RuntimeError(
            "No vectors retrieved from Pinecone. "
            "Check that the index is populated and that the namespace is correct."
        )

    emb_matrix = np.vstack(emb_list)  # shape (n, dim)
    umap_model = umap.UMAP(random_state=0).fit(emb_matrix)
    return emb_matrix, umap_model

def project_embeddings(vectors: np.ndarray, umap_model: "umap.UMAP") -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return umap_model.transform(vectors)


def _default_save_path(prefix: str = "embedding_plot") -> str:
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("plots", f"{prefix}_{ts}.png")


def plot_relevant_docs(
    projected_dataset: np.ndarray,
    projected_query: np.ndarray,
    projected_retrieved: np.ndarray,
    *,
    title: str = "Embedding Projection",
    save_path: str | None = None,
) -> str:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=projected_dataset[:, 0],
            y=projected_dataset[:, 1],
            mode="markers",
            name="Dataset",
            marker=dict(size=6, color="gray"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=projected_query[:, 0],
            y=projected_query[:, 1],
            mode="markers",
            name="Query",
            marker=dict(size=12, symbol="x", color="red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=projected_retrieved[:, 0],
            y=projected_retrieved[:, 1],
            mode="markers",
            name="Retrieved",
            marker=dict(size=10, symbol="circle-open", color="green"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title_x=0.5,
        showlegend=True,
    )

    if save_path is None:
        save_path = _default_save_path(prefix=title.replace(" ", "_"))

    fig.write_image(save_path)
    fig.show()

    return save_path

__all__ = [
    "load_and_fit_embeddings",
    "project_embeddings",
    "plot_relevant_docs",
]
