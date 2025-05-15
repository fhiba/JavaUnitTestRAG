import umap
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone
import os
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from src.RAG import retrieve_k_similar_docs

embedding_function=HuggingFaceEmbeddings(  # embedding=OpenAIEmbeddings() rate limit
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}  # TODO CHANGE IF NOT USING GPU
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vectors = index.query(vector=[0] * 384, top_k=10000, include_values=True)
embeddings = [v.values for v in vectors.matches]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

def project_embeddings(embeddings, umap_transform):
  """
  Projects high-dimensional embeddings into a 2D space using UMAP.

  Parameters:
  embeddings (numpy.ndarray): A 2D array-like object containing the embeddings to be transformed,
                              where each row represents an embedding vector.
  umap_transform (umap.UMAP): A pre-trained UMAP model to perform the transformation.

  Returns:
  numpy.ndarray: A 2D array where each row is a 2D embedding resulting from the UMAP transformation.
  """
  umap_embeddings = np.empty((len(embeddings),2))   # Mappeamos desde la longitud de nuestros embeddings a 2D

  for i, embedding in enumerate(tqdm(embeddings)):
      umap_embeddings[i] = umap_transform.transform([embedding])

  return umap_embeddings

projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

def plot_embeddings(projected_dataset_embeddings):
    """
    Plots 2D projected embeddings using Plotly Express.

    Parameters:
    projected_dataset_embeddings (numpy.ndarray): A 2D array containing the projected embeddings,
                                                  where each row represents a point in 2D space.
    """

    df_embeddings = pd.DataFrame(projected_dataset_embeddings, columns=['x', 'y'])

    fig = px.scatter(df_embeddings, x='x', y='y', title='Projected Embeddings')

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        title_x=0.5
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # Maintain aspect ratio
    fig.show()

def plot_relevant_docs(projected_dataset_embeddings,
                       projected_query_embedding,
                       projected_retrieved_embeddings,
                       query, title, projected_augmented_embeddings=[]):
    """
    Plots the query, dataset, and retrieved documents in 2D embedding space using Plotly.

    Parameters:
    projected_dataset_embeddings (numpy.ndarray): 2D array of projected dataset embeddings.
    projected_query_embedding (numpy.ndarray): 2D array containing the projected query embedding.
    projected_retrieved_embeddings (numpy.ndarray): 2D array of projected embeddings for retrieved documents.
    query (str): The query text to be displayed on hover.
    title (str): Title of the plot.
    """
    # Convert data to DataFrame for easier plotting
    df_dataset = pd.DataFrame(projected_dataset_embeddings, columns=['x', 'y'])
    df_query = pd.DataFrame(projected_query_embedding, columns=['x', 'y'])
    df_query['text'] = query  # Add query text for hover information
    df_retrieved = pd.DataFrame(projected_retrieved_embeddings, columns=['x', 'y'])
    df_augmented = pd.DataFrame(projected_augmented_embeddings, columns=['x', 'y'])

    # Create a scatter plot for the dataset points
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_dataset['x'],
        y=df_dataset['y'],
        mode='markers',
        marker=dict(size=10, color='gray'),
        name='Dataset Embeddings'
    ))

    fig.add_trace(go.Scatter(
        x=df_query['x'],
        y=df_query['y'],
        mode='markers',
        marker=dict(size=15, symbol='x', color='red'),
        name='Query Embedding',
        hovertext=[query],  # Add hover text
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=df_retrieved['x'],
        y=df_retrieved['y'],
        mode='markers',
        marker=dict(size=12, symbol='circle-open', line=dict(color='green', width=2)),
        name='Retrieved Embeddings'
    ))

    fig.add_trace(go.Scatter(
        x=df_augmented['x'],
        y=df_augmented['y'],
        mode='markers',
        marker=dict(size=15, symbol='x', color='pink'),
        name='Augmented Embeddings'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        title_x=0.5
    )

    fig.show()

def project_and_plot_relevant_docs(query, title):
    """
    Retrieves, projects, and plots documents that are relevant to the given query.
    
    Parameters:
    query (str): The query text to search for similar documents.
    title (str): Title for the resulting plot.
    
    Returns:
    list: The retrieved documents that are relevant to the query.
    """
    # Pass the index to retrieve_k_similar_docs as the first parameter
    retrieved_documents, results = retrieve_k_similar_docs(index, query, k=5)
    
    # Use the proper embedding function from imported embeddings
    query_embedding = embedding_function.embed_query(query)
    
    # Make sure the structure matches what's returned by retrieve_k_similar_docs
    retrieved_embeddings = results['embeddings']
    
    projected_query_embedding = project_embeddings([query_embedding], umap_transform)
    projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

    plot_relevant_docs(
        projected_dataset_embeddings, 
        projected_query_embedding, 
        projected_retrieved_embeddings,
        query, 
        title
    )
    
    return retrieved_documents

if __name__ == "__main__":
    plot_embeddings(projected_dataset_embeddings)