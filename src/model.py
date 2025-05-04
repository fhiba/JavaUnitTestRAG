import logging
import os

import langchain
import openai
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] - %(message)s ',
                    handlers=[
                        logging.FileHandler('/content/langchaindemo.log', mode='w'),
                        logging.StreamHandler(),
                    ],
                    force=True)
logger = logging.getLogger(__name__)
logger.info("Langchain Demo Initialized")

def get_docs():
    """
    Loads each file into one document (knowledge base)
    :return: docs
    """

    loader = DirectoryLoader(  # Reads custom data from local files
        path="docs",
        glob="*.txt",
        loader_cls=TextLoader,  # Loader class to use for loading files
    )

    docs = loader.load()
    return docs

docs = get_docs()
docs[1]

def get_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """
    Get chunks from docs. Our loaded doc may be too long for most models, and even if it fits it can struggle to find relevant context. So we generate chunks
    :param docs: docs to be splitted

    :return: chunks
    """

    text_splitter = RecursiveCharacterTextSplitter( # recommended splitter for generic text. split documents recursively by different characters - starting with "\n\n", then "\n", then " "
        chunk_size=chunk_size,        # max size (in terms of number of characters) of the final documents
        chunk_overlap=chunk_overlap,  # how much overlap there should be between chunks
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks

chunks = get_chunks(docs)
chunks

from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings( #  embedding=OpenAIEmbeddings() rate limit
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'} #TODO CHANGE IF NOT USING GPU
)
