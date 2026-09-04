import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.build_index import load_documents

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

docs_path = os.environ.get("DOCS_PATH", "docs")
documents = load_documents(docs_path)

vector_store = QdrantVectorStore.from_documents(documents, embedding=embeddings, location=QDRANT_URL, collection_name="methodology_docs")

vector_store.similarity_search("wood defect threshold")
