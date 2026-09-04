import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.build_index import load_documents

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

docs_path = os.environ.get("DOCS_PATH", "docs")
documents = load_documents(docs_path)

vector_store = QdrantVectorStore.from_documents(documents, embedding=embeddings, location=":memory:", collection_name="methodology_docs")

vector_store.similarity_search("wood defect threshold")
