from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.build_index import load_documents

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

documents = load_documents("/home/mahmoud/projects/RegionAwareVisionAssistant/docs")

vector_store = QdrantVectorStore.from_documents(documents, embedding=embeddings, location=":memory:", collection_name="methodology_docs")

vector_store.similarity_search("wood defect threshold")
