from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.build_index import load_documents

model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)

documents = load_documents("/home/mahmoud/projects/RegionAwareVisionAssistant/docs")

vector_store = QdrantVectorStore.from_documents(documents, embedding=model, location=":memory:", collection_name="methodology_docs")

vector_store.similarity_search("wood defect threshold")
