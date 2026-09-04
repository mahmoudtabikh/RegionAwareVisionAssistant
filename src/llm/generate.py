import json
import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.build_index import load_documents
from qdrant_client.models import Filter, FieldCondition, MatchAny
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

def setup_document_retrieval():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name="methodology_docs",
    )
    return vector_store

def prompt_using_retrieved_documents(vector_store, prediction, threshold):
    # Retrieve relevant documents based on the query
    category = prediction['category']
    region_summary = [
    {"region_id": r["region_id"], "area": r["area"], "compactness": r["compactness"]}
    for r in prediction["regions"]
]
    query = f"{category} defect threshold"
    retrieved_chunks = vector_store.similarity_search(
        query,
        k=3,
        filter=Filter(
            must=[
                FieldCondition(
                key="metadata.category",
                match=MatchAny(any=[category, "general"])
            )
        ]
        )
        )

    # Combine the retrieved documents into a single context string
    context = "\n".join([doc.page_content for doc in retrieved_chunks])

    # Create a prompt that includes the context and the original query
    prompt = f"""\
        You are explaining an anomaly detection result to a QA operative.

        Context (retrieved documentation):
        {context}

        Result to explain:
        Category: {category}
        Anomaly score: {prediction['pred_score']}
        Threshold: {threshold}
        regions detected: {region_summary}

        Follow the explanation rules from the context. Do not state a probability
        or confidence percentage.
        Do not reference file names or suggest the reader consult additional documents.
        Write the explanation as a complete, standalone response.
        "low compactness may indicate thresholding noise rather than a genuine defect — per the methodology doc.
        """.strip()
    return prompt


def call_qa_model_with_prediction(vector_store, model, prediction, threshold):
    prompt = prompt_using_retrieved_documents(vector_store, prediction, threshold)
    return model.invoke(input=prompt)

if __name__ == "__main__":

    results = json.load(open("/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/leather_test_full.json", 'r'))
    category = "leather"
    results_root_path = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{category}"
    final_metrics = json.load(open("/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/leather_final_metrics.json", 'r'))
    threshold = final_metrics["threshold"]

    vector_store = setup_document_retrieval()
    model = OllamaLLM(model="qwen3:8b")
    explanation = call_qa_model_with_prediction(vector_store, model, prediction=results[0], threshold=threshold)
    print(explanation)
