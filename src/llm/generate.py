import json
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.build_index import load_documents


def setup_document_retrieval():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    documents = load_documents("/home/mahmoud/projects/RegionAwareVisionAssistant/docs")
    vector_store = QdrantVectorStore.from_documents(documents, embedding=embeddings, location=":memory:", collection_name="methodology_docs")
    return vector_store

def prompt_using_retrieved_documents(query, vector_store, results, metrics, category):
    # Retrieve relevant documents based on the query
    retrieved_chunks = vector_store.similarity_search(query, k=3)

    # Combine the retrieved documents into a single context string
    context = "\n".join([doc.page_content for doc in retrieved_chunks])

    # Create a prompt that includes the context and the original query
    prompt = f"""\
        You are explaining an anomaly detection result to a QA operative.

        Context (retrieved documentation):
        {context}

        Result to explain:
        Category: {category}
        pred_label: {results['pred_label']}
        Anomaly score: {results['pred_score']}
        model metrics: {metrics}

        Follow the explanation rules from the context. Do not state a probability
        or confidence percentage.
        """.strip()
    return prompt


def main():
    result_idx = 0
    vector_store = setup_document_retrieval()
    category = "leather"
    results_root_path = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{category}"
    results = json.load(open(f"{results_root_path}/{category}_test_scores.json", 'r'))[result_idx]
    metrics = json.load(open(f"{results_root_path}/{category}_final_metrics.json", 'r'))
    print("writing prompt...")
    prompt = prompt_using_retrieved_documents("leather defect threshold", vector_store, results, metrics, category)
    print(prompt)
    model = OllamaLLM(model="qwen3:8b")
    print("invoking model...")

    result = model.invoke(input=prompt)
    print(result)


if __name__ == "__main__":
    main()