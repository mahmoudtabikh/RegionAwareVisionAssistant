import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

def get_paths(root_dir):
    paths_dict = {dir.replace('.md', ''): f"{root_dir}/{dir}" for dir in os.listdir(root_dir)}
    return paths_dict

def get_category_from_filename(filename):
    if "wood" in filename.lower():
        return "wood"
    elif "leather" in filename.lower():
        return "leather"
    else:
        return "general"

def get_doctype_from_filename(filename):
    if "methodology" in filename.lower():
        return "methodology"
    elif "category_performance" in filename.lower():
        return "category_performance"
    elif "examples" in filename.lower():
        return "examples"
    else:
        return "rules"

def load_documents(root_dir):
    paths_dict = get_paths(root_dir)
    return [
        Document(
            page_content=open(doc_path, "r").read(),
            metadata={
                "path": doc_path,
                "name": name,
                "category": get_category_from_filename(name),
                "doc_type": get_doctype_from_filename(name),
            }
        )
        for name, doc_path in paths_dict.items()
    ]

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    docs_path = os.environ.get("DOCS_PATH", "docs")
    documents = load_documents(docs_path)
    QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        location=QDRANT_URL,
        collection_name="methodology_docs",
    )
    print(f"Indexed {len(documents)} documents into Qdrant at {QDRANT_URL}")

