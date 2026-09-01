import os
from langchain_core.documents import Document

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
    documents = load_documents("/home/mahmoud/projects/RegionAwareVisionAssistant/docs")
    print(len(documents))