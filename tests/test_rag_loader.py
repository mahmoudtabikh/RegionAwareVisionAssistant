from src.rag import build_index


def test_get_paths_maps_document_names_to_paths(tmp_path):
    (tmp_path / "wood_methodology.md").write_text("Methodology")
    (tmp_path / "leather_examples.md").write_text("Examples")

    paths = build_index.get_paths(str(tmp_path))

    assert paths == {
        "wood_methodology": f"{tmp_path}/wood_methodology.md",
        "leather_examples": f"{tmp_path}/leather_examples.md",
    }


def test_get_paths_returns_empty_mapping_for_empty_directory(tmp_path):
    assert build_index.get_paths(str(tmp_path)) == {}


def test_get_category_from_filename():
    assert build_index.get_category_from_filename("WOOD_notes.md") == "wood"
    assert build_index.get_category_from_filename("leather_notes.md") == "leather"
    assert build_index.get_category_from_filename("general_notes.md") == "general"


def test_get_doctype_from_filename():
    assert build_index.get_doctype_from_filename("methodology.md") == "methodology"
    assert (
        build_index.get_doctype_from_filename("category_performance.md")
        == "category_performance"
    )
    assert build_index.get_doctype_from_filename("examples.md") == "examples"
    assert build_index.get_doctype_from_filename("other.md") == "rules"


def test_load_documents_reads_content_and_sets_metadata(tmp_path):
    filename = "leather_methodology.md"
    content = "# Leather methodology\nInspection details."
    (tmp_path / filename).write_text(content)

    documents = build_index.load_documents(str(tmp_path))

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == content
    assert document.metadata == {
        "path": f"{tmp_path}/{filename}",
        "name": "leather_methodology",
        "category": "leather",
        "doc_type": "methodology",
    }