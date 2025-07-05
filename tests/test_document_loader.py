import os
from pathlib import Path
from src.processors.document_loader import DocumentLoader
from src.config.settings import settings


def test_find_markdown_files():
    loader = DocumentLoader()
    files = loader.find_markdown_files(Path(settings.docs_path) / "grammar")
    assert isinstance(files, list)
    assert all(str(f).endswith(".md") for f in files)
    assert any("stmt.md" in str(f) for f in files)


def test_process_single_file():
    loader = DocumentLoader()
    file_path = Path(settings.docs_path) / "grammar" / "stmt.md"
    doc = loader.process_single_file(file_path)
    assert doc is not None
    assert doc.title
    assert doc.content
    assert doc.language in ("en", "zh")
    assert doc.category == "grammar"


def test_load_documents():
    loader = DocumentLoader()
    docs = loader.load_documents(max_workers=2)
    assert isinstance(docs, list)
    assert all(hasattr(doc, "title") for doc in docs)
    assert any(doc.category == "grammar" for doc in docs)


def test_create_all_chunks():
    loader = DocumentLoader()
    docs = loader.load_documents(max_workers=2)
    chunks = loader.create_all_chunks(docs)
    assert isinstance(chunks, list)
    assert all(hasattr(chunk, "content") for chunk in chunks)
    assert all(chunk.language in ("en", "zh") for chunk in chunks)


def test_get_document_statistics():
    loader = DocumentLoader()
    docs = loader.load_documents(max_workers=2)
    stats = loader.get_document_statistics(docs)
    assert "total_documents" in stats
    assert "languages" in stats
    assert "categories" in stats
    assert stats["total_documents"] == len(docs) 