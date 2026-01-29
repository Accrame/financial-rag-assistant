# Tests for document loading and chunking

import tempfile
from pathlib import Path

import pytest


class TestFinancialDocumentLoader:

    def test_load_text_file(self):
        from src.document_processing.loader import FinancialDocumentLoader

        loader = FinancialDocumentLoader()

        # Create temp text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document about revenue growth.")
            tmp_path = f.name

        try:
            docs = loader.load(tmp_path)
            assert len(docs) == 1
            assert "revenue growth" in docs[0].page_content
            assert docs[0].metadata["file_type"] == "txt"
        finally:
            Path(tmp_path).unlink()

    def test_generate_doc_id(self):
        """Doc IDs should be deterministic."""
        from src.document_processing.loader import FinancialDocumentLoader

        loader = FinancialDocumentLoader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            tmp_path = f.name

        try:
            path = Path(tmp_path)
            id1 = loader._generate_doc_id(path)
            id2 = loader._generate_doc_id(path)

            assert id1 == id2
            assert len(id1) == 12
        finally:
            Path(tmp_path).unlink()


class TestFinancialDocumentChunker:

    def test_chunk_text(self):
        from src.document_processing.chunker import FinancialDocumentChunker

        chunker = FinancialDocumentChunker(chunk_size=100, chunk_overlap=20)

        text = "This is a test. " * 50  # Long enough to require chunking
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 150 for chunk in chunks)  # Allow some flexibility

    def test_chunk_documents(self):
        """Metadata should be preserved on chunks."""
        from langchain_core.documents import Document

        from src.document_processing.chunker import FinancialDocumentChunker

        chunker = FinancialDocumentChunker(chunk_size=100, chunk_overlap=20)

        doc = Document(
            page_content="This is a test. " * 50, metadata={"source": "test.pdf", "page": 1}
        )

        chunked = chunker.chunk_documents([doc])

        assert len(chunked) > 1
        for chunk in chunked:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_empty_document(self):
        """Empty docs should produce no chunks."""
        from langchain_core.documents import Document

        from src.document_processing.chunker import FinancialDocumentChunker

        chunker = FinancialDocumentChunker()

        doc = Document(page_content="", metadata={})
        chunked = chunker.chunk_documents([doc])

        assert len(chunked) == 0


class TestEmbeddings:

    def test_local_embeddings(self):
        try:
            from src.vectorstore.embeddings import get_embeddings

            embeddings = get_embeddings(provider="local")

            texts = ["Test document one", "Test document two"]
            vectors = embeddings.embed_documents(texts)

            assert len(vectors) == 2
            assert len(vectors[0]) > 0  # Has dimensions

        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_query_embedding(self):
        try:
            from src.vectorstore.embeddings import get_embeddings

            embeddings = get_embeddings(provider="local")
            vector = embeddings.embed_query("What is the revenue?")

            assert len(vector) > 0

        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestVectorStore:

    @pytest.fixture
    def temp_store(self, tmp_path):
        try:
            from src.vectorstore.embeddings import get_embeddings
            from src.vectorstore.store import VectorStoreManager

            embeddings = get_embeddings(provider="local")
            store = VectorStoreManager(
                embeddings=embeddings,
                persist_directory=str(tmp_path / "test_chroma"),
                collection_name="test_collection",
            )

            yield store

            # Cleanup
            store.clear()

        except ImportError:
            pytest.skip("Required packages not installed")

    def test_add_and_search(self, temp_store):
        """Basic add + search roundtrip."""
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="Revenue increased by 15% to $10 billion",
                metadata={"source": "10k.pdf", "page": 1},
            ),
            Document(
                page_content="The company faces cybersecurity risks",
                metadata={"source": "10k.pdf", "page": 5},
            ),
        ]

        temp_store.add_documents(docs)

        assert temp_store.get_document_count() == 2

        results = temp_store.similarity_search_simple("revenue growth", k=1)
        assert len(results) == 1
        assert "Revenue" in results[0].page_content

    def test_delete_documents(self, temp_store):
        from langchain_core.documents import Document

        doc = Document(page_content="Test content", metadata={"doc_id": "test123"})

        ids = temp_store.add_documents([doc])
        assert temp_store.get_document_count() == 1

        temp_store.delete_documents(ids)
        assert temp_store.get_document_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
