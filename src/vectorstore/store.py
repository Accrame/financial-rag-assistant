# ChromaDB vector store wrapper

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager:
    """Wraps ChromaDB for doc storage, retrieval, and management."""

    DEFAULT_COLLECTION = "financial_documents"

    def __init__(self, embeddings, persist_directory=None, collection_name=DEFAULT_COLLECTION):
        self.embeddings = embeddings
        self.persist_directory = persist_directory or "./chroma_db"
        self.collection_name = collection_name

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize vector store
        self._vectorstore = self._create_vectorstore()

    def _create_vectorstore(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents, batch_size=100):
        """Add docs in batches, returns list of IDs."""
        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = self._vectorstore.add_documents(batch)
            all_ids.extend(ids)

        return all_ids

    def similarity_search(self, query, k=5, filter=None, score_threshold=None):
        """Search with relevance scores."""
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter,
        )

        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        return results

    def similarity_search_simple(self, query, k=5, filter=None):
        """Search without returning scores."""
        return self._vectorstore.similarity_search(
            query,
            k=k,
            filter=filter,
        )

    def mmr_search(self, query, k=5, fetch_k=20, lambda_mult=0.5, filter=None):
        """MMR search - balances relevance and diversity."""
        return self._vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def delete_documents(self, doc_ids):
        self._vectorstore.delete(ids=doc_ids)

    def delete_by_metadata(self, filter):
        """Delete all docs matching the given metadata filter."""
        # this is kind of hacky but works - chroma doesn't have a direct delete-by-filter
        results = self._vectorstore.get(where=filter)
        if results and results.get("ids"):
            self._vectorstore.delete(ids=results["ids"])

    def get_document_count(self):
        collection = self._vectorstore._collection
        return collection.count()

    def list_sources(self):
        collection = self._vectorstore._collection
        results = collection.get(include=["metadatas"])

        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])

        return sorted(sources)

    def get_retriever(self, search_type="mmr", k=5, **kwargs):
        """Get a LangChain retriever for use in RAG chains."""
        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs},
        )

    def clear(self):
        """Nuke everything in the collection."""
        # Delete and recreate collection
        self._vectorstore._client.delete_collection(self.collection_name)
        self._vectorstore = self._create_vectorstore()


def create_vectorstore(
    embeddings,
    documents=None,
    persist_directory="./chroma_db",
    collection_name="financial_documents",
):
    """Create a vector store, optionally pre-loading documents."""
    manager = VectorStoreManager(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if documents:
        manager.add_documents(documents)

    return manager


if __name__ == "__main__":
    from embeddings import get_embeddings

    print("Testing vector store...")

    # Create embeddings
    embeddings = get_embeddings(provider="local")

    # Create vector store
    store = VectorStoreManager(
        embeddings=embeddings,
        persist_directory="./test_chroma_db",
        collection_name="test_collection",
    )

    # Add test documents
    test_docs = [
        Document(
            page_content="Revenue increased by 15% year-over-year to $10 billion.",
            metadata={"source": "10k.pdf", "page": 1},
        ),
        Document(
            page_content="The company faces significant cybersecurity risks.",
            metadata={"source": "10k.pdf", "page": 5},
        ),
        Document(
            page_content="Net income was $2.5 billion, up from $2.0 billion last year.",
            metadata={"source": "10k.pdf", "page": 3},
        ),
    ]

    store.add_documents(test_docs)
    print(f"Added {len(test_docs)} documents")
    print(f"Total documents: {store.get_document_count()}")

    # Test search
    results = store.similarity_search_simple("What was the revenue?", k=2)
    print("\nSearch results for 'What was the revenue?':")
    for doc in results:
        print(f"  - {doc.page_content[:50]}...")

    # Cleanup
    store.clear()
    print("\nTest complete!")
