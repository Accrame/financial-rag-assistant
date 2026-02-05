# Embedding utilities - supports OpenAI and local sentence-transformers

import os

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Local embeddings via sentence-transformers. No API key needed."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        # TODO: add caching for embeddings

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    @property
    def embedding_dimension(self):
        return self._embedding_dim


def get_embeddings(provider="openai", model=None, api_key=None):
    """Get embeddings model for the given provider."""
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model or "text-embedding-3-small",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "local":
        return LocalEmbeddings(model_name=model or "all-MiniLM-L6-v2")

    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")

    else:
        raise ValueError(f"Unknown provider: {provider}")


def estimate_token_count(text, model="cl100k_base"):
    """Rough token count estimate."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding(model)
        return len(encoder.encode(text))
    except ImportError:
        # fallback: ~4 chars per token, not super accurate but good enough
        return len(text) // 4


if __name__ == "__main__":
    # Test local embeddings
    print("Testing local embeddings...")

    try:
        embeddings = get_embeddings(provider="local")
        test_texts = [
            "Revenue increased by 15% year-over-year",
            "The company faces significant market risk",
        ]

        vectors = embeddings.embed_documents(test_texts)
        print(f"Embedding dimension: {len(vectors[0])}")
        print(f"Embedded {len(test_texts)} documents")

        query_vector = embeddings.embed_query("What is the revenue growth?")
        print(f"Query vector dimension: {len(query_vector)}")

    except ImportError as e:
        print(f"Local embeddings not available: {e}")
        print("Install with: pip install sentence-transformers")
