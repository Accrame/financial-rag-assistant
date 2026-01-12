# Config for the RAG app - loads from .env via pydantic-settings

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Settings
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # Embedding Settings
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str | None = Field(default=None, alias="EMBEDDING_MODEL")

    # Vector Store Settings
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="financial_documents", alias="COLLECTION_NAME")

    # Chunking Settings
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # Retrieval Settings
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")
    search_type: str = Field(default="mmr", alias="SEARCH_TYPE")

    # API Settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Streamlit Settings
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    @property
    def has_openai(self):
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self):
        return bool(self.anthropic_api_key)

    def get_default_llm_model(self):
        """Return the model name, falling back to defaults per provider."""
        if self.llm_model:
            return self.llm_model
        # TODO: should probably make these configurable somewhere central
        if self.llm_provider == "openai":
            return "gpt-4o-mini"
        elif self.llm_provider == "anthropic":
            return "claude-3-haiku-20240307"
        else:
            return "gpt-4o-mini"

    def get_default_embedding_model(self):
        if self.embedding_model:
            return self.embedding_model
        if self.embedding_provider == "openai":
            return "text-embedding-3-small"
        else:
            return "all-MiniLM-L6-v2"


# Global settings instance
settings = Settings()


def get_settings():
    return settings


# Example .env file content
ENV_EXAMPLE = """# Financial RAG Assistant Configuration

# LLM Provider (openai or anthropic)
LLM_PROVIDER=openai

# API Keys (at least one required for LLM features)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Embedding Provider (openai or local)
# Use "local" for free, offline embeddings via sentence-transformers
EMBEDDING_PROVIDER=local

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=financial_documents

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_K=5
SEARCH_TYPE=mmr
"""


def create_env_example():
    with open(".env.example", "w") as f:
        f.write(ENV_EXAMPLE)


if __name__ == "__main__":
    # Print current settings
    s = get_settings()
    print("Current Settings:")
    print(f"  LLM Provider: {s.llm_provider}")
    print(f"  LLM Model: {s.get_default_llm_model()}")
    print(f"  Embedding Provider: {s.embedding_provider}")
    print(f"  Embedding Model: {s.get_default_embedding_model()}")
    print(f"  OpenAI configured: {s.has_openai}")
    print(f"  Anthropic configured: {s.has_anthropic}")

    # Create example .env
    create_env_example()
    print("\nCreated .env.example file")
