# Chunking strategies for financial documents

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


class FinancialDocumentChunker:
    """Chunks documents for retrieval. Supports recursive and token-based splitting."""

    # these separators are tuned for SEC filings but work ok for general docs too
    SECTION_SEPARATORS = [
        "\n\nItem ",  # SEC filing sections
        "\n\nPART ",  # 10-K parts
        "\n## ",  # Markdown headers
        "\n\n\n",  # Triple newlines
        "\n\n",  # Double newlines
        "\n",  # Single newlines
        ". ",  # Sentences
        " ",  # Words
    ]

    def __init__(self, chunk_size=1000, chunk_overlap=200, strategy="recursive"):
        # TODO: should probably validate chunk_size > chunk_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self._splitter = self._create_splitter()

    def _create_splitter(self):
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.SECTION_SEPARATORS,
                length_function=len,
            )
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",  # GPT-4 encoding
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def chunk_documents(self, documents, add_chunk_metadata=True):
        """Split documents into chunks, preserving metadata."""
        chunked_docs = []

        for doc in documents:
            chunks = self._splitter.split_text(doc.page_content)
            # print(f"DEBUG: {len(chunks)} chunks generated")

            for i, chunk_text in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()

                if add_chunk_metadata:
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["chunk_size"] = len(chunk_text)

                chunked_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
                chunked_docs.append(chunked_doc)

        return chunked_docs

    def chunk_text(self, text):
        """Chunk a single string, returns list of strings."""
        return self._splitter.split_text(text)


class SemanticChunker:
    """Tries to split on SEC filing section boundaries when possible."""

    # SEC filing section patterns
    SEC_SECTIONS = {
        "risk_factors": r"Item\s+1A[\.\s]+Risk\s+Factors",
        "business": r"Item\s+1[\.\s]+Business",
        "mda": r"Item\s+7[\.\s]+Management.s\s+Discussion",
        "financials": r"Item\s+8[\.\s]+Financial\s+Statements",
        "controls": r"Item\s+9A[\.\s]+Controls",
    }

    def __init__(self, max_chunk_size=2000, min_chunk_size=200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._fallback_chunker = FinancialDocumentChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
        )

    def chunk_by_sections(self, documents):
        """Chunk by SEC sections, falls back to recursive for unstructured docs."""
        import re

        chunked_docs = []

        for doc in documents:
            text = doc.page_content

            # Try to find SEC filing sections
            sections_found = []
            for section_name, pattern in self.SEC_SECTIONS.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    sections_found.append((match.start(), section_name))

            if sections_found:
                # Sort by position
                sections_found.sort(key=lambda x: x[0])

                # Extract section content
                for i, (start, section_name) in enumerate(sections_found):
                    if i + 1 < len(sections_found):
                        end = sections_found[i + 1][0]
                    else:
                        end = len(text)

                    section_text = text[start:end].strip()

                    # If section is too long, chunk it further
                    if len(section_text) > self.max_chunk_size:
                        sub_chunks = self._fallback_chunker.chunk_text(section_text)
                        for j, chunk in enumerate(sub_chunks):
                            chunk_metadata = doc.metadata.copy()
                            chunk_metadata["section"] = section_name
                            chunk_metadata["sub_chunk"] = j
                            chunked_docs.append(
                                Document(
                                    page_content=chunk,
                                    metadata=chunk_metadata,
                                )
                            )
                    else:
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata["section"] = section_name
                        chunked_docs.append(
                            Document(
                                page_content=section_text,
                                metadata=chunk_metadata,
                            )
                        )
            else:
                # Fall back to regular chunking
                chunked_docs.extend(self._fallback_chunker.chunk_documents([doc]))

        return chunked_docs


def create_chunker(strategy="recursive", chunk_size=1000, chunk_overlap=200):
    """Convenience factory for creating a chunker."""
    return FinancialDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )


if __name__ == "__main__":
    # Quick test
    chunker = FinancialDocumentChunker()

    sample_text = """
    Item 1A. Risk Factors

    The following discussion of risk factors contains forward-looking statements.
    These risk factors may be important to understanding other statements in this Form 10-K.

    Market Risk
    Our business is subject to various market risks including interest rate fluctuations,
    foreign currency exchange rate changes, and equity price volatility.

    Operational Risk
    We face operational risks related to our technology infrastructure, cybersecurity,
    and business continuity planning.
    """

    chunks = chunker.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks from sample text")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: {len(chunk)} chars")
