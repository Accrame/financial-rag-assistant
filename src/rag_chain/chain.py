# RAG chain - ties retrieval + LLM together for financial doc Q&A

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Prompts for financial document Q&A
SYSTEM_PROMPT = """You are a financial analyst assistant specialized in analyzing corporate documents like 10-K filings, earnings reports, and investment prospectuses.

Your task is to answer questions based ONLY on the provided context. Follow these guidelines:

1. Base your answers strictly on the provided context
2. If the context doesn't contain enough information, say so clearly
3. When citing numbers or facts, reference the source document and page
4. For financial metrics, be precise with numbers and units
5. If asked about trends, provide year-over-year comparisons when available
6. Highlight any risk factors or uncertainties mentioned in the context

Remember: Never make up information. If you're unsure, say "Based on the provided documents, I cannot find information about..."
"""

QA_PROMPT_TEMPLATE = """Context from financial documents:
{context}

---

Question: {question}

Provide a comprehensive answer based on the context above. Include specific numbers, quotes, and page references where relevant."""


class FinancialRAGChain:
    """RAG chain for financial document Q&A with source citations."""

    def __init__(self, retriever, llm_provider="openai", model_name=None,
                 temperature=0.0, api_key=None):
        self.retriever = retriever
        self.llm = self._create_llm(llm_provider, model_name, temperature, api_key)
        self.chain = self._create_chain()
        self.chat_history = []

    def _create_llm(self, provider, model_name, temperature, api_key):
        # TODO: add proper error handling for rate limits
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=temperature,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name or "claude-3-haiku-20240307",
                temperature=temperature,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _create_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", QA_PROMPT_TEMPLATE),
            ]
        )

        def format_docs(docs):
            """Format docs for the prompt context window."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page", "N/A")
                formatted.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question, return_sources=True):
        """Run a question through the RAG pipeline."""
        answer = self.chain.invoke(question)
        result = {"answer": answer}

        if return_sources:
            # not sure if this is the best way to do this - it runs retrieval twice
            docs = self.retriever.invoke(question)
            sources = []
            for doc in docs:
                sources.append(
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("filename", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                    }
                )
            result["sources"] = sources

        # Update chat history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return result

    async def aquery(self, question, return_sources=True):
        """Async version of query."""
        answer = await self.chain.ainvoke(question)

        result = {"answer": answer}

        if return_sources:
            docs = await self.retriever.ainvoke(question)
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                }
                for doc in docs
            ]
            result["sources"] = sources

        return result

    def stream(self, question):
        """Stream response tokens."""
        for chunk in self.chain.stream(question):
            yield chunk

    def clear_history(self):
        self.chat_history = []


class ConversationalRAGChain(FinancialRAGChain):
    """RAG chain with conversation memory for follow-up questions."""

    CONTEXTUALIZE_PROMPT = """Given the chat history and the latest question, 
reformulate the question to be standalone (understandable without the chat history).
Do NOT answer the question, just reformulate it if needed.

Chat History:
{chat_history}

Latest Question: {question}

Standalone Question:"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contextualize_chain = self._create_contextualize_chain()

    def _create_contextualize_chain(self):
        prompt = ChatPromptTemplate.from_template(self.CONTEXTUALIZE_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def _format_chat_history(self):
        if not self.chat_history:
            return "No previous conversation."

        formatted = []
        for msg in self.chat_history[-6:]:  # Last 3 exchanges
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def query(self, question, return_sources=True):
        # Contextualize if there's history
        if self.chat_history:
            standalone_question = self.contextualize_chain.invoke(
                {
                    "chat_history": self._format_chat_history(),
                    "question": question,
                }
            )
        else:
            standalone_question = question

        # Query with standalone question
        return super().query(standalone_question, return_sources)


def create_rag_chain(retriever, provider="openai", model=None, conversational=False):
    chain_class = ConversationalRAGChain if conversational else FinancialRAGChain

    return chain_class(
        retriever=retriever,
        llm_provider=provider,
        model_name=model,
    )


if __name__ == "__main__":
    print("RAG Chain module loaded successfully")
    print("Usage:")
    print("  chain = create_rag_chain(retriever, provider='openai')")
    print("  result = chain.query('What were the total revenues?')")
