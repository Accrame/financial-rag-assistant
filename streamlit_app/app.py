# Streamlit frontend for the Financial RAG Assistant

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f0f0;
    }
    .source-card {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .status-success { color: #2ca02c; }
    .status-warning { color: #ff7f0e; }
    .status-error { color: #d62728; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0


def get_embeddings():
    from src.vectorstore.embeddings import get_embeddings as get_emb

    # try openai first, fall back to local
    if os.getenv("OPENAI_API_KEY"):
        try:
            return get_emb(provider="openai")
        except Exception:
            pass
    return get_emb(provider="local")


def process_uploaded_files(uploaded_files):
    from src.document_processing.loader import FinancialDocumentLoader
    from src.document_processing.chunker import FinancialDocumentChunker
    from src.vectorstore.store import VectorStoreManager
    
    with st.spinner("Processing documents..."):
        loader = FinancialDocumentLoader()
        chunker = FinancialDocumentChunker(chunk_size=1000, chunk_overlap=200)
        
        all_documents = []
        
        for uploaded_file in uploaded_files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                # Load and chunk
                docs = loader.load(tmp_path)
                chunked_docs = chunker.chunk_documents(docs)
                
                # Update filename in metadata
                for doc in chunked_docs:
                    doc.metadata["filename"] = uploaded_file.name
                
                all_documents.extend(chunked_docs)
                st.success(f"Processed: {uploaded_file.name} ({len(chunked_docs)} chunks)")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            finally:
                # Cleanup temp file
                os.unlink(tmp_path)
        
        if all_documents:
            # Initialize or get vector store
            if st.session_state.vectorstore is None:
                embeddings = get_embeddings()
                st.session_state.vectorstore = VectorStoreManager(
                    embeddings=embeddings,
                    persist_directory="./chroma_db",
                )
            
            # Add documents
            st.session_state.vectorstore.add_documents(all_documents)
            st.session_state.documents_loaded = True
            st.session_state.document_count = st.session_state.vectorstore.get_document_count()
            
            st.success(f"Added {len(all_documents)} chunks to the knowledge base!")


def initialize_rag_chain():
    """Set up the RAG chain with whatever LLM provider is available."""
    from src.rag_chain.chain import ConversationalRAGChain
    
    if st.session_state.vectorstore is None:
        return None
    
    retriever = st.session_state.vectorstore.get_retriever(k=5)
    
    # Determine LLM provider
    if os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    else:
        st.warning("No API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        return None
    
    return ConversationalRAGChain(
        retriever=retriever,
        llm_provider=provider,
    )


def display_chat_history():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.write(content)
                
                # Show sources if available
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{source['source']}</strong> (Page {source['page']})<br>
                                {source['content']}
                            </div>
                            """, unsafe_allow_html=True)


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">📊 Financial RAG Assistant</p>', unsafe_allow_html=True)
    st.markdown("Upload financial documents and ask questions in natural language.")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Financial Documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Upload 10-K filings, earnings reports, or other financial documents",
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Status
        st.markdown("---")
        st.subheader("📊 Status")
        
        if st.session_state.documents_loaded:
            st.markdown(f'<p class="status-success">✓ {st.session_state.document_count} chunks indexed</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠ No documents loaded</p>', 
                       unsafe_allow_html=True)
        
        # API status
        if os.getenv("OPENAI_API_KEY"):
            st.markdown('<p class="status-success">✓ OpenAI API configured</p>', 
                       unsafe_allow_html=True)
        elif os.getenv("ANTHROPIC_API_KEY"):
            st.markdown('<p class="status-success">✓ Anthropic API configured</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">✗ No LLM API key found</p>', 
                       unsafe_allow_html=True)
            st.info("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        
        # Clear options
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_history()
            st.rerun()
        
        if st.button("Clear All Data"):
            if st.session_state.vectorstore:
                st.session_state.vectorstore.clear()
            st.session_state.messages = []
            st.session_state.documents_loaded = False
            st.session_state.document_count = 0
            st.session_state.rag_chain = None
            st.rerun()
    
    # Main chat area
    # TODO: add a way to select search_type (mmr vs similarity) from sidebar
    if not st.session_state.documents_loaded:
        st.info("👆 Upload financial documents using the sidebar to get started.")
        
        # Show example questions
        st.subheader("Example Questions You Can Ask:")
        examples = [
            "What were the total revenues for the fiscal year?",
            "Summarize the main risk factors mentioned in the 10-K",
            "How did net income compare to the previous year?",
            "What does management say about future growth prospects?",
            "What are the company's main business segments?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")
    
    else:
        # Initialize RAG chain if needed
        if st.session_state.rag_chain is None:
            st.session_state.rag_chain = initialize_rag_chain()
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            if st.session_state.rag_chain:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            result = st.session_state.rag_chain.query(prompt, return_sources=True)
                            
                            st.write(result["answer"])
                            
                            # Show sources
                            if result.get("sources"):
                                with st.expander("📚 Sources"):
                                    for source in result["sources"]:
                                        st.markdown(f"""
                                        <div class="source-card">
                                            <strong>{source['source']}</strong> (Page {source['page']})<br>
                                            {source['content']}
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result.get("sources", []),
                            })
                            
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
            else:
                st.warning("Please configure an API key to enable chat functionality.")


if __name__ == "__main__":
    main()
