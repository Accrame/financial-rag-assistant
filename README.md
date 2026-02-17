# Financial RAG Assistant

RAG system for querying financial documents (10-K filings, earnings reports) using LangChain, ChromaDB, and OpenAI/Claude. Upload a PDF, ask questions in natural language, get answers with citations back to the source pages.

## The idea

Instead of ctrl-F'ing through a 200-page 10-K filing, you can just ask "what were the main risk factors?" and get an answer grounded in the actual document. The system chunks the PDF, embeds the chunks into ChromaDB, retrieves the most relevant ones for your question, and feeds them to an LLM to generate an answer.

The Streamlit demo works but I haven't wired up a real API yet -- it's mostly the chat interface for now.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy .env.example to .env and add your API key
cp .env.example .env

# Run the chat interface
streamlit run streamlit_app/app.py
```

You need an OpenAI API key for embeddings + generation, or you can use local embeddings (`sentence-transformers`) and just pay for the LLM calls.

There's a sample Apple 10-K (FY2024) in `data/sample_docs/` so you can test immediately without finding your own PDF.

## Project structure

```
src/
├── document_processing/   # PDF parsing, text chunking
├── vectorstore/           # ChromaDB setup, embedding logic
├── rag_chain/             # LangChain retrieval + generation
└── api/                   # FastAPI endpoints (barely started)
```

## Things I struggled with

- **Chunk size is a guessing game.** Too small and you lose context, too big and retrieval gets noisy. I settled on 1000 tokens with 200 overlap after a lot of trial and error, but honestly I'm not sure it's optimal
- **ChromaDB version issues.** The API changed between 0.3.x and 0.4.x and half the tutorials online use the old one. Spent way too long debugging import errors that turned out to be version mismatches
- **Citations are harder than they sound.** Getting the LLM to actually reference specific pages instead of making things up required a lot of prompt tweaking. It still occasionally hallucinates a page number
- **Picking an embedding model.** OpenAI's `text-embedding-3-small` works fine but costs money. The local `all-MiniLM-L6-v2` is free but noticeably worse on financial jargon. Never found the perfect middle ground
- **API key management across providers.** Supporting both OpenAI and Anthropic means juggling two different API key setups, client configs, and error handling. It's messy

## What I'd do differently

- Try `bge-large` or another finance-aware embedding model instead of the generic ones
- Add hybrid retrieval (BM25 + semantic) -- pure vector search misses exact terms like ticker symbols and specific dollar amounts
- Implement a re-ranking step after retrieval. Right now the top-k chunks go straight to the LLM, and sometimes the most relevant one isn't ranked first
- Build actual evaluation -- I have no metrics on retrieval quality, just vibes. Should have used RAGAS or something similar from the start
- The table extraction is pretty bad. `pdfplumber` gets the text but financial tables come out garbled more often than not

## License

MIT

