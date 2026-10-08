# RAG Document Q&A System

A chatbot that answers questions over PDF documents with cited sources.

**Stack:** LangChain, OpenAI, ChromaDB, FastAPI, Streamlit, RAGAS

## Setup
1. `python -m venv venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI key
4. Put PDFs in `data/raw/`
5. `python scripts/check_setup.py`

## Status
- [x] Phase 1: Setup
- [ ] Phase 2: Ingestion
- [ ] Phase 3: Retrieval and generation
- [ ] Phase 4: Evaluation
- [ ] Phase 5: API and UI
- [ ] Phase 6: Deployment



## Ingestion
python scripts/ingest.py --strategy recursive --chunk-size 1000 --overlap 200 --reset
python scripts/ingest.py --strategy structure
python scripts/ingest.py --strategy semantic
python scripts/query_index.py "your question" --collection docs_recursive_1000_200