# MenoGuide — Multimodal RAG for Menopause Health Information

A peer-reviewed Multimodal Retrieval-Augmented Generation (MRAG) framework that provides
evidence-based menopause health information grounded in curated PLOS ONE articles.

## Architecture

```
User Query → LLM Query Classifier (96% accuracy)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Direct Response         RAG Pipeline
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              Dense Search          BM25 Search
              (BGE-large)           (k1=1.5, b=0.75)
                    │                     │
                    └──────┬──────────────┘
                           ▼
                   Score Fusion (α=0.7)
                           ▼
                   Cross-Encoder Reranking
                   (ms-marco-MiniLM-L-6-v2)
                           ▼
                   Semantic Repacking
                   (Diversity: 9.2/10)
                           ▼
                   Gemini 2.0 Flash Generation
                           ▼
                   Grounded Response + Citations
```

## Published Metrics

| Metric | Score | 95% CI |
|--------|-------|--------|
| Faithfulness | 88% | ±3.1% |
| Answer Relevance | 90% | ±2.8% |
| Context Precision | 85% | ±3.6% |
| SUS Score | 82/100 | — |
| Routing Accuracy | 96% | — |

## Tech Stack

- **LLM:** Gemini 2.0 Flash
- **Embeddings:** BAAI/bge-large-en-v1.5 (d=1024)
- **Vector Store:** ChromaDB (local, privacy-focused)
- **Sparse Retrieval:** BM25 (rank-bm25)
- **Reranking:** ms-marco-MiniLM-L-6-v2
- **Document Processing:** Unstructured library (hi-res strategy)
- **Frontend:** Streamlit (4 tabs)
- **PDF Reports:** FPDF2
- **Evaluation:** RAGAS framework
- **Dependency Management:** Poetry

## Setup

### Prerequisites

- Python 3.11+
- Poetry
- Tesseract OCR (for PDF processing)
- Poppler (for PDF-to-image conversion)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd menopause-multimodal-rag

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your API keys (GEMINI_API_KEY, HF_API_TOKEN)
```

### Running the Application

```bash
# Run the Streamlit app
poetry run streamlit run src/app/main.py

# Run document ingestion pipeline
poetry run python -m scripts.run_ingestion

# Run RAGAS evaluation
poetry run python -m scripts.run_evaluation
```

## Project Structure

```
mrag-menopause/
├── pyproject.toml              # Poetry dependencies
├── .env.example                # Environment template
├── src/
│   ├── config/settings.py      # Pydantic BaseSettings
│   ├── ingestion/              # PDF scraping, parsing, chunking, captioning
│   ├── retrieval/              # Embeddings, vectorstore, BM25, hybrid, reranking, repacking
│   ├── generation/             # Query classifier, prompts, RAG generator
│   ├── evaluation/             # RAGAS evaluation pipeline
│   ├── app/                    # Streamlit interface (4 tabs)
│   └── utils/                  # Logging, helpers
├── data/
│   ├── raw/                    # PLOS ONE PDFs
│   ├── processed/              # Parsed chunks, captions, manifest
│   └── indices/                # ChromaDB, BM25 pickle
├── tests/                      # pytest test suite
├── scripts/                    # CLI entry points
└── notebooks/                  # Exploration notebooks
```

## Application Tabs

1. **RAG Chatbot** — Conversational interface with hybrid retrieval and source citations
2. **Symptom Tracker** — 7-symptom severity tracking (0-10 scale)
3. **PDF Report Generator** — Clinical report with AI-grounded recommendations
4. **Educational Cards** — Curated fact cards from the PLOS ONE corpus

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8501
```

Ensure your `.env` file contains valid API keys before building.
The container mounts `data/` volumes for persistence and caches HuggingFace models.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.