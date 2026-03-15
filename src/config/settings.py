"""Configuration settings for the MenoGuide MRAG system.

Uses Pydantic BaseSettings for environment variable loading and validation.
All system parameters from the IEEE ICCITX 2026 paper are centralized here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        gemini_api_key: Google Gemini API key for LLM and captioning.
        hf_api_token: HuggingFace API token for model downloads.
        embedding_model_name: Dense embedding model identifier.
        llm_model_name: LLM model identifier for generation.
        llm_temperature: Temperature for LLM generation.
        reranker_model_name: Cross-encoder model for reranking.
        chunk_size: Token count per text chunk (paper: 512).
        chunk_overlap: Token overlap between chunks (paper: 64).
        hybrid_alpha: Dense/sparse fusion weight (paper Eq.1: 0.7).
        bm25_k1: BM25 term frequency saturation (paper Eq.2: 1.5).
        bm25_b: BM25 document length normalization (paper Eq.2: 0.75).
        retrieval_top_k: Number of documents to retrieve per query.
        reranking_enabled: Whether to apply cross-encoder reranking.
        repacking_enabled: Whether to apply semantic repacking.
        repacking_method: Repacking strategy ('similarity' or 'token_limit').
        max_tokens_per_group: Max tokens per repacked group.
        collection_name: ChromaDB collection name.
        gemini_rate_limit: Max Gemini API requests per minute.
        log_level: Logging verbosity level.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Keys ---
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    hf_api_token: str = Field(default="", description="HuggingFace API token")

    # --- Model Identifiers ---
    embedding_model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Dense embedding model (Table I: 92.3% precision)",
    )
    llm_model_name: str = Field(
        default="gemini-2.0-flash",
        description="LLM for generation and captioning",
    )
    llm_temperature: float = Field(
        default=0.3,
        description="LLM generation temperature",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder for reranking (Table IV: NDCG=0.847)",
    )

    # --- Chunking Parameters (Section III-B) ---
    chunk_size: int = Field(
        default=512,
        description="Token count per text chunk",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Token overlap between adjacent chunks",
    )

    # --- Hybrid Retrieval (Eq. 1, Table II) ---
    hybrid_alpha: float = Field(
        default=0.7,
        description="Dense/sparse fusion weight (alpha)",
    )
    bm25_k1: float = Field(
        default=1.5,
        description="BM25 term frequency saturation parameter",
    )
    bm25_b: float = Field(
        default=0.75,
        description="BM25 document length normalization parameter",
    )
    retrieval_top_k: int = Field(
        default=10,
        description="Number of documents to retrieve per query",
    )
    reranking_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking",
    )
    repacking_enabled: bool = Field(
        default=True,
        description="Enable semantic repacking for diversity",
    )
    repacking_method: str = Field(
        default="similarity",
        description="Repacking strategy: 'similarity' or 'token_limit'",
    )
    max_tokens_per_group: int = Field(
        default=1500,
        description="Max tokens per repacked document group",
    )

    # --- Storage ---
    collection_name: str = Field(
        default="MenoGuide",
        description="ChromaDB collection name",
    )

    # --- Rate Limiting ---
    gemini_rate_limit: int = Field(
        default=14,
        description="Max Gemini API requests per minute",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    @property
    def base_dir(self) -> Path:
        """Project root directory."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Data directory containing corpus and indices."""
        return self.base_dir / "data"

    @property
    def raw_pdf_dir(self) -> Path:
        """Directory containing raw downloaded PDFs."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for intermediate processed outputs."""
        return self.data_dir / "processed"

    @property
    def indices_dir(self) -> Path:
        """Directory for vector store and BM25 indices."""
        return self.data_dir / "indices"

    @property
    def chroma_db_dir(self) -> Path:
        """ChromaDB persistence directory."""
        return self.indices_dir / "chroma_db"

    @property
    def bm25_index_path(self) -> Path:
        """BM25 serialized corpus path."""
        return self.indices_dir / "bm25_corpus.pkl"

    @property
    def manifest_path(self) -> Path:
        """Ingestion manifest for idempotent processing."""
        return self.processed_dir / "manifest.json"

    @property
    def assets_dir(self) -> Path:
        """Static assets directory for the Streamlit app."""
        return self.base_dir / "src" / "app" / "assets"

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for directory in [
            self.raw_pdf_dir,
            self.processed_dir,
            self.indices_dir,
            self.chroma_db_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached singleton Settings instance.

    Returns:
        The application Settings loaded from environment.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
