"""Separate semantic stores for agent experiences and document knowledge."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    """Load the local embedding model once per Python process."""
    return SentenceTransformer(MODEL_NAME)


def _safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only metadata types supported by ChromaDB."""
    return {
        key: value
        for key, value in (metadata or {}).items()
        if isinstance(value, (str, int, float, bool))
    }


class EpisodicMemory:
    """Persist and retrieve past task experiences and reflections."""

    def __init__(
        self,
        persist_directory: str = "workspace/memory_db",
        collection_name: str = "agent_episodes",
        embedding_model: Optional[Any] = None,
    ) -> None:
        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embedding_model = embedding_model or _load_embedding_model()
        self.client = chromadb.PersistentClient(path=str(path.resolve()))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Agent experience traces and reflections",
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """Save a successful trace, reflection, or other experience."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Episode text cannot be empty")

        identifier = episode_id or str(uuid4())
        safe_metadata = _safe_metadata(metadata) or {"type": "episode"}
        embedding = self.embedding_model.encode(text).tolist()

        self.collection.upsert(
            ids=[identifier],
            documents=[text],
            metadatas=[safe_metadata],
            embeddings=[embedding],
        )
        return identifier

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find semantically similar experiences using cosine distance."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")
        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        total = self.collection.count()
        if total == 0:
            return []

        embedding = self.embedding_model.encode(query).tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
        )

        episodes: List[Dict[str, Any]] = []
        for index, document in enumerate(result["documents"][0]):
            episodes.append(
                {
                    "id": result["ids"][0][index],
                    "text": document,
                    "metadata": (
                        (result.get("metadatas") or [[{}]])[0][index] or {}
                    ),
                    "distance": (
                        result["distances"][0][index]
                        if result.get("distances")
                        and result["distances"][0]
                        else None
                    ),
                }
            )
        return episodes

    def count(self) -> int:
        """Return the number of stored experiences."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all experiences from this collection only."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Agent experience traces and reflections",
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )


class KnowledgeStore:
    """Ingest documents as overlapping token chunks and retrieve by meaning."""

    def __init__(
        self,
        persist_directory: str = "workspace/memory_db",
        collection_name: str = "knowledge_documents",
        chunk_tokens: int = 300,
        overlap_tokens: int = 50,
        embedding_model: Optional[Any] = None,
    ) -> None:
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be at least 1")
        if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
            raise ValueError(
                "overlap_tokens must be non-negative and smaller than chunk_tokens"
            )

        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_model = embedding_model or _load_embedding_model()
        self.client = chromadb.PersistentClient(path=str(path.resolve()))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Course documents and technical knowledge",
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )

    def _tokenize(self, text: str) -> List[Any]:
        """Tokenize text using the embedding model's tokenizer when available."""
        tokenizer = getattr(self.embedding_model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.encode(text, add_special_tokens=False)
        return text.split()

    def _decode(self, tokens: List[Any]) -> str:
        """Convert a token chunk back into text."""
        tokenizer = getattr(self.embedding_model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(token) for token in tokens)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with the configured token overlap."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document text cannot be empty")

        tokens = self._tokenize(text)
        chunks: List[str] = []
        start = 0
        stride = self.chunk_tokens - self.overlap_tokens

        while start < len(tokens):
            end = min(start + self.chunk_tokens, len(tokens))
            chunk = self._decode(tokens[start:end]).strip()

            if chunk:
                chunks.append(chunk)

            if end == len(tokens):
                break

            start += stride

        return chunks

    def ingest_text(
        self,
        text: str,
        source: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Chunk, embed, and persist a document's text."""
        chunks = self.chunk_text(text)
        if not chunks:
            return []

        ids = [str(uuid4()) for _ in chunks]
        base_metadata = _safe_metadata(metadata)

        metadatas = [
            {**base_metadata, "source": source, "chunk_index": index}
            for index in range(len(chunks))
        ]
        embeddings = [
            self.embedding_model.encode(chunk).tolist()
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return ids

    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Ingest a UTF-8 text, Markdown, RST, or PDF file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()

        if suffix in {".txt", ".md", ".rst"}:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ImportError(
                    "PDF ingestion requires pypdf; install it with pip install pypdf"
                ) from exc

            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            raise ValueError(
                "Supported document types: .txt, .md, .rst, .pdf"
            )

        return self.ingest_text(text, source=path.name, metadata=metadata)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most semantically similar knowledge chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")
        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        total = self.collection.count()
        if total == 0:
            return []

        embedding = self.embedding_model.encode(query).tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
        )

        return [
            {
                "id": result["ids"][0][index],
                "text": document,
                "metadata": (
                    (result.get("metadatas") or [[{}]])[0][index] or {}
                ),
                "distance": (
                    result["distances"][0][index]
                    if result.get("distances")
                    and result["distances"][0]
                    else None
                ),
            }
            for index, document in enumerate(result["documents"][0])
        ]

    def count(self) -> int:
        """Return the number of stored document chunks."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete knowledge chunks without touching experience memory."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Course documents and technical knowledge",
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )
