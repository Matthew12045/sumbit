"""Local MLX sentence embeddings for the RAG pipeline (lazy import).

``mlx_embedding_models`` is imported inside :class:`Embedder.__init__`
(mirroring ``summarizer.py``'s lazy anthropic import), so this module stays
importable — and unit-testable with a mocked model — without the MLX stack.

Model default is ``BAAI/bge-m3``, which is symmetric: no query/passage
prefixes are needed, so ``embed_query`` and ``embed_texts`` share one code
path. All outputs are L2-normalized float32 rows so cosine similarity in
:class:`meeting_bot.rag_store.VectorIndex` is a plain dot product.

Compatibility: ``mlx_embedding_models`` 0.0.11 tokenizes via
``tokenizer.batch_encode_plus``, which was removed in ``transformers`` 5.x.
When that method is absent we monkeypatch the model instance with an
equivalent ``_tokenize`` built on the modern ``tokenizer(...)`` API — same
arguments, same jagged-array output shape.
"""

from __future__ import annotations

import types

import numpy as np

__all__ = ["Embedder"]

_EMBED_BATCH = 32  # bound peak memory on long transcripts


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _tokenize_compat(self, sentences, min_length=None):  # pragma: no cover
    """Drop-in replacement for ``EmbeddingModel._tokenize`` under
    ``transformers>=5``. Only the no-``min_length`` path is needed by
    :meth:`Embedder.embed_texts` / :meth:`Embedder.embed_query`."""
    if min_length is not None:
        raise NotImplementedError(
            "min_length tokenization is not supported under transformers>=5"
        )
    import awkward as ak

    encoded = self.tokenizer(
        list(sentences),
        padding=False,
        truncation=True,
        max_length=self.max_length,
    )
    return {key: ak.Array(value) for key, value in encoded.items()}


def _apply_transformers5_shim(model) -> None:
    """Patch ``model._tokenize`` only when the legacy API is missing."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and not hasattr(tokenizer, "batch_encode_plus"):
        model._tokenize = types.MethodType(_tokenize_compat, model)


class Embedder:
    """Sentence embeddings via ``mlx-embedding-models`` (Apple Silicon).

    Wraps ``mlx_embedding_models.embedding.EmbeddingModel``, whose
    ``encode(sentences, batch_size)`` returns an ``(n, d)`` numpy array.
    """

    def __init__(self, model: str = "BAAI/bge-m3"):
        try:
            from mlx_embedding_models.embedding import EmbeddingModel
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "mlx-embedding-models is required for RAG embeddings "
                "(pip install mlx-embedding-models)"
            ) from exc
        self.model_name = model
        self._model = EmbeddingModel.from_pretrained(model)
        _apply_transformers5_shim(self._model)
        self._dim: int | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=_EMBED_BATCH, show_progress=False),
            dtype=np.float32,
        )

    @property
    def dim(self) -> int:
        """Embedding width (resolved lazily; bge-m3 = 1024)."""
        if self._dim is None:
            self._dim = int(self._encode([""]).shape[-1])
        return self._dim

    def embed_texts(self, texts) -> np.ndarray:
        """Embed a list of passages into ``(n, d)`` float32, L2-normalized.

        Returns a ``(0, d)`` array for an empty input.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(self._encode(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query string into a single ``(d,)`` float32 vector."""
        return self.embed_texts([text])[0]
