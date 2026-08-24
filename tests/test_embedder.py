"""Tests for meeting_bot.embedder with a mocked model (no MLX/Metal).

The mlx_embedding_models import happens inside ``Embedder.__init__``; we
monkeypatch sys.modules so the suite runs without the package installed.
"""

import sys
import types

import numpy as np
import pytest

import meeting_bot.embedder as embedder_mod
from meeting_bot.embedder import Embedder


class _FakeModel:
    """Mimics mlx_embedding_models EmbeddingModel.encode."""

    def __init__(self, d: int = 8):
        self.d = d
        self.calls: list[list[str]] = []

    def encode(self, sentences, batch_size=64, show_progress=True, **kwargs):
        self.calls.append(list(sentences))
        vecs = np.zeros((len(sentences), self.d), dtype=np.float32)
        for i, s in enumerate(sentences):
            # deterministic pseudo-vector from the text content; +1 keeps
            # every row strictly non-zero so normalization is meaningful
            seed = sum(ord(ch) for ch in s) or 1
            for j in range(self.d):
                vecs[i, j] = float((seed * (j + 1)) % 5 + 1)
        return vecs


def _make_embedder(monkeypatch, d: int = 8) -> tuple[Embedder, _FakeModel]:
    fake = _FakeModel(d=d)

    mod = types.ModuleType("mlx_embedding_models")
    emb_mod = types.ModuleType("mlx_embedding_models.embedding")
    emb_mod.EmbeddingModel = types.SimpleNamespace(
        from_pretrained=lambda name: fake
    )
    mod.embedding = emb_mod
    monkeypatch.setitem(sys.modules, "mlx_embedding_models", mod)
    monkeypatch.setitem(sys.modules, "mlx_embedding_models.embedding", emb_mod)

    return Embedder(model="BAAI/bge-m3"), fake


class TestEmbedderInit:
    def test_missing_package_raises_clear_importerror(self, monkeypatch):
        # Make the import fail as if the package were absent.
        monkeypatch.setitem(
            sys.modules,
            "mlx_embedding_models",
            None,  # `import mlx_embedding_models` raises ImportError
        )
        with pytest.raises(ImportError, match="mlx-embedding-models"):
            Embedder(model="BAAI/bge-m3")

    def test_lazy_module_importable_without_package(self):
        # Module scope imports only numpy — importing it must not require
        # the MLX stack.
        assert embedder_mod.__name__ == "meeting_bot.embedder"


class TestEmbedTexts:
    def test_shapes_and_dtype(self, monkeypatch):
        emb, _ = _make_embedder(monkeypatch, d=16)
        out = emb.embed_texts(["หนึ่ง", "สอง", "สาม"])
        assert out.shape == (3, 16)
        assert out.dtype == np.float32

    def test_rows_l2_normalized(self, monkeypatch):
        emb, _ = _make_embedder(monkeypatch)
        out = emb.embed_texts(["a", "b", "c"])
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_input_returns_zero_by_dim(self, monkeypatch):
        emb, _ = _make_embedder(monkeypatch, d=8)
        out = emb.embed_texts([])
        assert out.shape == (0, 8)

    def test_dim_property_resolves_from_probe(self, monkeypatch):
        emb, fake = _make_embedder(monkeypatch, d=11)
        assert emb.dim == 11
        assert len(fake.calls) == 1  # probe call

    def test_identical_texts_give_identical_vectors(self, monkeypatch):
        emb, _ = _make_embedder(monkeypatch)
        out = emb.embed_texts(["ซ้ำ", "ซ้ำ"])
        assert np.allclose(out[0], out[1])


class TestEmbedQuery:
    def test_returns_single_vector(self, monkeypatch):
        emb, _ = _make_embedder(monkeypatch)
        q = emb.embed_query("การตัดสินใจ")
        assert q.shape == (emb.dim,)
        assert pytest.approx(float(np.linalg.norm(q)), abs=1e-5) == 1.0

    def test_query_matches_texts_pipeline(self, monkeypatch):
        emb, _ = _make_embedder(monkeypatch)
        texts = emb.embed_texts(["การตัดสินใจ"])
        q = emb.embed_query("การตัดสินใจ")
        assert np.allclose(q, texts[0])
