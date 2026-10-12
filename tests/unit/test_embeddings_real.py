"""T-301 acceptance with the REAL model (downloads ~470MB). Run: `uv run pytest -m embedding`."""

import math

import pytest

from duitku.memory.embeddings import SentenceTransformerEmbedder

pytestmark = pytest.mark.embedding

# (anchor, paraphrase, unrelated)
TRIPLES = [
    ("saya suka makan nasi goreng", "aku gemar menyantap nasi goreng", "harga saham naik hari ini"),
    (
        "tolong ingat aku ngopi tiap pagi",
        "setiap pagi saya biasa minum kopi",
        "kereta ke Bandung berangkat jam tujuh",
    ),
    (
        "gaji saya dibayar tanggal 25",
        "tiap tanggal 25 aku terima gajian",
        "kucingku suka tidur di sofa",
    ),
    (
        "Gojek masuk kategori transport",
        "ongkos ojek online dicatat sebagai transportasi",
        "resep rendang sapi yang empuk",
    ),
]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_real_model_dim_and_paraphrase_margin(capsys: pytest.CaptureFixture[str]) -> None:
    from duitku.config import get_settings

    s = get_settings()
    e = SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim)
    rows: list[tuple[float, float]] = []
    for anchor, para, other in TRIPLES:
        a, p, o = await e.embed_passages([anchor, para, other])
        assert len(a) == s.embedding_dim
        assert math.isclose(_dot(a, a), 1.0, abs_tol=1e-3)  # normalized
        rows.append((_dot(a, p), _dot(a, o)))
    with capsys.disabled():
        for (cp, co), t in zip(rows, TRIPLES, strict=True):
            print(f"\n  para={cp:.3f} unrelated={co:.3f}  {t[0]!r}", end="")
    for cp, co in rows:
        assert cp > co + 0.05
    q = await e.embed_query("makanan favoritku apa?")
    assert len(q) == s.embedding_dim
