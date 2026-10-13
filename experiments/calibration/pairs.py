"""Labeled Indonesian pairs for T-303 (plain data, meant to be reviewed and extended).

LABEL POLICY
RECALL (query, memory, relevant): relevant = a good assistant should inject this memory into the
prompt when the user sends this message. Negatives include hard ones: same domain but not what is
asked (transport query vs food memory), two date facts, two "langganan" facts, plus chit-chat
("halo") that must recall nothing. Auto-recall runs on EVERY message, so false positives cost
prompt noise on every turn; false negatives are softened by the explicit `recall_memory` tool.

DEDUPE (memory_a, memory_b, label): `save` overwrites the nearest same-kind row when
similarity >= threshold, so a wrong merge destroys a fact while a missed merge only leaves a
near-duplicate row. Hence HIGH precision matters most. Three labels:
- "dup":       paraphrase / slang / reordering of the SAME fact (same slot, same value). Merge.
- "distinct": a different fact: other merchant (Fore vs Starbucks: the user may use both, merging
              would erase one), other category with the same number, other item with the same
              date, or unrelated. Do not merge.
- "supersede": CONFLICTING UPDATE: same slot, new value ("budget makan 1,5 juta" -> "2 juta"). The
              spec's behaviour (overwrite the row, newest wins) is the RIGHT outcome: keeping both
              would feed contradictory facts into the prompt. The cost is silent loss of the old
              value, accepted (history is not a requirement). Embeddings cannot tell a changed
              number from a paraphrase, so these pairs are NOT used to fit the threshold; they are
              reported as "how many get merged" only. Merging them is desired, not an error.
Pairs are split deterministically 50/50 per label into dev (pick threshold) and test (report).
"""

from dataclasses import dataclass
from typing import Literal

from calibration.metrics import assign_split

DedupeLabel = Literal["dup", "distinct", "supersede"]


@dataclass(frozen=True)
class RecallPair:
    query: str
    memory: str
    relevant: bool
    tag: str  # easy | hard | slang | short | chat
    split: str = "dev"


@dataclass(frozen=True)
class DedupePair:
    a: str
    b: str
    label: DedupeLabel
    tag: str  # paraphrase | slang | merchant | category | date | item | unrelated | update
    split: str = "dev"


# fmt: off
_RECALL: list[tuple[str, str, bool, str]] = [
    # --- relevant ---
    ("ngopi di mana biasanya?", "Langganan kopi di Fore Coffee", True, "easy"),
    ("kapan gajian?", "Gaji masuk tanggal 25 setiap bulan", True, "easy"),
    ("kpn gajian", "Gaji masuk tanggal 25 setiap bulan", True, "slang"),
    ("Gaji MASUK tgl brp?", "Gaji masuk tanggal 25 setiap bulan", True, "slang"),
    ("gojek masuk kategori apa?", "Gojek dicatat sebagai kategori Transport", True, "easy"),
    ("gofud biasa masuk mana?", "GoFood dicatat sebagai kategori Makan", True, "slang"),
    ("budget makan berapa?", "Budget makan bulanan Rp1,5 juta", True, "easy"),
    ("aku nabung buat apa ya?", "Target tabungan: dana darurat Rp30 juta akhir tahun", True, "easy"),
    ("KOS BAYAR KAPAN", "Bayar kos tiap tanggal 1, Rp1,8 juta", True, "slang"),
    ("naik apa ke kantor?", "Ke kantor biasanya naik KRL dari Bekasi", True, "easy"),
    ("langganan streaming apa aja?", "Langganan Netflix dan Spotify tiap bulan", True, "easy"),
    ("kopi", "Langganan kopi di Fore Coffee", True, "short"),
    ("tmpt ngopi fav?", "Langganan kopi di Fore Coffee", True, "slang"),
    ("belanja bulanan dimana?", "Belanja bulanan biasanya di Indomaret dekat rumah", True, "easy"),
    ("format laporan yg aku suka?", "Suka laporan pengeluaran ringkas per minggu", True, "slang"),
    ("cicilan apa aja?", "Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027", True, "easy"),
    ("makan siang biasanya habis berapa?", "Makan siang kantor rata-rata Rp35 ribu", True, "easy"),
    ("bensin", "Isi bensin Pertalite tiap hari Sabtu", True, "short"),
    ("rencana liburan tahun depan?", "Rencana liburan ke Bali tahun depan, target Rp10 juta", True, "easy"),
    # --- irrelevant ---
    ("ngopi di mana biasanya?", "Gaji masuk tanggal 25 setiap bulan", False, "easy"),
    ("ngopi di mana biasanya?", "Bayar kos tiap tanggal 1, Rp1,8 juta", False, "easy"),
    ("ngopi di mana biasanya?", "Belanja bulanan biasanya di Indomaret dekat rumah", False, "hard"),
    ("langganan kopi di mana?", "Langganan Netflix dan Spotify tiap bulan", False, "hard"),
    ("naik apa ke kantor?", "GoFood dicatat sebagai kategori Makan", False, "hard"),
    ("gojek masuk kategori apa?", "Budget makan bulanan Rp1,5 juta", False, "hard"),
    ("budget makan berapa?", "Budget transport Rp500 ribu per bulan", False, "hard"),
    ("kapan gajian?", "Bayar kos tiap tanggal 1, Rp1,8 juta", False, "hard"),
    ("kapan gajian?", "Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027", False, "hard"),
    ("KOS BAYAR KAPAN", "Gaji masuk tanggal 25 setiap bulan", False, "hard"),
    ("kpn gajian", "Suka laporan pengeluaran ringkas per minggu", False, "slang"),
    ("gofud biasa masuk mana?", "Ke kantor biasanya naik KRL dari Bekasi", False, "slang"),
    ("aku nabung buat apa ya?", "Makan siang kantor rata-rata Rp35 ribu", False, "easy"),
    ("rencana liburan tahun depan?", "Target tabungan: dana darurat Rp30 juta akhir tahun", False, "hard"),
    ("cicilan apa aja?", "Rencana liburan ke Bali tahun depan, target Rp10 juta", False, "hard"),
    ("makan siang biasanya habis berapa?", "Isi bensin Pertalite tiap hari Sabtu", False, "hard"),
    ("kopi", "Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027", False, "short"),
    ("bensin", "Langganan Netflix dan Spotify tiap bulan", False, "short"),
    ("belanja bulanan dimana?", "Rencana liburan ke Bali tahun depan, target Rp10 juta", False, "easy"),
    ("halo", "Langganan kopi di Fore Coffee", False, "chat"),
    ("makasih ya", "Gaji masuk tanggal 25 setiap bulan", False, "chat"),
    ("cuaca hari ini gimana?", "Budget makan bulanan Rp1,5 juta", False, "chat"),
    ("siapa presiden indonesia sekarang?", "Ke kantor biasanya naik KRL dari Bekasi", False, "chat"),
]

_DEDUPE: list[tuple[str, str, DedupeLabel, str]] = [
    # --- dup: same fact, different wording ---
    ("Langganan kopi di Fore Coffee", "Biasa ngopi di Fore Coffee", "dup", "paraphrase"),
    ("Gaji masuk tanggal 25 setiap bulan", "Tiap tanggal 25 aku gajian", "dup", "paraphrase"),
    ("Gojek dicatat sebagai kategori Transport", "Gojek = kategori Transport", "dup", "paraphrase"),
    ("Budget makan bulanan Rp1,5 juta", "Budget makan per bulan 1,5 juta", "dup", "paraphrase"),
    ("Bayar kos tiap tanggal 1, Rp1,8 juta", "Kos Rp1,8 juta dibayar setiap tanggal 1", "dup", "paraphrase"),
    ("Ke kantor biasanya naik KRL dari Bekasi", "Berangkat kerja pakai KRL dari Bekasi", "dup", "paraphrase"),
    ("Target dana darurat Rp30 juta akhir tahun", "Mau punya dana darurat 30 juta di akhir tahun", "dup", "paraphrase"),
    ("Langganan Netflix dan Spotify tiap bulan", "Aku langganan Spotify sama Netflix", "dup", "paraphrase"),
    ("GoFood dicatat sebagai kategori Makan", "gofud itu masuk makan", "dup", "slang"),
    ("Suka laporan pengeluaran ringkas per minggu", "kasih laporan singkat tiap minggu ya", "dup", "slang"),
    ("Isi bensin Pertalite tiap hari Sabtu", "tiap sabtu aku isi pertalite", "dup", "slang"),
    ("Ada cicilan motor Rp850 ribu per bulan", "Bayar cicilan motor 850rb sebulan", "dup", "slang"),
    # --- distinct: related-but-different or unrelated ---
    ("Langganan kopi di Fore Coffee", "Langganan kopi di Starbucks", "distinct", "merchant"),
    ("Budget makan bulanan Rp1,5 juta", "Budget transport bulanan Rp1,5 juta", "distinct", "category"),
    ("Gojek dicatat sebagai kategori Transport", "GoFood dicatat sebagai kategori Makan", "distinct", "category"),
    ("Gojek dicatat sebagai kategori Transport", "Grab dicatat sebagai kategori Transport", "distinct", "merchant"),
    ("Gaji masuk tanggal 25 setiap bulan", "Bayar kos tiap tanggal 25", "distinct", "date"),
    ("Bayar kos tiap tanggal 1, Rp1,8 juta", "Bayar listrik tiap tanggal 1", "distinct", "date"),
    ("Ada cicilan motor Rp850 ribu per bulan", "Ada cicilan laptop Rp850 ribu per bulan", "distinct", "item"),
    ("Langganan Netflix tiap bulan", "Langganan Spotify tiap bulan", "distinct", "item"),
    ("Ke kantor biasanya naik KRL dari Bekasi", "Pulang kantor biasanya naik ojek online", "distinct", "item"),
    ("Target dana darurat Rp30 juta akhir tahun", "Target liburan ke Bali Rp10 juta tahun depan", "distinct", "item"),
    ("Langganan kopi di Fore Coffee", "Gaji masuk tanggal 25 setiap bulan", "distinct", "unrelated"),
    ("Suka laporan pengeluaran ringkas per minggu", "Isi bensin Pertalite tiap hari Sabtu", "distinct", "unrelated"),
    ("Makan siang kantor rata-rata Rp35 ribu", "Belanja bulanan di Indomaret dekat rumah", "distinct", "unrelated"),
    # --- supersede: same slot, new value (merge is the desired outcome, not fitted) ---
    ("Budget makan bulanan Rp1,5 juta", "Budget makan bulanan Rp2 juta", "supersede", "update"),
    ("Gaji masuk tanggal 25 setiap bulan", "Gaji sekarang masuk tanggal 28 setiap bulan", "supersede", "update"),
    ("Bayar kos tiap tanggal 1, Rp1,8 juta", "Bayar kos tiap tanggal 1, naik jadi Rp2 juta", "supersede", "update"),
    ("Target dana darurat Rp30 juta akhir tahun", "Target dana darurat naik jadi Rp50 juta akhir tahun", "supersede", "update"),
    ("Langganan kopi di Fore Coffee", "Sekarang pindah langganan kopi ke Kopi Kenangan", "supersede", "update"),
]
# fmt: on


def recall_pairs() -> list[RecallPair]:
    splits = assign_split([r for _, _, r, _ in _RECALL])
    return [RecallPair(q, m, r, t, s) for (q, m, r, t), s in zip(_RECALL, splits, strict=True)]


def dedupe_pairs() -> list[DedupePair]:
    strict = [lab == "dup" for _, _, lab, _ in _DEDUPE if lab != "supersede"]
    splits = iter(assign_split(strict))
    return [
        DedupePair(a, b, lab, t, "n/a" if lab == "supersede" else next(splits))
        for a, b, lab, t in _DEDUPE
    ]
