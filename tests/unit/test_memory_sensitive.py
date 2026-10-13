"""T-304: sensitive-data guard in validate_content. Pure regex tests, no DB, no model."""

import pytest

from duitku.memory.validation import InvalidMemoryError, SensitiveDataError, validate_content

SHOULD_REJECT = [
    # payment cards (Luhn-valid test numbers), plain and obfuscated
    "kartu kredit gue 4111111111111111",
    "cc: 4111 1111 1111 1111",
    "kartu 4111-1111-1111-1111 limit 20jt",
    "5555 5555 5555 4444 exp 12/28",
    "amex 3782 822463 10005",
    "378282246310005",
    "4222222222222",
    "visa 4012888888881881 buat belanja online",
    # 16-digit runs that are not Luhn-valid: still a card/NIK shape
    "1234567890123456",
    "1234 5678 9012 3456",
    "1234-5678-9012-3456",
    # NIK / ID
    "nik gue 3174051503900001",
    "NIK: 3174 0515 0390 0001",
    "nik 3174xxxxxxxxxxxx",
    "no ktp 3174051503900001",
    "npwp 09.254.294.3-407.000",
    # bank account
    "no rekening BCA 1234567890",
    "rek mandiri 123-00-1234567-8",
    "norek 0123456789",
    "account number: 987654321012",
    "No. Rek 5271234567 a.n. Keina",
    # CVV / PIN / OTP
    "cvv 123",
    "CVC: 4567",
    "kode keamanan kartu 789",
    "pin gue 123456",
    "PIN ATM: 4829",
    "otp 482913",
    "kode verifikasi 5521",
    "verification code is 884412",
    # passwords
    "password wifi: kopi123",
    "kata sandi m-banking gue abc123",
    "sandi: rahasia99",
    "sandi m-banking gue 12ab34",
    "Passcode hp 777",
    # API keys / tokens
    "sk-abcdefghijklmnopqrstuvwxyz123456",
    "api key: abcd1234efgh5678",
    "token: ghp_abcdefghijklmnop1234",
    "Bearer abcdefghijklmnopqrstuvwxyz0123",
    "secret = Zm9vYmFyYmF6cXV4MTIzNDU2",
    "da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "AKIAIOSFODNN7EXAMPLE",
    "kunci: Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MGFiY2RlZg",
    # full-width digits / zero-width chars to dodge the regex
    "pin gue \uff11\uff12\uff13\uff14\uff15\uff16",
    "kartu 4111\u200b111111111111",
]

SHOULD_ACCEPT = [
    "gaji masuk tanggal 25",
    "budget makan 1.500.000 per bulan",
    "langganan Netflix 54000",
    "kos Rp 1.250.000",
    "Gojek = transport",
    "kopi masuk kategori makan-minum",
    "tabungan target Rp 50.000.000 akhir tahun",
    "cicilan motor 1.250.000 tiap tanggal 5",
    "utang ke Sandi 500rb",
    "token listrik 100rb tiap bulan",
    "rekening listrik Rp 15000000 bulan ini",
    "rekening listrik sekitar 350rb",
    "liburan 2026-12-20 2026-12-27 ke Bali",
    "gajian 2026-10-25 dan 2026-11-25",
    "budget 1500000 2000000 per bulan",
    "nasi goreng langganan di warung dekat kantor",
    "goal: dana darurat 6x pengeluaran bulanan",
    "pengeluaran bulanan sekitar 8.500.000",
    "suka dikasih ringkasan mingguan tiap Senin",
    "kartu kredit BCA tagihan tanggal 17",
    "limit kartu kredit 20jt",
    "pin lokasi kantor di Sudirman",
    "OTP jangan pernah disimpan",
    "bayar kos tiap tanggal 1, 3 juta",
    "Spotify 54.990 per bulan, Netflix 186.000",
    "nomor hp 081234567890 buat notifikasi",
    "email gue keina@example.com",
    "transfer ke BCA 2500000 tiap awal bulan",
    "asuransi premi 1.234.567 per bulan",
    "dana pensiun target 1.000.000.000.000",
    "kunci jawaban kategori: transport",
    "https://example.com/belanja/promo-akhir-tahun-diskon-besar-besaran",
    "wifi kos 50rb per bulan, patungan 4 orang",
]


@pytest.mark.parametrize("text", SHOULD_REJECT)
def test_sensitive_content_rejected(text: str) -> None:
    with pytest.raises(SensitiveDataError) as exc:
        validate_content(text)
    assert isinstance(exc.value, InvalidMemoryError)  # tool layer can still map it generically
    assert text not in str(exc.value)  # label only, never the matched text


@pytest.mark.parametrize("text", SHOULD_ACCEPT)
def test_normal_finance_content_accepted(text: str) -> None:
    assert validate_content(text) == text


def test_table_sizes() -> None:
    assert len(SHOULD_REJECT) >= 25 and len(SHOULD_ACCEPT) >= 25


def test_known_false_positive_bare_password_word() -> None:
    # Documented trade-off: a finance bot has no reason to store anything mentioning passwords.
    with pytest.raises(SensitiveDataError):
        validate_content("ganti password rutin itu penting")
