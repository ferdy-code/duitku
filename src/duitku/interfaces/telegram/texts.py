"""User-facing Indonesian strings for the Telegram interface (the agent's own replies come from
the model; these are the bot's fixed ones)."""

MAX_TEXT_CHARS = 2000

RATE_LIMITED = "Terlalu banyak pesan dalam semenit. Tunggu sebentar lalu coba lagi ya."
TOO_LONG = (
    f"Pesanmu terlalu panjang (maksimal {MAX_TEXT_CHARS} karakter). "
    "Coba ringkas atau kirim dalam beberapa pesan ya."
)
GENERIC_APOLOGY = "Maaf, ada gangguan di sisiku. Coba lagi sebentar lagi ya."

START = (
    "Halo! Aku Duitku, asisten pencatat pengeluaranmu.\n\n"
    "Tinggal ceritakan pengeluaranmu seperti ngobrol biasa, misalnya:\n"
    "- kopi 18rb\n"
    "- makan siang 35rb di warteg\n"
    "- total pengeluaran bulan ini berapa?\n\n"
    "Ketik /bantuan untuk contoh lainnya."
)

HELP = (
    "Contoh yang bisa kamu ketik:\n"
    "- catat: kopi 18rb, bensin 50 ribu kemarin\n"
    "- lihat: pengeluaran hari ini, total makan minum minggu ini\n"
    "- ubah: ubah yang terakhir jadi 40rb\n"
    "- ingat: aku tidak suka kopi pahit\n\n"
    "Perintah:\n"
    "/start - perkenalan\n"
    "/batal - batalkan permintaan konfirmasi yang sedang menunggu\n"
    "/bantuan - pesan ini"
)

BTN_APPROVE = "✅ Setuju"
BTN_REJECT = "❌ Batal"
OUTCOME_APPROVED = "✅ Disetujui"
OUTCOME_REJECTED = "❌ Dibatalkan"
