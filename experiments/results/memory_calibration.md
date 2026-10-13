# T-303 memory threshold calibration

Dataset: 42 recall pairs (19 relevant), 25 dedupe pairs (12 dup / 13 distinct) + 5 supersede pairs reported separately. INDICATIVE ONLY: tiny, hand-written, 50/50 dev/test split (threshold chosen on dev, scored on test). A threshold is `score >= t`; t is the midpoint of the gap between adjacent scores.

### Recall (query vs memory)

| model | relevant min/med/max | irrelevant min/med/max | AUC | margin | best F1 (all) | best F1 on dev -> test | max R @ P>=0.9 | max R @ P=1.0 | P=1.0 on dev -> test |
|---|---|---|---|---|---|---|---|---|---|
| e5 | 0.783 / 0.851 / 0.907 | 0.763 / 0.798 / 0.849 | 0.867 | -0.066 | t=0.824 P=0.83 R=0.79 F1=0.81 | dev t=0.819 P=0.73 R=0.80 F1=0.76 -> test t=0.819 P=0.80 R=0.89 F1=0.84 | t=0.835 P=0.93 R=0.68 F1=0.79 | t=0.850 P=1.00 R=0.53 F1=0.69 | t=0.853 P=1.00 R=0.44 F1=0.62 |
| bge | 0.382 / 0.611 / 0.810 | 0.299 / 0.399 / 0.570 | 0.902 | -0.188 | t=0.440 P=0.72 R=0.95 F1=0.82 | dev t=0.475 P=0.75 R=0.90 F1=0.82 -> test t=0.475 P=0.73 R=0.89 F1=0.80 | t=0.533 P=0.93 R=0.68 F1=0.79 | t=0.581 P=1.00 R=0.58 F1=0.73 | t=0.634 P=1.00 R=0.33 F1=0.50 |

### Dedupe (memory vs memory)

| model | dup min/med/max | distinct min/med/max | AUC | margin | best F1 (all) | best F1 on dev -> test | max R @ P>=0.9 | max R @ P=1.0 | P=1.0 on dev -> test |
|---|---|---|---|---|---|---|---|---|---|
| e5 | 0.907 / 0.945 / 0.988 | 0.834 / 0.940 / 0.979 | 0.654 | -0.072 | t=0.905 P=0.57 R=1.00 F1=0.73 | dev t=0.907 P=0.60 R=1.00 F1=0.75 -> test t=0.907 P=0.50 R=0.83 F1=0.62 | t=0.980 P=1.00 R=0.17 F1=0.29 | t=0.980 P=1.00 R=0.17 F1=0.29 | t=0.975 P=0.50 R=0.17 F1=0.25 |
| bge | 0.643 / 0.858 / 0.966 | 0.405 / 0.715 / 0.865 | 0.808 | -0.222 | t=0.756 P=0.69 R=0.92 F1=0.79 | dev t=0.756 P=0.71 R=0.83 F1=0.77 -> test t=0.756 P=0.67 R=1.00 F1=0.80 | t=0.874 P=1.00 R=0.50 F1=0.67 | t=0.874 P=1.00 R=0.50 F1=0.67 | t=0.874 P=1.00 R=0.67 F1=0.80 |

### Supersede (conflicting update) pairs: cosine, merged at the dup threshold?

- e5: scores 0.973, 0.961, 0.952, 0.956, 0.918; merged 0/5 at t=0.980 (P=1.0 threshold)
- bge: scores 0.894, 0.846, 0.845, 0.828, 0.682; merged 1/5 at t=0.874 (P=1.0 threshold)

### Cost (CPU)

| model | dim | load s | RSS base MB | RSS loaded MB | RSS after encode MB | 1 text ms median / p95 | 32 texts ms | torch threads |
|---|---|---|---|---|---|---|---|---|
| e5 | 384 | 18.7 | 426 | 805 | 908 | 39.7 / 64.5 | 317 | 4 |
| bge | 1024 | 14.6 | 908 | 1137 | 2215 | 271.9 / 338.2 | 4509 | 4 |

## Appendix: raw scores

Recall (query -> memory)

| split | rel | tag | query | memory | e5 | bge |
|---|---|---|---|---|---|---|
| dev | 1 | easy | ngopi di mana biasanya? | Langganan kopi di Fore Coffee | 0.783 | 0.484 |
| test | 1 | easy | kapan gajian? | Gaji masuk tanggal 25 setiap bulan | 0.851 | 0.723 |
| dev | 1 | slang | kpn gajian | Gaji masuk tanggal 25 setiap bulan | 0.820 | 0.537 |
| test | 1 | slang | Gaji MASUK tgl brp? | Gaji masuk tanggal 25 setiap bulan | 0.837 | 0.497 |
| dev | 1 | easy | gojek masuk kategori apa? | Gojek dicatat sebagai kategori Transport | 0.884 | 0.810 |
| test | 1 | slang | gofud biasa masuk mana? | GoFood dicatat sebagai kategori Makan | 0.832 | 0.443 |
| dev | 1 | easy | budget makan berapa? | Budget makan bulanan Rp1,5 juta | 0.907 | 0.770 |
| test | 1 | easy | aku nabung buat apa ya? | Target tabungan: dana darurat Rp30 juta akhir tahun | 0.803 | 0.534 |
| dev | 1 | slang | KOS BAYAR KAPAN | Bayar kos tiap tanggal 1, Rp1,8 juta | 0.839 | 0.501 |
| test | 1 | easy | naik apa ke kantor? | Ke kantor biasanya naik KRL dari Bekasi | 0.866 | 0.679 |
| dev | 1 | easy | langganan streaming apa aja? | Langganan Netflix dan Spotify tiap bulan | 0.825 | 0.698 |
| test | 1 | short | kopi | Langganan kopi di Fore Coffee | 0.878 | 0.611 |
| dev | 1 | slang | tmpt ngopi fav? | Langganan kopi di Fore Coffee | 0.788 | 0.382 |
| test | 1 | easy | belanja bulanan dimana? | Belanja bulanan biasanya di Indomaret dekat rumah | 0.882 | 0.786 |
| dev | 1 | slang | format laporan yg aku suka? | Suka laporan pengeluaran ringkas per minggu | 0.858 | 0.703 |
| test | 1 | easy | cicilan apa aja? | Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027 | 0.840 | 0.593 |
| dev | 1 | easy | makan siang biasanya habis berapa? | Makan siang kantor rata-rata Rp35 ribu | 0.867 | 0.742 |
| test | 1 | short | bensin | Isi bensin Pertalite tiap hari Sabtu | 0.868 | 0.508 |
| dev | 1 | easy | rencana liburan tahun depan? | Rencana liburan ke Bali tahun depan, target Rp10 juta | 0.902 | 0.719 |
| dev | 0 | easy | ngopi di mana biasanya? | Gaji masuk tanggal 25 setiap bulan | 0.779 | 0.343 |
| test | 0 | easy | ngopi di mana biasanya? | Bayar kos tiap tanggal 1, Rp1,8 juta | 0.763 | 0.367 |
| dev | 0 | hard | ngopi di mana biasanya? | Belanja bulanan biasanya di Indomaret dekat rumah | 0.797 | 0.500 |
| test | 0 | hard | langganan kopi di mana? | Langganan Netflix dan Spotify tiap bulan | 0.770 | 0.515 |
| dev | 0 | hard | naik apa ke kantor? | GoFood dicatat sebagai kategori Makan | 0.791 | 0.313 |
| test | 0 | hard | gojek masuk kategori apa? | Budget makan bulanan Rp1,5 juta | 0.774 | 0.334 |
| dev | 0 | hard | budget makan berapa? | Budget transport Rp500 ribu per bulan | 0.849 | 0.570 |
| test | 0 | hard | kapan gajian? | Bayar kos tiap tanggal 1, Rp1,8 juta | 0.823 | 0.532 |
| dev | 0 | hard | kapan gajian? | Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027 | 0.823 | 0.513 |
| test | 0 | hard | KOS BAYAR KAPAN | Gaji masuk tanggal 25 setiap bulan | 0.815 | 0.362 |
| dev | 0 | slang | kpn gajian | Suka laporan pengeluaran ringkas per minggu | 0.798 | 0.422 |
| test | 0 | slang | gofud biasa masuk mana? | Ke kantor biasanya naik KRL dari Bekasi | 0.803 | 0.299 |
| dev | 0 | easy | aku nabung buat apa ya? | Makan siang kantor rata-rata Rp35 ribu | 0.818 | 0.373 |
| test | 0 | hard | rencana liburan tahun depan? | Target tabungan: dana darurat Rp30 juta akhir tahun | 0.831 | 0.504 |
| dev | 0 | hard | cicilan apa aja? | Rencana liburan ke Bali tahun depan, target Rp10 juta | 0.798 | 0.421 |
| test | 0 | hard | makan siang biasanya habis berapa? | Isi bensin Pertalite tiap hari Sabtu | 0.811 | 0.437 |
| dev | 0 | short | kopi | Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027 | 0.783 | 0.313 |
| test | 0 | short | bensin | Langganan Netflix dan Spotify tiap bulan | 0.790 | 0.365 |
| dev | 0 | easy | belanja bulanan dimana? | Rencana liburan ke Bali tahun depan, target Rp10 juta | 0.818 | 0.466 |
| test | 0 | chat | halo | Langganan kopi di Fore Coffee | 0.784 | 0.433 |
| dev | 0 | chat | makasih ya | Gaji masuk tanggal 25 setiap bulan | 0.834 | 0.312 |
| test | 0 | chat | cuaca hari ini gimana? | Budget makan bulanan Rp1,5 juta | 0.805 | 0.383 |
| dev | 0 | chat | siapa presiden indonesia sekarang? | Ke kantor biasanya naik KRL dari Bekasi | 0.772 | 0.399 |

Dedupe (a <-> b)

| split | label | tag | a | b | e5 | bge |
|---|---|---|---|---|---|---|
| dev | dup | paraphrase | Langganan kopi di Fore Coffee | Biasa ngopi di Fore Coffee | 0.968 | 0.884 |
| test | dup | paraphrase | Gaji masuk tanggal 25 setiap bulan | Tiap tanggal 25 aku gajian | 0.915 | 0.905 |
| dev | dup | paraphrase | Gojek dicatat sebagai kategori Transport | Gojek = kategori Transport | 0.955 | 0.766 |
| test | dup | paraphrase | Budget makan bulanan Rp1,5 juta | Budget makan per bulan 1,5 juta | 0.988 | 0.966 |
| dev | dup | paraphrase | Bayar kos tiap tanggal 1, Rp1,8 juta | Kos Rp1,8 juta dibayar setiap tanggal 1 | 0.978 | 0.935 |
| test | dup | paraphrase | Ke kantor biasanya naik KRL dari Bekasi | Berangkat kerja pakai KRL dari Bekasi | 0.938 | 0.900 |
| dev | dup | paraphrase | Target dana darurat Rp30 juta akhir tahun | Mau punya dana darurat 30 juta di akhir tahun | 0.982 | 0.833 |
| test | dup | paraphrase | Langganan Netflix dan Spotify tiap bulan | Aku langganan Spotify sama Netflix | 0.907 | 0.799 |
| dev | dup | slang | GoFood dicatat sebagai kategori Makan | gofud itu masuk makan | 0.912 | 0.643 |
| test | dup | slang | Suka laporan pengeluaran ringkas per minggu | kasih laporan singkat tiap minggu ya | 0.941 | 0.820 |
| dev | dup | slang | Isi bensin Pertalite tiap hari Sabtu | tiap sabtu aku isi pertalite | 0.926 | 0.820 |
| test | dup | slang | Ada cicilan motor Rp850 ribu per bulan | Bayar cicilan motor 850rb sebulan | 0.949 | 0.942 |
| dev | distinct | merchant | Langganan kopi di Fore Coffee | Langganan kopi di Starbucks | 0.942 | 0.746 |
| test | distinct | category | Budget makan bulanan Rp1,5 juta | Budget transport bulanan Rp1,5 juta | 0.963 | 0.837 |
| dev | distinct | category | Gojek dicatat sebagai kategori Transport | GoFood dicatat sebagai kategori Makan | 0.942 | 0.672 |
| test | distinct | merchant | Gojek dicatat sebagai kategori Transport | Grab dicatat sebagai kategori Transport | 0.979 | 0.818 |
| dev | distinct | date | Gaji masuk tanggal 25 setiap bulan | Bayar kos tiap tanggal 25 | 0.940 | 0.842 |
| test | distinct | date | Bayar kos tiap tanggal 1, Rp1,8 juta | Bayar listrik tiap tanggal 1 | 0.911 | 0.715 |
| dev | distinct | item | Ada cicilan motor Rp850 ribu per bulan | Ada cicilan laptop Rp850 ribu per bulan | 0.972 | 0.865 |
| test | distinct | item | Langganan Netflix tiap bulan | Langganan Spotify tiap bulan | 0.948 | 0.825 |
| dev | distinct | item | Ke kantor biasanya naik KRL dari Bekasi | Pulang kantor biasanya naik ojek online | 0.902 | 0.678 |
| test | distinct | item | Target dana darurat Rp30 juta akhir tahun | Target liburan ke Bali Rp10 juta tahun depan | 0.919 | 0.579 |
| dev | distinct | unrelated | Langganan kopi di Fore Coffee | Gaji masuk tanggal 25 setiap bulan | 0.834 | 0.405 |
| test | distinct | unrelated | Suka laporan pengeluaran ringkas per minggu | Isi bensin Pertalite tiap hari Sabtu | 0.884 | 0.492 |
| dev | distinct | unrelated | Makan siang kantor rata-rata Rp35 ribu | Belanja bulanan di Indomaret dekat rumah | 0.877 | 0.418 |
| n/a | supersede | update | Budget makan bulanan Rp1,5 juta | Budget makan bulanan Rp2 juta | 0.973 | 0.894 |
| n/a | supersede | update | Gaji masuk tanggal 25 setiap bulan | Gaji sekarang masuk tanggal 28 setiap bulan | 0.961 | 0.846 |
| n/a | supersede | update | Bayar kos tiap tanggal 1, Rp1,8 juta | Bayar kos tiap tanggal 1, naik jadi Rp2 juta | 0.952 | 0.845 |
| n/a | supersede | update | Target dana darurat Rp30 juta akhir tahun | Target dana darurat naik jadi Rp50 juta akhir tahun | 0.956 | 0.828 |
| n/a | supersede | update | Langganan kopi di Fore Coffee | Sekarang pindah langganan kopi ke Kopi Kenangan | 0.918 | 0.682 |
