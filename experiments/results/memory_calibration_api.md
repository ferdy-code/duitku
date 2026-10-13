# Embedding model comparison: gateway API models vs T-303 local baseline

Dataset: 42 recall pairs (19 relevant), 25 dedupe pairs (12 dup / 13 distinct) + 5 supersede pairs reported separately. INDICATIVE ONLY: tiny, hand-written, 50/50 dev/test split (threshold chosen on dev, scored on test). A threshold is `score >= t`; t is the midpoint of the gap between adjacent scores.

### Recall (query vs memory)

| model | relevant min/med/max | irrelevant min/med/max | AUC | margin | best F1 (all) | best F1 on dev -> test | max R @ P>=0.9 | max R @ P=1.0 | P=1.0 on dev -> test |
|---|---|---|---|---|---|---|---|---|---|
| e5 | 0.783 / 0.851 / 0.907 | 0.763 / 0.798 / 0.849 | 0.870 | -0.066 | t=0.824 P=0.83 R=0.79 F1=0.81 | dev t=0.819 P=0.73 R=0.80 F1=0.76 -> test t=0.819 P=0.80 R=0.89 F1=0.84 | t=0.835 P=0.93 R=0.68 F1=0.79 | t=0.850 P=1.00 R=0.53 F1=0.69 | t=0.853 P=1.00 R=0.44 F1=0.62 |
| bge | 0.382 / 0.611 / 0.810 | 0.299 / 0.399 / 0.570 | 0.902 | -0.188 | t=0.440 P=0.72 R=0.95 F1=0.82 | dev t=0.475 P=0.75 R=0.90 F1=0.82 -> test t=0.475 P=0.73 R=0.89 F1=0.80 | t=0.533 P=0.93 R=0.68 F1=0.79 | t=0.581 P=1.00 R=0.58 F1=0.73 | t=0.634 P=1.00 R=0.33 F1=0.50 |
| api-bge-m3 | 0.382 / 0.612 / 0.810 | 0.299 / 0.399 / 0.570 | 0.902 | -0.187 | t=0.440 P=0.72 R=0.95 F1=0.82 | dev t=0.475 P=0.75 R=0.90 F1=0.82 -> test t=0.475 P=0.73 R=0.89 F1=0.80 | t=0.533 P=0.93 R=0.68 F1=0.79 | t=0.581 P=1.00 R=0.58 F1=0.73 | t=0.634 P=1.00 R=0.33 F1=0.50 |
| api-qwen3 | 0.301 / 0.511 / 0.756 | 0.163 / 0.304 / 0.485 | 0.915 | -0.184 | t=0.392 P=0.84 R=0.84 F1=0.84 | dev t=0.427 P=0.88 R=0.70 F1=0.78 -> test t=0.427 P=1.00 R=0.78 F1=0.88 | t=0.438 P=0.93 R=0.74 F1=0.82 | t=0.490 P=1.00 R=0.63 F1=0.77 | t=0.512 P=1.00 R=0.33 F1=0.50 |
| api-qwen3-noinst | 0.359 / 0.527 / 0.738 | 0.239 / 0.344 / 0.541 | 0.915 | -0.182 | t=0.478 P=0.89 R=0.84 F1=0.86 | dev t=0.462 P=0.90 R=0.90 F1=0.90 -> test t=0.462 P=0.78 R=0.78 F1=0.78 | t=0.483 P=0.94 R=0.79 F1=0.86 | t=0.542 P=1.00 R=0.42 F1=0.59 | t=0.559 P=1.00 R=0.33 F1=0.50 |
| api-3small | 0.211 / 0.532 / 0.755 | 0.135 / 0.291 / 0.476 | 0.842 | -0.265 | t=0.304 P=0.70 R=0.84 F1=0.76 | dev t=0.508 P=1.00 R=0.60 F1=0.75 -> test t=0.508 P=1.00 R=0.56 F1=0.71 | t=0.410 P=0.92 R=0.63 F1=0.75 | t=0.503 P=1.00 R=0.58 F1=0.73 | t=0.508 P=1.00 R=0.56 F1=0.71 |
| api-3small-384 | 0.165 / 0.566 / 0.764 | 0.131 / 0.315 / 0.470 | 0.824 | -0.306 | t=0.357 P=0.71 R=0.79 F1=0.75 | dev t=0.519 P=1.00 R=0.60 F1=0.75 -> test t=0.519 P=1.00 R=0.44 F1=0.62 | t=0.449 P=0.92 R=0.63 F1=0.75 | t=0.490 P=1.00 R=0.58 F1=0.73 | t=0.519 P=1.00 R=0.44 F1=0.62 |
| api-3large | 0.365 / 0.555 / 0.761 | 0.184 / 0.300 / 0.565 | 0.954 | -0.200 | t=0.467 P=0.89 R=0.89 F1=0.89 | dev t=0.463 P=0.91 R=1.00 F1=0.95 -> test t=0.463 P=0.78 R=0.78 F1=0.78 | t=0.473 P=0.94 R=0.84 F1=0.89 | t=0.577 P=1.00 R=0.47 F1=0.64 | t=0.577 P=1.00 R=0.44 F1=0.62 |
| api-3large-384 | 0.422 / 0.600 / 0.780 | 0.230 / 0.339 / 0.614 | 0.954 | -0.192 | t=0.512 P=0.94 R=0.89 F1=0.92 | dev t=0.511 P=0.91 R=1.00 F1=0.95 -> test t=0.511 P=1.00 R=0.78 F1=0.88 | t=0.512 P=0.94 R=0.89 F1=0.92 | t=0.630 P=1.00 R=0.47 F1=0.64 | t=0.630 P=1.00 R=0.44 F1=0.62 |
| api-mistral | 0.616 / 0.748 / 0.838 | 0.544 / 0.645 / 0.731 | 0.835 | -0.115 | t=0.669 P=0.68 R=0.89 F1=0.77 | dev t=0.743 P=1.00 R=0.70 F1=0.82 -> test t=0.743 P=1.00 R=0.33 F1=0.50 | t=0.739 P=1.00 R=0.53 F1=0.69 | t=0.739 P=1.00 R=0.53 F1=0.69 | t=0.743 P=1.00 R=0.33 F1=0.50 |

### Dedupe (memory vs memory)

| model | dup min/med/max | distinct min/med/max | AUC | margin | best F1 (all) | best F1 on dev -> test | max R @ P>=0.9 | max R @ P=1.0 | P=1.0 on dev -> test |
|---|---|---|---|---|---|---|---|---|---|
| e5 | 0.907 / 0.945 / 0.988 | 0.834 / 0.940 / 0.979 | 0.654 | -0.072 | t=0.905 P=0.57 R=1.00 F1=0.73 | dev t=0.907 P=0.60 R=1.00 F1=0.75 -> test t=0.907 P=0.55 R=1.00 F1=0.71 | t=0.980 P=1.00 R=0.17 F1=0.29 | t=0.980 P=1.00 R=0.17 F1=0.29 | t=0.975 P=0.50 R=0.17 F1=0.25 |
| bge | 0.643 / 0.859 / 0.966 | 0.405 / 0.715 / 0.865 | 0.808 | -0.222 | t=0.756 P=0.69 R=0.92 F1=0.79 | dev t=0.756 P=0.71 R=0.83 F1=0.77 -> test t=0.756 P=0.67 R=1.00 F1=0.80 | t=0.875 P=1.00 R=0.50 F1=0.67 | t=0.875 P=1.00 R=0.50 F1=0.67 | t=0.875 P=1.00 R=0.67 F1=0.80 |
| api-bge-m3 | 0.644 / 0.858 / 0.966 | 0.405 / 0.715 / 0.865 | 0.808 | -0.221 | t=0.756 P=0.69 R=0.92 F1=0.79 | dev t=0.756 P=0.71 R=0.83 F1=0.77 -> test t=0.756 P=0.67 R=1.00 F1=0.80 | t=0.874 P=1.00 R=0.50 F1=0.67 | t=0.874 P=1.00 R=0.50 F1=0.67 | t=0.874 P=1.00 R=0.67 F1=0.80 |
| api-qwen3 | 0.650 / 0.829 / 0.901 | 0.235 / 0.637 / 0.792 | 0.865 | -0.143 | t=0.810 P=1.00 R=0.67 F1=0.80 | dev t=0.810 P=1.00 R=0.67 F1=0.80 -> test t=0.810 P=1.00 R=0.67 F1=0.80 | t=0.810 P=1.00 R=0.67 F1=0.80 | t=0.810 P=1.00 R=0.67 F1=0.80 | t=0.810 P=1.00 R=0.67 F1=0.80 |
| api-qwen3-noinst | 0.650 / 0.829 / 0.901 | 0.235 / 0.637 / 0.792 | 0.865 | -0.143 | t=0.810 P=1.00 R=0.67 F1=0.80 | dev t=0.810 P=1.00 R=0.67 F1=0.80 -> test t=0.810 P=1.00 R=0.67 F1=0.80 | t=0.810 P=1.00 R=0.67 F1=0.80 | t=0.810 P=1.00 R=0.67 F1=0.80 | t=0.810 P=1.00 R=0.67 F1=0.80 |
| api-3small | 0.517 / 0.767 / 0.939 | 0.227 / 0.694 / 0.850 | 0.718 | -0.334 | t=0.471 P=0.55 R=1.00 F1=0.71 | dev t=0.471 P=0.55 R=1.00 F1=0.71 -> test t=0.471 P=0.55 R=1.00 F1=0.71 | t=0.855 P=1.00 R=0.33 F1=0.50 | t=0.855 P=1.00 R=0.33 F1=0.50 | t=0.855 P=1.00 R=0.33 F1=0.50 |
| api-3small-384 | 0.524 / 0.782 / 0.939 | 0.263 / 0.721 / 0.856 | 0.699 | -0.332 | t=0.517 P=0.55 R=1.00 F1=0.71 | dev t=0.517 P=0.55 R=1.00 F1=0.71 -> test t=0.517 P=0.55 R=1.00 F1=0.71 | t=0.866 P=1.00 R=0.33 F1=0.50 | t=0.866 P=1.00 R=0.33 F1=0.50 | t=0.866 P=1.00 R=0.33 F1=0.50 |
| api-3large | 0.504 / 0.801 / 0.919 | 0.293 / 0.681 / 0.819 | 0.776 | -0.314 | t=0.748 P=0.71 R=0.83 F1=0.77 | dev t=0.761 P=0.83 R=0.83 F1=0.83 -> test t=0.761 P=0.57 R=0.67 F1=0.62 | t=0.819 P=1.00 R=0.33 F1=0.50 | t=0.819 P=1.00 R=0.33 F1=0.50 | t=0.817 P=0.67 R=0.33 F1=0.44 |
| api-3large-384 | 0.550 / 0.823 / 0.929 | 0.365 / 0.730 / 0.829 | 0.833 | -0.279 | t=0.801 P=0.83 R=0.83 F1=0.83 | dev t=0.785 P=0.83 R=0.83 F1=0.83 -> test t=0.785 P=0.62 R=0.83 F1=0.71 | t=0.830 P=1.00 R=0.42 F1=0.59 | t=0.830 P=1.00 R=0.42 F1=0.59 | t=0.830 P=1.00 R=0.50 F1=0.67 |
| api-mistral | 0.776 / 0.887 / 0.961 | 0.631 / 0.857 / 0.907 | 0.641 | -0.131 | t=0.739 P=0.55 R=1.00 F1=0.71 | dev t=0.737 P=0.55 R=1.00 F1=0.71 -> test t=0.737 P=0.55 R=1.00 F1=0.71 | t=0.909 P=1.00 R=0.33 F1=0.50 | t=0.909 P=1.00 R=0.33 F1=0.50 | t=0.919 P=1.00 R=0.17 F1=0.29 |

### Supersede (conflicting update) pairs: cosine, merged at the dup threshold?

- e5: scores 0.973, 0.961, 0.952, 0.956, 0.918; merged 0/5 at t=0.980 (P=1.0 threshold)
- bge: scores 0.894, 0.846, 0.845, 0.828, 0.682; merged 1/5 at t=0.875 (P=1.0 threshold)
- api-bge-m3: scores 0.894, 0.846, 0.845, 0.828, 0.682; merged 1/5 at t=0.874 (P=1.0 threshold)
- api-qwen3: scores 0.937, 0.808, 0.880, 0.867, 0.586; merged 3/5 at t=0.810 (P=1.0 threshold)
- api-qwen3-noinst: scores 0.937, 0.808, 0.880, 0.867, 0.586; merged 3/5 at t=0.810 (P=1.0 threshold)
- api-3small: scores 0.910, 0.834, 0.897, 0.896, 0.653; merged 3/5 at t=0.855 (P=1.0 threshold)
- api-3small-384: scores 0.915, 0.854, 0.902, 0.902, 0.674; merged 3/5 at t=0.866 (P=1.0 threshold)
- api-3large: scores 0.894, 0.836, 0.821, 0.897, 0.584; merged 4/5 at t=0.819 (P=1.0 threshold)
- api-3large-384: scores 0.893, 0.873, 0.835, 0.911, 0.603; merged 4/5 at t=0.830 (P=1.0 threshold)
- api-mistral: scores 0.944, 0.929, 0.960, 0.960, 0.771; merged 4/5 at t=0.909 (P=1.0 threshold)

### Cost (gateway, from this machine)

| model | dim | 1 text ms median / p95 | 32 texts ms | tokens used (whole run) |
|---|---|---|---|---|
| api-bge-m3 | 1024 | 477 / 563 | 789 | not reported |
| api-qwen3 | 1024 | 742 / 3883 | 2696 | not reported |
| api-qwen3-noinst | 1024 | 864 / 5244 | 1883 | not reported |
| api-3small | 1536 | 3020 / 10206 | 19214 | 1590 |
| api-3small-384 | 384 | 2451 / 6721 | 9818 | 1590 |
| api-3large | 3072 | 4056 / 7601 | 50598 | 1590 |
| api-3large-384 | 384 | 3573 / 7273 | 17493 | 1590 |
| api-mistral | 1024 | 454 / 831 | 812 | 2338 |

## Appendix: raw scores

Recall (query -> memory)

| split | rel | tag | query | memory | e5 | bge | api-bge-m3 | api-qwen3 | api-qwen3-noinst | api-3small | api-3small-384 | api-3large | api-3large-384 | api-mistral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | 1 | easy | ngopi di mana biasanya? | Langganan kopi di Fore Coffee | 0.783 | 0.484 | 0.484 | 0.322 | 0.383 | 0.318 | 0.379 | 0.493 | 0.564 | 0.616 |
| test | 1 | easy | kapan gajian? | Gaji masuk tanggal 25 setiap bulan | 0.851 | 0.723 | 0.723 | 0.505 | 0.528 | 0.415 | 0.455 | 0.612 | 0.679 | 0.708 |
| dev | 1 | slang | kpn gajian | Gaji masuk tanggal 25 setiap bulan | 0.820 | 0.537 | 0.537 | 0.352 | 0.485 | 0.283 | 0.279 | 0.467 | 0.535 | 0.671 |
| test | 1 | slang | Gaji MASUK tgl brp? | Gaji masuk tanggal 25 setiap bulan | 0.837 | 0.497 | 0.497 | 0.707 | 0.726 | 0.669 | 0.700 | 0.710 | 0.767 | 0.819 |
| dev | 1 | easy | gojek masuk kategori apa? | Gojek dicatat sebagai kategori Transport | 0.884 | 0.810 | 0.810 | 0.752 | 0.738 | 0.755 | 0.764 | 0.761 | 0.780 | 0.833 |
| test | 1 | slang | gofud biasa masuk mana? | GoFood dicatat sebagai kategori Makan | 0.832 | 0.443 | 0.443 | 0.422 | 0.478 | 0.316 | 0.363 | 0.365 | 0.422 | 0.672 |
| dev | 1 | easy | budget makan berapa? | Budget makan bulanan Rp1,5 juta | 0.907 | 0.770 | 0.770 | 0.756 | 0.738 | 0.703 | 0.748 | 0.664 | 0.727 | 0.783 |
| test | 1 | easy | aku nabung buat apa ya? | Target tabungan: dana darurat Rp30 juta akhir tahun | 0.803 | 0.534 | 0.534 | 0.397 | 0.359 | 0.305 | 0.296 | 0.416 | 0.445 | 0.698 |
| dev | 1 | slang | KOS BAYAR KAPAN | Bayar kos tiap tanggal 1, Rp1,8 juta | 0.839 | 0.501 | 0.500 | 0.539 | 0.527 | 0.354 | 0.373 | 0.589 | 0.647 | 0.770 |
| test | 1 | easy | naik apa ke kantor? | Ke kantor biasanya naik KRL dari Bekasi | 0.866 | 0.679 | 0.679 | 0.511 | 0.543 | 0.551 | 0.573 | 0.612 | 0.646 | 0.725 |
| dev | 1 | easy | langganan streaming apa aja? | Langganan Netflix dan Spotify tiap bulan | 0.825 | 0.698 | 0.698 | 0.641 | 0.578 | 0.540 | 0.567 | 0.645 | 0.684 | 0.781 |
| test | 1 | short | kopi | Langganan kopi di Fore Coffee | 0.878 | 0.611 | 0.612 | 0.615 | 0.681 | 0.531 | 0.510 | 0.499 | 0.567 | 0.728 |
| dev | 1 | slang | tmpt ngopi fav? | Langganan kopi di Fore Coffee | 0.788 | 0.382 | 0.382 | 0.301 | 0.488 | 0.251 | 0.306 | 0.474 | 0.551 | 0.616 |
| test | 1 | easy | belanja bulanan dimana? | Belanja bulanan biasanya di Indomaret dekat rumah | 0.882 | 0.786 | 0.786 | 0.679 | 0.693 | 0.727 | 0.748 | 0.705 | 0.770 | 0.838 |
| dev | 1 | slang | format laporan yg aku suka? | Suka laporan pengeluaran ringkas per minggu | 0.858 | 0.703 | 0.703 | 0.453 | 0.596 | 0.558 | 0.596 | 0.554 | 0.600 | 0.755 |
| test | 1 | easy | cicilan apa aja? | Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027 | 0.840 | 0.593 | 0.593 | 0.495 | 0.398 | 0.532 | 0.566 | 0.555 | 0.588 | 0.695 |
| dev | 1 | easy | makan siang biasanya habis berapa? | Makan siang kantor rata-rata Rp35 ribu | 0.867 | 0.742 | 0.742 | 0.619 | 0.526 | 0.601 | 0.599 | 0.504 | 0.561 | 0.806 |
| test | 1 | short | bensin | Isi bensin Pertalite tiap hari Sabtu | 0.868 | 0.508 | 0.507 | 0.483 | 0.505 | 0.211 | 0.165 | 0.524 | 0.563 | 0.748 |
| dev | 1 | easy | rencana liburan tahun depan? | Rencana liburan ke Bali tahun depan, target Rp10 juta | 0.902 | 0.719 | 0.719 | 0.596 | 0.511 | 0.597 | 0.615 | 0.639 | 0.664 | 0.831 |
| dev | 0 | easy | ngopi di mana biasanya? | Gaji masuk tanggal 25 setiap bulan | 0.779 | 0.343 | 0.343 | 0.352 | 0.381 | 0.292 | 0.351 | 0.276 | 0.351 | 0.631 |
| test | 0 | easy | ngopi di mana biasanya? | Bayar kos tiap tanggal 1, Rp1,8 juta | 0.763 | 0.367 | 0.367 | 0.301 | 0.264 | 0.236 | 0.329 | 0.272 | 0.270 | 0.655 |
| dev | 0 | hard | ngopi di mana biasanya? | Belanja bulanan biasanya di Indomaret dekat rumah | 0.797 | 0.500 | 0.500 | 0.372 | 0.421 | 0.361 | 0.373 | 0.459 | 0.487 | 0.731 |
| test | 0 | hard | langganan kopi di mana? | Langganan Netflix dan Spotify tiap bulan | 0.770 | 0.515 | 0.515 | 0.326 | 0.366 | 0.295 | 0.339 | 0.471 | 0.477 | 0.712 |
| dev | 0 | hard | naik apa ke kantor? | GoFood dicatat sebagai kategori Makan | 0.791 | 0.313 | 0.313 | 0.349 | 0.344 | 0.189 | 0.152 | 0.220 | 0.263 | 0.574 |
| test | 0 | hard | gojek masuk kategori apa? | Budget makan bulanan Rp1,5 juta | 0.774 | 0.334 | 0.334 | 0.268 | 0.339 | 0.300 | 0.386 | 0.271 | 0.316 | 0.586 |
| dev | 0 | hard | budget makan berapa? | Budget transport Rp500 ribu per bulan | 0.849 | 0.570 | 0.570 | 0.485 | 0.541 | 0.476 | 0.470 | 0.565 | 0.614 | 0.728 |
| test | 0 | hard | kapan gajian? | Bayar kos tiap tanggal 1, Rp1,8 juta | 0.823 | 0.532 | 0.532 | 0.304 | 0.334 | 0.311 | 0.348 | 0.466 | 0.489 | 0.667 |
| dev | 0 | hard | kapan gajian? | Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027 | 0.823 | 0.513 | 0.513 | 0.222 | 0.289 | 0.259 | 0.315 | 0.374 | 0.441 | 0.613 |
| test | 0 | hard | KOS BAYAR KAPAN | Gaji masuk tanggal 25 setiap bulan | 0.815 | 0.362 | 0.362 | 0.386 | 0.481 | 0.243 | 0.248 | 0.405 | 0.445 | 0.638 |
| dev | 0 | slang | kpn gajian | Suka laporan pengeluaran ringkas per minggu | 0.798 | 0.422 | 0.422 | 0.273 | 0.439 | 0.259 | 0.250 | 0.358 | 0.433 | 0.595 |
| test | 0 | slang | gofud biasa masuk mana? | Ke kantor biasanya naik KRL dari Bekasi | 0.803 | 0.299 | 0.299 | 0.378 | 0.386 | 0.280 | 0.289 | 0.312 | 0.388 | 0.674 |
| dev | 0 | easy | aku nabung buat apa ya? | Makan siang kantor rata-rata Rp35 ribu | 0.818 | 0.373 | 0.373 | 0.284 | 0.239 | 0.302 | 0.325 | 0.278 | 0.312 | 0.645 |
| test | 0 | hard | rencana liburan tahun depan? | Target tabungan: dana darurat Rp30 juta akhir tahun | 0.831 | 0.504 | 0.504 | 0.423 | 0.400 | 0.333 | 0.311 | 0.361 | 0.331 | 0.722 |
| dev | 0 | hard | cicilan apa aja? | Rencana liburan ke Bali tahun depan, target Rp10 juta | 0.798 | 0.421 | 0.421 | 0.346 | 0.286 | 0.325 | 0.374 | 0.309 | 0.295 | 0.636 |
| test | 0 | hard | makan siang biasanya habis berapa? | Isi bensin Pertalite tiap hari Sabtu | 0.811 | 0.437 | 0.437 | 0.249 | 0.296 | 0.329 | 0.376 | 0.248 | 0.296 | 0.712 |
| dev | 0 | short | kopi | Ada cicilan motor Rp850 ribu per bulan sampai Desember 2027 | 0.783 | 0.313 | 0.313 | 0.163 | 0.284 | 0.245 | 0.231 | 0.186 | 0.230 | 0.629 |
| test | 0 | short | bensin | Langganan Netflix dan Spotify tiap bulan | 0.790 | 0.365 | 0.366 | 0.263 | 0.477 | 0.135 | 0.131 | 0.300 | 0.341 | 0.604 |
| dev | 0 | easy | belanja bulanan dimana? | Rencana liburan ke Bali tahun depan, target Rp10 juta | 0.818 | 0.466 | 0.466 | 0.401 | 0.344 | 0.406 | 0.443 | 0.391 | 0.399 | 0.662 |
| test | 0 | chat | halo | Langganan kopi di Fore Coffee | 0.784 | 0.433 | 0.433 | 0.231 | 0.428 | 0.136 | 0.140 | 0.184 | 0.268 | 0.544 |
| dev | 0 | chat | makasih ya | Gaji masuk tanggal 25 setiap bulan | 0.834 | 0.312 | 0.312 | 0.366 | 0.433 | 0.291 | 0.295 | 0.292 | 0.339 | 0.693 |
| test | 0 | chat | cuaca hari ini gimana? | Budget makan bulanan Rp1,5 juta | 0.805 | 0.383 | 0.383 | 0.299 | 0.260 | 0.150 | 0.222 | 0.219 | 0.293 | 0.642 |
| dev | 0 | chat | siapa presiden indonesia sekarang? | Ke kantor biasanya naik KRL dari Bekasi | 0.772 | 0.399 | 0.399 | 0.278 | 0.264 | 0.243 | 0.264 | 0.245 | 0.277 | 0.724 |

Dedupe (a <-> b)

| split | label | tag | a | b | e5 | bge | api-bge-m3 | api-qwen3 | api-qwen3-noinst | api-3small | api-3small-384 | api-3large | api-3large-384 | api-mistral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | dup | paraphrase | Langganan kopi di Fore Coffee | Biasa ngopi di Fore Coffee | 0.968 | 0.884 | 0.884 | 0.827 | 0.827 | 0.725 | 0.718 | 0.810 | 0.813 | 0.806 |
| test | dup | paraphrase | Gaji masuk tanggal 25 setiap bulan | Tiap tanggal 25 aku gajian | 0.915 | 0.905 | 0.905 | 0.665 | 0.665 | 0.670 | 0.683 | 0.800 | 0.838 | 0.854 |
| dev | dup | paraphrase | Gojek dicatat sebagai kategori Transport | Gojek = kategori Transport | 0.955 | 0.766 | 0.766 | 0.852 | 0.852 | 0.939 | 0.939 | 0.862 | 0.873 | 0.953 |
| test | dup | paraphrase | Budget makan bulanan Rp1,5 juta | Budget makan per bulan 1,5 juta | 0.988 | 0.966 | 0.966 | 0.901 | 0.901 | 0.928 | 0.933 | 0.919 | 0.929 | 0.961 |
| dev | dup | paraphrase | Bayar kos tiap tanggal 1, Rp1,8 juta | Kos Rp1,8 juta dibayar setiap tanggal 1 | 0.978 | 0.935 | 0.935 | 0.855 | 0.855 | 0.860 | 0.877 | 0.820 | 0.830 | 0.931 |
| test | dup | paraphrase | Ke kantor biasanya naik KRL dari Bekasi | Berangkat kerja pakai KRL dari Bekasi | 0.938 | 0.900 | 0.900 | 0.829 | 0.829 | 0.767 | 0.779 | 0.760 | 0.803 | 0.899 |
| dev | dup | paraphrase | Target dana darurat Rp30 juta akhir tahun | Mau punya dana darurat 30 juta di akhir tahun | 0.982 | 0.833 | 0.833 | 0.876 | 0.876 | 0.795 | 0.796 | 0.787 | 0.804 | 0.889 |
| test | dup | paraphrase | Langganan Netflix dan Spotify tiap bulan | Aku langganan Spotify sama Netflix | 0.907 | 0.799 | 0.799 | 0.872 | 0.872 | 0.767 | 0.786 | 0.790 | 0.829 | 0.910 |
| dev | dup | slang | GoFood dicatat sebagai kategori Makan | gofud itu masuk makan | 0.912 | 0.643 | 0.644 | 0.674 | 0.674 | 0.517 | 0.524 | 0.504 | 0.550 | 0.776 |
| test | dup | slang | Suka laporan pengeluaran ringkas per minggu | kasih laporan singkat tiap minggu ya | 0.941 | 0.820 | 0.820 | 0.725 | 0.725 | 0.738 | 0.736 | 0.679 | 0.713 | 0.832 |
| dev | dup | slang | Isi bensin Pertalite tiap hari Sabtu | tiap sabtu aku isi pertalite | 0.926 | 0.820 | 0.820 | 0.650 | 0.650 | 0.597 | 0.607 | 0.801 | 0.816 | 0.832 |
| test | dup | slang | Ada cicilan motor Rp850 ribu per bulan | Bayar cicilan motor 850rb sebulan | 0.949 | 0.942 | 0.942 | 0.829 | 0.829 | 0.882 | 0.893 | 0.874 | 0.894 | 0.885 |
| dev | distinct | merchant | Langganan kopi di Fore Coffee | Langganan kopi di Starbucks | 0.942 | 0.746 | 0.746 | 0.781 | 0.781 | 0.763 | 0.749 | 0.713 | 0.730 | 0.843 |
| test | distinct | category | Budget makan bulanan Rp1,5 juta | Budget transport bulanan Rp1,5 juta | 0.963 | 0.837 | 0.836 | 0.737 | 0.737 | 0.776 | 0.785 | 0.802 | 0.798 | 0.878 |
| dev | distinct | category | Gojek dicatat sebagai kategori Transport | GoFood dicatat sebagai kategori Makan | 0.942 | 0.672 | 0.672 | 0.544 | 0.544 | 0.614 | 0.654 | 0.681 | 0.678 | 0.857 |
| test | distinct | merchant | Gojek dicatat sebagai kategori Transport | Grab dicatat sebagai kategori Transport | 0.979 | 0.818 | 0.817 | 0.757 | 0.757 | 0.745 | 0.754 | 0.787 | 0.794 | 0.889 |
| dev | distinct | date | Gaji masuk tanggal 25 setiap bulan | Bayar kos tiap tanggal 25 | 0.940 | 0.842 | 0.842 | 0.658 | 0.658 | 0.737 | 0.767 | 0.736 | 0.765 | 0.839 |
| test | distinct | date | Bayar kos tiap tanggal 1, Rp1,8 juta | Bayar listrik tiap tanggal 1 | 0.911 | 0.715 | 0.715 | 0.637 | 0.637 | 0.694 | 0.721 | 0.677 | 0.737 | 0.889 |
| dev | distinct | item | Ada cicilan motor Rp850 ribu per bulan | Ada cicilan laptop Rp850 ribu per bulan | 0.972 | 0.865 | 0.865 | 0.792 | 0.792 | 0.850 | 0.856 | 0.814 | 0.829 | 0.907 |
| test | distinct | item | Langganan Netflix tiap bulan | Langganan Spotify tiap bulan | 0.948 | 0.825 | 0.826 | 0.752 | 0.752 | 0.783 | 0.801 | 0.819 | 0.823 | 0.892 |
| dev | distinct | item | Ke kantor biasanya naik KRL dari Bekasi | Pulang kantor biasanya naik ojek online | 0.902 | 0.678 | 0.678 | 0.571 | 0.571 | 0.644 | 0.665 | 0.588 | 0.594 | 0.849 |
| test | distinct | item | Target dana darurat Rp30 juta akhir tahun | Target liburan ke Bali Rp10 juta tahun depan | 0.919 | 0.579 | 0.579 | 0.506 | 0.506 | 0.569 | 0.611 | 0.565 | 0.596 | 0.868 |
| dev | distinct | unrelated | Langganan kopi di Fore Coffee | Gaji masuk tanggal 25 setiap bulan | 0.834 | 0.405 | 0.405 | 0.332 | 0.332 | 0.227 | 0.263 | 0.293 | 0.365 | 0.631 |
| test | distinct | unrelated | Suka laporan pengeluaran ringkas per minggu | Isi bensin Pertalite tiap hari Sabtu | 0.884 | 0.492 | 0.492 | 0.431 | 0.431 | 0.420 | 0.465 | 0.349 | 0.454 | 0.702 |
| dev | distinct | unrelated | Makan siang kantor rata-rata Rp35 ribu | Belanja bulanan di Indomaret dekat rumah | 0.877 | 0.418 | 0.418 | 0.235 | 0.235 | 0.426 | 0.509 | 0.405 | 0.490 | 0.697 |
| n/a | supersede | update | Budget makan bulanan Rp1,5 juta | Budget makan bulanan Rp2 juta | 0.973 | 0.894 | 0.894 | 0.937 | 0.937 | 0.910 | 0.915 | 0.894 | 0.893 | 0.944 |
| n/a | supersede | update | Gaji masuk tanggal 25 setiap bulan | Gaji sekarang masuk tanggal 28 setiap bulan | 0.961 | 0.846 | 0.846 | 0.808 | 0.808 | 0.834 | 0.854 | 0.836 | 0.873 | 0.929 |
| n/a | supersede | update | Bayar kos tiap tanggal 1, Rp1,8 juta | Bayar kos tiap tanggal 1, naik jadi Rp2 juta | 0.952 | 0.845 | 0.845 | 0.880 | 0.880 | 0.897 | 0.902 | 0.821 | 0.835 | 0.960 |
| n/a | supersede | update | Target dana darurat Rp30 juta akhir tahun | Target dana darurat naik jadi Rp50 juta akhir tahun | 0.956 | 0.828 | 0.828 | 0.867 | 0.867 | 0.896 | 0.902 | 0.897 | 0.911 | 0.960 |
| n/a | supersede | update | Langganan kopi di Fore Coffee | Sekarang pindah langganan kopi ke Kopi Kenangan | 0.918 | 0.682 | 0.682 | 0.586 | 0.586 | 0.653 | 0.674 | 0.584 | 0.603 | 0.771 |
