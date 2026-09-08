# tools/ — Modul 03

Gate kualitas Modul 03. Semua skrip dijalankan dari direktori ini.

| Skrip | Gunanya |
|---|---|
| `bahasa_rules.py` | Daftar kata terlarang + pembatas pembuka kalimat; dipakai keempat validator |
| `validate_notebooks.py` | Struktur 2 notebook: marker wajib, nol output tersimpan, `pip` wajib ter-pin, Latihan tepat sebelum Ringkasan |
| `validate_slides.py` | Deck: 60–80 frame, ≥8 `\intuisi`, log bersih (0 error, 0 overfull), jumlah entri notes = jumlah halaman deck |
| `validate_quiz.py` | Quiz: 28–34 soal konsep murni, 4 opsi, panjang opsi seimbang, cakupan konsep M03 |
| `validate_cheatsheet.py` | Cheatsheet: ≥12 kartu, istilah wajib, PDF 1 halaman dan tidak lebih tua dari HTML |
| `build_quiz.py` | Membangun ulang `../nlp-fundamentals-quiz.html` dari daftar soal di dalamnya |
| `nbedit.py` | Helper edit `.ipynb` programatik (`load`/`find`/`replace`/`save`) |
| `run_nb.sh` | Eksekusi notebook penuh; output ke luar repo |

## Menjalankan semuanya

```bash
for v in validate_notebooks validate_slides validate_quiz validate_cheatsheet; do
    python3 $v.py || echo "GAGAL: $v"
done
```

`M03_DECK_PARTIAL=1` menurunkan cek jumlah frame/intuisi/notes jadi peringatan,
untuk dipakai saat deck sedang dalam pengerjaan.

## Regenerasi artefak

```bash
python3 build_quiz.py                 # quiz HTML
cd ../slides && ./build.sh            # deck PDF + speaker notes PDF
# cheatsheet PDF (setelah mengedit HTML-nya):
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=nlp-fundamentals-cheatsheet.pdf \
  file://$PWD/nlp-fundamentals-cheatsheet.html
```
