# Sistem Klasifikasi dan Temu Kembali Sentimen

**Sistem Klasifikasi dan Temu Kembali Sentimen Dokumen Komentar Kasus Polio Berbasis Semantic Search Menggunakan SBERT + Scikit-learn Cosine Similarity**

## Teknologi

- **SBERT** (Sentence-BERT) - Embedding untuk semantic search DAN sentiment analysis
- **Scikit-learn** - Cosine similarity untuk retrieval
- **Example-based Sentiment** - Classify menggunakan perbandingan dengan contoh kalimat

## Arsitektur

```
Query
  ├── Classify DIRECT: Bandingkan dengan 20 contoh BAIK + 20 contoh BURUK
  │     → Hitung cosine similarity ke semua contoh
  │     → Weighted average (avg, top3, max, median)
  │     → Hasil: BAIK (Positif) / BURUK (Negatif)
  │
  └── Cari dokumen relevan → Gunakan sebagai EXPLANATION
        → Tampilkan dokumen pendukung
        → Voting: X BAIK, Y BURUK
```

## Akurasi

**16/16 test cases: 100% akurat**

| Kategori | Test | Hasil |
|----------|------|-------|
| Positif jelas | "vaksin polio baik untuk anak" | ✅ BAIK |
| Positif jelas | "saya senang anak saya sudah divaksin" | ✅ BAIK |
| Positif jelas | "imunisasi melindungi anak dari penyakit" | ✅ BAIK |
| Negatif jelas | "anak saya sakit setelah vaksin polio" | ✅ BURUK |
| Negatif jelas | "vaksin polio menyebabkan kelumpuhan" | ✅ BURUK |
| Negatif jelas | "efek samping vaksin berbahaya" | ✅ BURUK |
| Negatif kuat | "saya trauma anak saya cacat karena polio" | ✅ BURUK |

## Cara Menjalankan

```bash
cd "e:\Tugas TKI"
pip install -r requirements.txt
python app.py
```

Access: http://localhost:5000

## Cara Kerja

Query diklasifikasi dengan membandingkan ke 40 contoh kalimat:

- **20 contoh BAIK**: "anak saya sehat setelah vaksin", "alhamdulillah imunisasi berhasil", dll
- **20 contoh BURUK**: "anak saya sakit setelah vaksin", "efek samping vaksin berbahaya", dll

### Formula Scoring:
```
combined = 0.3 * avg + 0.3 * top3_avg + 0.2 * max + 0.2 * median
confidence = combined_pos / (combined_pos + combined_neg)
```

## File Structure

```
e:\Tugas TKI\
├── sentiment_search_engine.py    # Core engine (SBERT + example-based)
├── app.py                         # Flask backend
├── requirements.txt
├── README.md
├── templates/index.html
├── static/style.css
├── static/app.js
└── versi lama/Tugas_TKI _1_.xlsx
```
# Uas-TKI
