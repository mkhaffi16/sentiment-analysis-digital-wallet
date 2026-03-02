# 🎯 Analisis Sentimen Berbasis Aspek pada Review Aplikasi Digital Wallet
### Menggunakan Metode Extreme Gradient Boosting (XGBoost)
## 📌 Deskripsi Proyek

Penelitian ini membangun sistem **Aspect-Based Sentiment Analysis (ABSA)** untuk menganalisis ulasan pengguna aplikasi digital wallet di Indonesia. Sistem ini mampu mengidentifikasi **aspek-aspek spesifik** dalam sebuah ulasan (seperti fitur, antarmuka, keamanan, layanan) beserta **polaritas sentimen** (positif, negatif, netral) pada masing-masing aspek.

Model klasifikasi dibangun menggunakan algoritma **Extreme Gradient Boosting (XGBoost)** yang dipadukan dengan teknik ekstraksi fitur teks.

> 📄 Dokumen skripsi lengkap tersedia di file SKRIPSI_181402102.pdf

---

## 🔍 Latar Belakang

Industri fintech di Indonesia berkembang pesat, dengan jutaan ulasan pengguna tersebar di Google Play Store. Analisis sentimen berbasis aspek memungkinkan pengembang aplikasi memahami **apa yang disukai dan tidak disukai pengguna secara spesifik**, bukan sekadar rating bintang.

---

## 🧠 Metodologi

```
Data Collection → Preprocessing → Feature Extraction → Modeling → Evaluation
     ↓                ↓                  ↓                ↓            ↓
Google Play      Cleaning, Case          Word2Vec       XGBoost    Accuracy,
Store Reviews    Folding, Stopword                     Classifier  F1-Score,
                 Removal, Stemming,                                Precision,
                 normalization,                                     Recall
```              punctual removal

### Tahapan Penelitian:
1. **Pengumpulan Data** — Scraping review dari Google Play Store
2. **Preprocessing** — Cleaning, normalisasi teks, stopword removal, stemming (Sastrawi)
3. **Aspect Extraction** — Aspek diekstrak menggunakan LDA (Latent Dirichlet Allocation)
4. **Feature Engineering** — Word2Vec
5. **Modeling** — Pelatihan model XGBoost
6. **Evaluasi** — Pengukuran performa dengan confusion matrix & classification report

---

## 📊 Hasil & Performa Model

| Metric | Score |
|--------|-------|
| Accuracy | 90% |
| Precision | bisa dilihat di dokumen skripsi |
| Recall | bisa dilihat di dokumen skripsi |
| F1-Score | bisa dilihat di dokumen skripsi |


## 📂 Dataset

Dataset berisi ulasan pengguna aplikasi digital wallet yang dikumpulkan dari **Google Play Store**. Dataset telah dilabeli secara manual dengan aspek dan polaritas sentimen.


## 👤 Author

**[Muhammad Khaffi Irwan]**
- 🎓 [Universitas Sumatera Utara] — [Teknologi Informasi], [2023]
- 💼 [LinkedIn](https://linkedin.com/in/muhammad-khaffi)
- 📧 mkhaffi16@gmail.com

---

## 📜 Lisensi

Proyek ini dibuat untuk keperluan akademik. Silakan hubungi penulis jika ingin menggunakan atau mengembangkan lebih lanjut.



*⭐ Jika repositori ini bermanfaat, jangan lupa beri bintang!*
