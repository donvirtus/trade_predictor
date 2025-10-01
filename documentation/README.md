# 📚 Dokumentasi Trading Predictor

Folder ini berisi dokumentasi lengkap untuk sistem Trading Predictor.

## 📋 Daftar Isi

- `penggunaan.md` - Panduan penggunaan sistem
- `konfigurasi.md` - Penjelasan konfigurasi sistem
- `troubleshooting.md` - Panduan mengatasi masalah

## 🚀 Cara Penggunaan Cepat

```bash
# Aktivasi environment
source ~/miniconda3/bin/activate && conda activate projects

# Jalankan pipeline otomatis
python utils/auto_pipeline.py

# Jalankan prediksi
python scripts/predict_enhanced.py --timeframe 15m --model auto --single --output json
```

## 📞 Bantuan

Untuk bantuan lebih lanjut, periksa file log di `data/logs/` atau jalankan validator sistem:

```bash
python utils/system_validator.py
```