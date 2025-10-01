# 🚀 Panduan Penggunaan Trading Predictor

## ⚡ Setup Cepat (3 Langkah)

```bash
# 1. Aktivasi environment (SELALU lakukan ini terlebih dahulu)
source ~/miniconda3/bin/activate && conda activate projects

# 2. Jalankan setup otomatis lengkap
python utils/auto_pipeline.py

# 3. Mulai prediksi
python scripts/predict_enhanced.py --timeframe 15m --model auto --single --output json
```

**Selesai!** Sistem Anda sekarang sudah otomatis dari pengumpulan data hingga prediksi.

---

## 📋 Penggunaan Harian

### Setup Pagi (Pertama Kali atau Mingguan)
```bash
source ~/miniconda3/bin/activate && conda activate projects
python utils/auto_pipeline.py
```

### Prediksi Cepat (Sepanjang Hari)
```bash
source ~/miniconda3/bin/activate && conda activate projects
python scripts/predict_enhanced.py --timeframe 15m --model auto --single --output json
```

### Mode Produksi (Biarkan Berjalan)
```bash
source ~/miniconda3/bin/activate && conda activate projects
nohup python scripts/predict_enhanced.py --timeframe 15m --model auto --interval 300 --continuous \
> data/logs/production_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 🔍 Cek Status Sistem

### Cek Kesehatan Cepat
```bash
source ~/miniconda3/bin/activate && conda activate projects
python utils/system_validator.py --quick
```

### Laporan Detail
```bash
source ~/miniconda3/bin/activate && conda activate projects
python utils/system_validator.py
```

### Lihat Prediksi Terbaru
```bash
tail -f data/logs/production_*.log
```

---

## 🛠️ Tugas Umum

### Update Data (Bulanan)
```bash
source ~/miniconda3/bin/activate && conda activate projects
python utils/auto_pipeline.py --months 13 --force-regenerate
```

### Latih Ulang Model (Mingguan)
```bash
source ~/miniconda3/bin/activate && conda activate projects
python scripts/train_tree_model.py --model lightgbm --target direction_h1
```

### Bersihkan File Lama
```bash
source ~/miniconda3/bin/activate && conda activate projects
python utils/auto_clean.py --logs-only --cache-only
```

---

## ⚙️ Kustomisasi

### Ubah Pair/Timeframe
Edit `config/config.yaml`:
```yaml
pairs: [BTCUSDT, ETHUSDT, DOGEUSDT]
timeframes: [15m, 1h, 4h]
```

### Sesuaikan Threshold Prediksi
```bash
# Confidence tinggi (prediksi sedikit tapi akurat)
python scripts/predict_enhanced.py --timeframe 15m --model auto --confidence-threshold 0.8

# Confidence rendah (prediksi lebih banyak, kurang akurat)
python scripts/predict_enhanced.py --timeframe 15m --model auto --confidence-threshold 0.6
```

---

## 🚨 Mengatasi Masalah

| Masalah | Solusi |
|---------|----------|
| `python command not found` | `source ~/miniconda3/bin/activate && conda activate projects` |
| `database not found` | `python utils/auto_pipeline.py` |
| `model not found` | `python scripts/train_tree_model.py --model lightgbm --target direction_h1` |
| Akurasi rendah | `python utils/auto_pipeline.py --force-regenerate` |
| Kehabisan memori | Kurangi parameter `--months` atau tambah RAM |

---

## 📊 Memahami Hasil

### Output Prediksi
```json
{
  "timestamp": "2025-10-01T10:30:00Z",
  "pair": "BTCUSDT",
  "timeframe": "15m",
  "prediction": "BUY",
  "confidence": 0.408,
  "threshold_used": 0.15
}
```

### Laporan Kesehatan
```
🏆 LAPORAN KESEHATAN SISTEM
Kesehatan Keseluruhan: 🟢 EXCELLENT (95.2/100)
   🗄️  Database: 3 ✅
   🧠 Model: 2 ✅
   📈 Akurasi Rata-rata: 38.36% ✅
   🔮 Prediksi: ✅ Berjalan
```

---

## 📞 Butuh Bantuan?

1. **Cek log**: `tail -f data/logs/production_*.log`
2. **Jalankan validator**: `python utils/system_validator.py`
3. **Bersihkan dan restart**: `python utils/auto_clean.py --all --force && python utils/auto_pipeline.py`

**Pro tip:** Selalu aktivasi environment terlebih dahulu: `source ~/miniconda3/bin/activate && conda activate projects`

---

**Selamat Trading!** 🎯📈