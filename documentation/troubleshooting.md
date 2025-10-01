# 🔧 Troubleshooting Trading Predictor

## 🚨 Masalah Umum & Solusi

### 1. Environment Issues

#### ❌ `python command not found`
**Gejala:** Tidak bisa menjalankan python
**Solusi:**
```bash
source ~/miniconda3/bin/activate && conda activate projects
```

#### ❌ `conda: command not found`
**Gejala:** Conda tidak terdeteksi
**Solusi:**
```bash
# Tambahkan conda ke PATH
export PATH="~/miniconda3/bin:$PATH"
source ~/.bashrc
```

#### ❌ `Environment 'projects' not found`
**Gejala:** Environment conda tidak ada
**Solusi:**
```bash
# Buat environment baru
conda create -n projects python=3.12
conda activate projects
pip install -r requirements.txt
```

---

### 2. Database Issues

#### ❌ `Database not found`
**Gejala:** File database sqlite tidak ditemukan
**Solusi:**
```bash
# Auto-create database
python utils/auto_pipeline.py

# Manual create untuk pair spesifik
python pipeline/build_dataset.py --config config/config.yaml --months 6
```

#### ❌ `Database locked`
**Gejala:** Database sedang digunakan proses lain
**Solusi:**
```bash
# Cek proses yang menggunakan database
lsof data/db/*.sqlite

# Kill proses jika perlu
pkill -f "python.*predict"

# Restart proses
python scripts/predict_enhanced.py --timeframe 15m --model auto --single
```

#### ❌ `Corrupted database`
**Gejala:** Error saat baca database
**Solusi:**
```bash
# Backup database lama
mv data/db/btc_15m.sqlite data/db/btc_15m.sqlite.backup

# Rebuild database
python utils/auto_pipeline.py --force-regenerate
```

---

### 3. Model Issues

#### ❌ `Model not found`
**Gejala:** File model tidak ditemukan
**Solusi:**
```bash
# Train model baru
python scripts/train_tree_model.py --model lightgbm --target direction_h1

# Cek model yang tersedia
ls -la models/tree_models/
```

#### ❌ `Model loading failed`
**Gejala:** Error saat load model
**Solusi:**
```bash
# Hapus model rusak
rm models/tree_models/btc_15m_*.pkl

# Train ulang
python scripts/train_tree_model.py --model lightgbm --target direction_h1
```

#### ❌ `Low model accuracy`
**Gejala:** Akurasi model rendah (< 30%)
**Solusi:**
```bash
# Re-optimize threshold
python utils/auto_pipeline.py --force-regenerate

# Train dengan data lebih banyak
python utils/auto_pipeline.py --months 12

# Cek distribusi target
python utils/system_validator.py
```

---

### 4. Prediction Issues

#### ❌ `No predictions generated`
**Gejala:** Script jalan tapi tidak ada output
**Solusi:**
```bash
# Cek confidence threshold
python scripts/predict_enhanced.py --timeframe 15m --model auto --confidence-threshold 0.5

# Cek data terbaru
python scripts/predict_enhanced.py --timeframe 15m --model auto --single --verbose
```

#### ❌ `Prediction confidence too low`
**Gejala:** Semua prediksi ditolak karena confidence rendah
**Solusi:**
```bash
# Turunkan threshold
python scripts/predict_enhanced.py --timeframe 15m --model auto --confidence-threshold 0.3

# Re-train model
python scripts/train_tree_model.py --model lightgbm --target direction_h1
```

---

### 5. Performance Issues

#### ❌ `Out of memory`
**Gejala:** Sistem kehabisan RAM
**Solusi:**
```bash
# Kurangi data history
python utils/auto_pipeline.py --months 6

# Bersihkan cache
python utils/auto_clean.py --cache-only

# Monitor memory
htop
```

#### ❌ `Training too slow`
**Gejala:** Training model memakan waktu lama
**Solusi:**
```bash
# Gunakan distributed training
python scripts/train_tree_model_ray.py --model lightgbm

# Kurangi data
python scripts/train_tree_model.py --model lightgbm --months 3
```

#### ❌ `High CPU usage`
**Gejala:** CPU usage 100%
**Solusi:**
```bash
# Kurangi interval prediksi
python scripts/predict_enhanced.py --interval 600  # 10 menit

# Limit CPU cores
export OMP_NUM_THREADS=2
```

---

### 6. Configuration Issues

#### ❌ `Config file not found`
**Gejala:** Error config.yaml tidak ditemukan
**Solusi:**
```bash
# Cek file config
ls -la config/config.yaml

# Copy dari template jika tidak ada
cp config/config_template.yaml config/config.yaml
```

#### ❌ `Invalid YAML syntax`
**Gejala:** Error parsing config.yaml
**Solusi:**
```bash
# Validasi YAML
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Fix indentasi dan syntax
```

#### ❌ `Threshold config not found`
**Gejala:** auto_balanced_thresholds.json tidak ada
**Solusi:**
```bash
# Generate threshold config
python utils/auto_pipeline.py --validate-only
```

---

### 7. Network Issues

#### ❌ `Connection timeout`
**Gejala:** Gagal download data dari Binance
**Solusi:**
```bash
# Cek koneksi internet
ping -c 4 api.binance.com

# Retry dengan timeout lebih besar
python data/binance_fetch.py --timeout 60

# Gunakan proxy jika perlu
export https_proxy=http://proxy:8080
```

---

### 8. Log & Monitoring Issues

#### ❌ `Log files too large`
**Gejala:** Log memenuhi disk
**Solusi:**
```bash
# Bersihkan log lama
python utils/auto_clean.py --logs-only

# Rotate log manual
find data/logs/ -name "*.log" -mtime +7 -delete
```

#### ❌ `No log output`
**Gejala:** Tidak ada log yang ter-generate
**Solusi:**
```bash
# Cek permission folder log
chmod 755 data/logs/

# Manual log check
python scripts/predict_enhanced.py --timeframe 15m --model auto --single --verbose
```

---

## 🔍 Debug Commands

### System Health Check
```bash
# Quick check
python utils/system_validator.py --quick

# Detailed check
python utils/system_validator.py

# Check specific component
python utils/system_validator.py --component database
```

### Environment Verification
```bash
# Check Python
which python
python --version

# Check packages
python -c "import pandas, numpy, lightgbm; print('All packages OK')"

# Check conda env
conda info --envs
```

### Database Inspection
```bash
# Check database size
ls -lh data/db/*.sqlite

# Count records
sqlite3 data/db/btc_15m.sqlite "SELECT COUNT(*) FROM features;"

# Check latest data
sqlite3 data/db/btc_15m.sqlite "SELECT timestamp FROM features ORDER BY timestamp DESC LIMIT 5;"
```

### Model Verification
```bash
# List models
ls -la models/tree_models/

# Check model size
ls -lh models/tree_models/*.pkl

# Test model loading
python -c "import pickle; print('Model OK' if pickle.load(open('models/tree_models/btc_15m_direction_h1.pkl', 'rb')) else 'Model Error')"
```

---

## 🛡️ Emergency Recovery

### Complete System Reset
```bash
# Backup important files
cp -r config/ config_backup/
cp -r data/auto_balanced_thresholds.json data/threshold_backup.json

# Clean everything
python utils/auto_clean.py --all --force

# Rebuild from scratch
python utils/auto_pipeline.py

# Retrain models
python scripts/train_tree_model.py --model lightgbm --target direction_h1
```

### Quick Recovery
```bash
# Stop all processes
pkill -f "python.*predict"

# Clean cache only
python utils/auto_clean.py --cache-only --force

# Restart prediction
python scripts/predict_enhanced.py --timeframe 15m --model auto --single
```

---

## 📊 Monitoring Commands

### Resource Usage
```bash
# Memory usage
free -h

# Disk usage
df -h

# CPU usage
top -p $(pgrep -f python)
```

### Process Monitoring
```bash
# Active Python processes
ps aux | grep python

# Prediction processes
ps aux | grep predict

# Log tail
tail -f data/logs/production_*.log
```

---

## 📞 Getting Help

### Debug Information
Saat meminta bantuan, sertakan informasi berikut:

```bash
# System info
uname -a
python --version
conda --version

# Error logs
tail -50 data/logs/production_*.log

# System health
python utils/system_validator.py --quick
```

### Common Questions

**Q: Mengapa akurasi model rendah?**
A: Coba re-optimize threshold dengan `python utils/auto_pipeline.py --force-regenerate`

**Q: Prediksi tidak muncul?**
A: Cek confidence threshold, mungkin terlalu tinggi

**Q: Sistem lambat?**
A: Kurangi interval prediksi atau data history

**Q: Database error?**
A: Rebuild database dengan `python utils/auto_pipeline.py`

---

## 💡 Prevention Tips

1. **Regular maintenance**: Jalankan `python utils/auto_clean.py` mingguan
2. **Monitor resources**: Cek disk/memory usage berkala  
3. **Backup configs**: Simpan konfigurasi yang sudah optimal
4. **Update models**: Retrain model bulanan
5. **Health checks**: Jalankan system validator rutin