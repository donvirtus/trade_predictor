# ⚙️ Konfigurasi Trading Predictor

## 📄 File Konfigurasi Utama

### `config/config.yaml` - Konfigurasi Utama

```yaml
# Pair dan timeframe trading
pairs: [BTCUSDT, ETHUSDT, DOGEUSDT]
timeframes: [5m, 15m, 30m, 1h, 2h, 4h, 6h]

# Pengaturan target
target:
  horizon: 20                    # Horizon prediksi
  threshold_mode: "auto_balanced"  # auto_balanced | manual
  auto_balanced_config: "data/auto_balanced_thresholds.json"

# Analisis multi-horizon
multi_horizon:
  horizons: [1, 5, 10, 20]      # Horizon analisis
  enable_extremes: true          # Prediksi high/low masa depan

# Auto-balance distribusi target
auto_balance:
  target_distribution:
    UP: 33.0         # Target 33% sinyal UP
    SIDEWAYS: 34.0   # Target 34% SIDEWAYS
    DOWN: 33.0       # Target 33% sinyal DOWN
```

### `data/auto_balanced_thresholds.json` - Threshold Otomatis

```json
{
  "individual_results": {
    "BTC 15M": {
      "optimal_threshold": 0.15,
      "balance_score": 93.5,
      "distribution": {"UP": 33.2, "SIDEWAYS": 33.8, "DOWN": 33.0}
    }
  },
  "average_optimal_threshold": 0.15,
  "recommendation": {
    "threshold": 0.15,
    "description": "Auto-generated optimal threshold untuk distribusi seimbang"
  }
}
```

---

## 🔧 Pengaturan Sistem

### Database
- **Lokasi**: `data/db/`
- **Format**: SQLite
- **Penamaan**: `{pair}_{timeframe}.sqlite` (contoh: `btc_15m.sqlite`)

### Model
- **Lokasi**: `models/tree_models/`
- **Tipe**: LightGBM (rekomendasi)
- **Penamaan**: `{pair}_{timeframe}_{target}.pkl`

### Log
- **Lokasi**: `data/logs/`
- **Format**: Text file dengan timestamp
- **Rotasi**: Manual (gunakan `utils/auto_clean.py`)

---

## 📊 Parameter Kustomisasi

### 1. Trading Pairs
```yaml
# Tambah/kurangi pair sesuai kebutuhan
pairs: 
  - BTCUSDT
  - ETHUSDT  
  - DOGEUSDT
  - ADAUSDT   # Tambah pair baru
```

### 2. Timeframes
```yaml
# Pilih timeframe yang diinginkan
timeframes: 
  - 5m     # Scalping
  - 15m    # Swing trading
  - 1h     # Position trading
  - 4h     # Long term
```

### 3. Horizon Prediksi
```yaml
target:
  horizon: 20    # Prediksi 20 candle ke depan
  
multi_horizon:
  horizons: [1, 5, 10, 20]  # Analisis berbagai horizon
```

### 4. Threshold Mode
```yaml
target:
  threshold_mode: "auto_balanced"  # Otomatis (rekomendasi)
  # threshold_mode: "manual"       # Manual
```

---

## 🎯 Optimasi Performa

### 1. Memory Usage
```yaml
# Untuk sistem dengan RAM terbatas
data_processing:
  chunk_size: 1000      # Kurangi jika RAM terbatas
  max_history: 10000    # Batasi data history
```

### 2. Training Speed
```yaml
model:
  n_estimators: 100     # Kurangi untuk training cepat
  max_depth: 6          # Batasi kedalaman tree
  num_leaves: 31        # Kontrol kompleksitas
```

### 3. Prediction Frequency
```bash
# Interval prediksi (detik)
--interval 300    # 5 menit
--interval 900    # 15 menit
--interval 3600   # 1 jam
```

---

## 🔄 Environment Variables

### Conda Environment
```bash
# Setup environment
export CONDA_ENV="projects"
export PYTHON_PATH="~/miniconda3/envs/projects/bin/python"
```

### API Keys (jika diperlukan)
```bash
# Binance API (opsional)
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"
```

---

## 📈 Monitoring Configuration

### System Health
```yaml
monitoring:
  health_check_interval: 3600  # 1 jam
  log_rotation_days: 7         # Rotasi log 7 hari
  backup_retention_days: 30    # Simpan backup 30 hari
```

### Alerts
```yaml
alerts:
  low_accuracy_threshold: 0.25  # Alert jika akurasi < 25%
  memory_usage_threshold: 0.85  # Alert jika RAM > 85%
  disk_usage_threshold: 0.90    # Alert jika disk > 90%
```

---

## 🛠️ Advanced Settings

### Feature Engineering
```yaml
features:
  technical_indicators: true
  market_structure: true
  volatility_measures: true
  volume_analysis: true
  custom_features: []
```

### Model Ensemble
```yaml
ensemble:
  enable: false
  models: ["lightgbm", "xgboost", "catboost"]
  voting_strategy: "soft"
  weights: [0.5, 0.3, 0.2]
```

---

## 📞 Troubleshooting Konfigurasi

### Error: Config Not Found
```bash
# Pastikan file config.yaml ada
ls -la config/config.yaml

# Jika tidak ada, copy dari template
cp config/config_template.yaml config/config.yaml
```

### Error: Invalid YAML
```bash
# Validasi syntax YAML
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

### Error: Permission Denied
```bash
# Fix permission
chmod 644 config/config.yaml
chmod 755 data/db/
```

---

## 📝 Best Practices

1. **Backup konfigurasi** sebelum perubahan besar
2. **Test konfigurasi** dengan dataset kecil terlebih dahulu
3. **Monitor performa** setelah perubahan konfigurasi
4. **Update threshold** secara berkala untuk akurasi optimal
5. **Validasi konfigurasi** dengan `utils/system_validator.py`

---

## 💡 Tips Optimasi

- **Gunakan auto_balanced** untuk threshold optimal
- **Monitor distribusi target** untuk deteksi drift
- **Sesuaikan horizon** berdasarkan strategi trading
- **Update model** secara berkala untuk performa terbaik
- **Backup konfigurasi** yang sudah optimal