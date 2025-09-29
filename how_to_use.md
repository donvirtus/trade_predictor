# Cara Menggunakan (End-to-End)

Panduan singkat untuk: koleksi data, training model multi-horizon + extremes, prediksi live, dan analisis paper-trade.

## Quickstart (perintah ringkas)

```bash
# 0) OPTIONAL: Optimize thresholds untuk perfect class balance (RECOMMENDED)
python3 utils/threshold_balancer.py  # Generate auto_balanced_thresholds.json

# 1) Koleksi data (sesuai config dengan optimal thresholds)
python3 pipeline/build_dataset.py --config config/config.yaml --timeframe config --months 1 --force-full-rebuild

# 1b) MULTI-ASSET: Koleksi multiple pairs sekaligus
python3 pipeline/build_dataset.py --pairs btcusdt ethusdt dogeusdt --timeframe 1h --months 1 --force-full-rebuild

# 2) Training semua timeframes dari config + semua direction horizons + extremes (artefak stabil)
python3 scripts/train_tree_model.py \
  --timeframe config --model LightGBM \
  --multi-horizon-all --include-extremes --overwrite \
  --config config/config.yaml

# 3) Prediksi sekali (15m)
python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --single

# 4) Prediksi kontinu (5m, futures, prefer RR, paper-trade CSV)
python3 scripts/predict_enhanced.py --timeframe 5m --model LightGBM --continuous \
  --interval 60 --market future --prefer-rr-mode --rr-min 1.2 \
  --paper-trade-log data/logs/paper_trades_5m.csv --paper-timeout-mins 150

# 5) Analisis hasil paper-trade CSV
python3 scripts/analyze_paper_trades.py --log data/logs/paper_trades_5m.csv \
  --from "2025-09-01" --to "2025-09-27"
```

## Persiapan lingkungan (opsional)

```bash
conda create -n trade python=3.12 -y
conda activate trade
pip install -r requirements.txt
```

Minimal: pandas, numpy, scikit-learn, requests, pyyaml, ccxt, ta (TA-Lib opsional).

## Konfigurasi

Edit `config/config.yaml` (atau gunakan varian cepat `config/config_test_separate.yaml`).

- pairs: contoh ["BTCUSDT"]
- timeframes: contoh ["5m", "15m", "1h"]
- months/years: jendela historis
- **threshold_mode**: "manual" | "auto_balanced" | "adaptive" (RECOMMENDED: auto_balanced)
- multi_horizon: horizons, enable_extremes, rr_min
- data.market_type: spot atau future (default jika tidak override via CLI)

### 🎯 **Threshold Optimization (CRITICAL for ML Performance)**

**📚 COMPLETE GUIDE:** See [THRESHOLD_MODES_GUIDE.md](THRESHOLD_MODES_GUIDE.md) for detailed explanation of all threshold modes.

**RECOMMENDED:** Use `threshold_mode: "auto_balanced"` for optimal class distribution:

```yaml
target:
  threshold_mode: "auto_balanced"  # Perfect 33%/33%/33% balance
  auto_balanced_config: "data/auto_balanced_thresholds.json"
```

**Generate optimal thresholds:**
```bash
# Calculate optimal thresholds scientifically
python3 utils/threshold_balancer.py

# Validate threshold impact
python3 validate_threshold.py --asset btc --timeframe 1h --threshold-range 0.2,2.0,0.1
```

**Performance comparison:**
- Manual (1.0%): 37-38/100 balance score ❌
- Auto-balanced: **89-96/100** balance score ✅
- Adaptive: 65-82/100 balance score ⚖️

**Quick Mode Selection:**
```yaml
# Production (Recommended)
threshold_mode: "auto_balanced"  # 92-96/100 balance score

# Advanced Dynamic  
threshold_mode: "adaptive"       # 65-82/100 balance score

# Development Only
threshold_mode: "manual"         # 37-38/100 balance score
```

## Koleksi data (SQLite per timeframe)

```bash
# Koleksi sesuai config (separate DB per timeframe di data/db)
python3 pipeline/build_dataset.py --config config/config.yaml

# MULTI-ASSET: Koleksi multiple pairs sekaligus
python3 pipeline/build_dataset.py --pairs btcusdt ethusdt dogeusdt --timeframe 1h --months 3

# Single pair, single timeframe
python3 pipeline/build_dataset.py --pairs btcusdt --timeframe 15m --months 1

# Opsi jendela historis
python3 pipeline/build_dataset.py --config config/config.yaml --years 1 --no-force-full-rebuild
python3 pipeline/build_dataset.py --config config/config_test_separate.yaml  # mode cepat
```

**Output per asset (contoh):**
- data/db/btc_5m.sqlite, data/db/btc_15m.sqlite, data/db/btc_1h.sqlite
- data/db/eth_5m.sqlite, data/db/eth_15m.sqlite, data/db/eth_1h.sqlite  
- data/db/doge_5m.sqlite, data/db/doge_15m.sqlite, data/db/doge_1h.sqlite

## Training model (direction + extremes)

```bash
# Satu target
python3 scripts/train_tree_model.py --timeframe 15m --model LightGBM --target direction_h5 --overwrite

# MULTI-ASSET: Training multiple pairs
python3 scripts/train_tree_model.py --pairs btcusdt ethusdt dogeusdt --timeframe 15m --model LightGBM --multi-horizon-all --overwrite

# Single pair, all horizons
python3 scripts/train_tree_model.py --pairs btcusdt --timeframe 15m --model LightGBM --multi-horizon-all --overwrite

# Semua horizons + extremes di semua timeframe pada config
python3 scripts/train_tree_model.py \
  --timeframe config --model LightGBM \
  --multi-horizon-all --include-extremes --overwrite \
  --config config/config.yaml

# MULTI-ASSET: All assets, all timeframes, all targets  
python3 scripts/train_tree_model.py \
  --pairs btcusdt ethusdt dogeusdt \
  --timeframe config --model LightGBM \
  --multi-horizon-all --include-extremes --overwrite
```

Catatan:
- Artefak stabil saat pakai --overwrite. Registry tersimpan di `models/metadata/registry_<model>_<tf>.json`.
- Task otomatis: direction_* = classification; future_max/min_* = regression.

## Prediksi (live + gating R:R + paper-trade)

```bash
# Sekali (single-shot)
python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --single

# Kontinu (disarankan 5m interval 60 detik), pakai futures + prefer RR + logging CSV
python3 scripts/predict_enhanced.py --timeframe 5m --model LightGBM --continuous \
  --interval 60 --market future --prefer-rr-mode --rr-min 1.2 \
  --paper-trade-log data/logs/paper_trades_5m.csv --paper-timeout-mins 150

# Output JSON (sekali)
python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --single --output json
```

Opsi penting:
- --market spot|future (usdm)
- --rr-min angka minimum reward:risk (default ambil dari config)
- --prefer-rr-mode aktifkan preferensi arah dengan RR >= rr_min saat ML belum yakin
- --paper-trade-log path CSV append-only; --paper-timeout-mins auto close jika tidak kena TP/SL

## Cek cepat isi DB (opsional)

```bash
python3 - <<'PY'
import sqlite3, pandas as pd

for tf in ['5m','15m','1h']:
    path = f'data/db/btc_{tf}.sqlite'
    try:
        conn = sqlite3.connect(path)
        rows = pd.read_sql('select count(*) as n from features', conn)['n'].iloc[0]
        meta = pd.read_sql('select * from metadata order by created_at desc limit 1', conn)
        print(f'[{tf}] {rows} rows | latest meta:\n{meta}\n')
        conn.close()
    except Exception as e:
        pass
PY
```

## Analisis paper-trade CSV

```bash
python3 scripts/analyze_paper_trades.py --log data/logs/paper_trades_5m.csv \
  --from "2025-09-01" --to "2025-09-27"
```

Menampilkan total closed trades, win rate, avg pnl, avg hold, outcome breakdown, dan by-direction stats.

# Cara Menggunakan Pipeline Dataset

Dokumen ini menjelaskan langkah demi langkah untuk menyiapkan lingkungan, menjalankan pipeline pengumpulan & preprocessing data, serta memeriksa output.

## 1. Struktur Direktori (Setelah Flatten)
```
(project root)/
  data/
    collect_preprocess.py  (entrypoint utama baru untuk koleksi + preprocess)
    external/
 pandas, numpy, ccxt, ta, pyyaml, requests, scikit-learn
      coinmetrics.py
      dune.py (placeholder)
  features/
    indicators.py
    config.py
    logging.py
  config/
    config.yaml            (konfigurasi utama)
    config_quick.yaml      (konfigurasi cepat 1 bulan)
  how_to_use.md
  requirements.txt
```

## 2. Persiapan Lingkungan Python
```bash
python -m scripts.collect_preprocess --config config/config_test_separate.yaml
```

## 4. Menjalankan Pipeline (Mode Utama - IMPROVED)

**Approach Baru (Recommended): Separate Timeframe Databases**

```bash
# Collect data dengan separate databases per timeframe (lebih efisien)
python3 scripts/collect_preprocess.py --config config/config.yaml

# Untuk data historical penuh (1 tahun):
python3 scripts/collect_preprocess.py --config config/config.yaml --years 1 --no-force-full-rebuild

# Untuk test cepat (1 bulan):
python3 scripts/collect_preprocess.py --config config/config_test_separate.yaml
```

### Contoh (conda)
```bash
**Legacy Mode (Backward Compatibility):**
```bash
python3 scripts/collect_preprocess.py --config config.yaml
conda activate trade
        collect_preprocess.py  (entrypoint utama baru untuk koleksi + preprocess)
python -c "
from pipeline.build_dataset import build
build('config/config.yaml', 'data/db/preprocessed.sqlite')
"
Minimal library yang dibutuhkan pipeline saat ini:
```bash
python3 scripts/collect_preprocess.py --config config/config_test_separate.yaml
```

Jika ingin meminimalkan: hapus item non-esensial (tensorflow, cupy, dll) dari `requirements.txt` sebelum install.

## 3. Konfigurasi
- `pairs`: daftar pasangan simbol (contoh: `BTCUSDT`)
- `timeframes`: list timeframe (contoh: `5m`)
- `months`: berapa bulan historis untuk diambil
- `bb_periods`, `bb_devs`, `ma_periods`, dll: parameter indikator
- `target.horizon`: berapa candle ke depan untuk label arah
- `external.enable_coingecko`: set `true` jika ingin menambah data eksternal (butuh koneksi internet)
- `output.sqlite_db`: path database SQLite output (bisa override lewat argumen CLI)

Untuk test cepat gunakan `config/config_quick.yaml` (1 bulan, eksternal dimatikan):
```bash
python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --single
     - pandas, numpy, ccxt, ta, pyyaml, requests, scikit-learn

python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --interval 300 --continuous

**Approach Baru (Recommended): Separate Timeframe Databases**
python3 scripts/predict_enhanced.py --model XGBoost --timeframe 5m
python -m scripts.collect_preprocess --config config/config_test_separate.yaml
```
python3 scripts/predict_enhanced.py --timeframe 5m --model LightGBM --continuous \
  --interval 60 --market future --prefer-rr-mode --rr-min 1.2 \
  --paper-trade-log data/logs/paper_trades_5m.csv --paper-timeout-mins 150
- `data/db/btc_15m.sqlite` - Database khusus untuk 15min timeframe
- Masing-masing berisi tabel `features` dan `metadata`

```
 - `--market`: Pilih market (spot/future). Default mengikuti config (`data.market_type`)
 - `--prefer-rr-mode`: Prioritaskan arah dengan R:R >= rr_min saat ML belum yakin
 - `--rr-min`: Ambang minimum R:R untuk gating (default ambil dari config.multi_horizon.rr_min)
 - `--paper-trade-log`: Path CSV untuk simulasi paper-trade (append-only)
 - `--paper-timeout-mins`: Timeout trade paper (menit) untuk auto-close jika tidak kena TP/SL
    python3 scripts/collect_preprocess.py --config config/config.yaml --years 1 --no-force-full-rebuild
"
```
    python3 scripts/collect_preprocess.py --config config/config_test_separate.yaml
## 5. Output yang Dihasilkan

**New Improved Structure (Separate Timeframes):**
- File SQLite per timeframe: `data/db/btc_5m.sqlite`, `data/db/btc_15m.sqlite`
- Setiap database berisi:
  - Tabel `features`: berisi semua baris fitur + label untuk timeframe spesifik
  - Tabel `metadata`: catatan pembuatan (pair, timeframe, timestamp, dll)

**Legacy Structure (Mixed Timeframes):**
- File SQLite: `data/db/preprocessed.sqlite`
  - Tabel `features`: berisi semua baris fitur + label (mixed timeframes)
  - Tabel `metadata`: catatan pembuatan (pairs, timeframes, timestamp)

Cek isi cepat (new structure):
```bash
python - <<'PY'
import sqlite3, pandas as pd

# Check 15min database
conn = sqlite3.connect('data/db/btc_15m.sqlite')
    python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --single
print(pd.read_sql('select count(*) as rows from features', conn))
print(pd.read_sql('select * from metadata order by created_at desc limit 1', conn))
    python3 scripts/predict_enhanced.py --model LightGBM --timeframe 15m --interval 300 --continuous

# Check 5min database  
    python3 scripts/predict_enhanced.py --model XGBoost --timeframe 5m
print("\n=== 5min Database ===")
print(pd.read_sql('select count(*) as rows from features', conn))
    python3 scripts/predict_enhanced.py --timeframe 5m --model LightGBM --continuous \
      --interval 60 --market future --prefer-rr-mode --rr-min 1.2 \
      --paper-trade-log data/logs/paper_trades_5m.csv --paper-timeout-mins 150
print(pd.read_sql('select * from metadata order by created_at desc limit 1', conn))
conn.close()
PY
     - `--market`: Pilih market (spot/future). Default mengikuti config (`data.market_type`)
     - `--prefer-rr-mode`: Prioritaskan arah dengan R:R >= rr_min saat ML belum yakin
     - `--rr-min`: Ambang minimum R:R untuk gating (default ambil dari config.multi_horizon.rr_min)
     - `--paper-trade-log`: Path CSV untuk simulasi paper-trade (append-only)
     - `--paper-timeout-mins`: Timeout trade paper (menit) untuk auto-close jika tidak kena TP/SL
- Turunan: `ret_1`, `ret_2`, rasio harga terhadap MA / Bollinger
- Label: `direction_label` (UP / SIDEWAYS / DOWN), `vol_regime` (LOW / MID / HIGH)

## 7. Penanganan Data Eksternal
Jika `external.enable_coingecko = true`, pipeline akan mencoba mengambil snapshot market (butuh internet). Coinmetrics & Dune hanya aktif jika API key tersedia di environment (`COINMETRICS_API_KEY`, `DUNE_API_KEY`). Jika key tidak ada, modul akan kembali DataFrame kosong tanpa error fatal.

## 8. Logging
Logger default ke stdout. Jika ingin menulis ke file, bisa modifikasi `utils/logging.py` untuk menambah FileHandler.

## 9. Masalah Umum
- ImportError `No module named scripts`: pastikan jalankan dari root project (folder yang berisi direktori `scripts/`).
- Data kosong: Binance kadang rate limit; coba ulang atau kurangi `months`.
- Kolom label NaN: Pastikan cukup panjang data relatif terhadap `target.horizon` dan rolling window indikator.

## 10. Langkah Lanjut
- Tambah trainer model (CatBoost) yang membaca tabel `features`.
- Menyederhanakan `requirements.txt` sesuai kebutuhan nyata.
- Menambah tests (pytest) untuk fungsi indikator & label.
- Menambah caching lokal untuk OHLCV.

---
Dokumen ini akan diperbarui setelah modul training ditambahkan.

## 11. Enhanced Prediction System

### Live Prediction dengan Bollinger Bands dan Advanced Analytics

Pipeline ini sekarang mendukung sistem prediksi real-time yang ditingkatkan yang menggabungkan sinyal dari model ML dan analisis teknikal.

### Fitur Enhanced Prediction:

1. **Integrasi Live Price**: Mengambil harga terbaru dari Binance dan membandingkannya dengan data historis
2. **Analisis Bollinger Bands**: Analisis posisi harga terhadap BB di berbagai periode (BB48, BB96)
3. **Slope Analysis**: Analisis trend Bollinger Bands (naik/turun/sideways)
4. **Signal Integration**: Integrasi sinyal dari multiple Bollinger Bands dengan weighting yang tepat
5. **Conflict Resolution**: Penanganan konflik antara prediksi ML dan sinyal Bollinger Bands
6. **TP/SL Optimization**: Perhitungan TP/SL yang optimal berdasarkan level Bollinger Bands
7. **Advanced Trading Analytics**: Analisis tambahan seperti risk-reward ratio, trend strength, volume confirmation, dll.

### Cara Menggunakan:

```bash
# Menjalankan prediksi tunggal dengan model LightGBM pada timeframe 15m
python -m scripts.predict_enhanced --model LightGBM --timeframe 15m --single

# Menjalankan prediksi kontinu dengan update setiap 5 menit
python -m scripts.predict_enhanced --model LightGBM --timeframe 15m --interval 300

# Menggunakan model dan timeframe lainnya
python -m scripts.predict_enhanced --model XGBoost --timeframe 5m
```

### Opsi Lain:
- `--model`: Pilih model ML (LightGBM, XGBoost, CatBoost)
- `--timeframe`: Pilih timeframe (5m, 15m, 1h)
- `--interval`: Interval update dalam detik (default: 300)
- `--single`: Jalankan hanya satu prediksi dan keluar
- `--continuous`: Jalankan prediksi kontinu dengan interval
- `--config`: Path ke file konfigurasi alternatif

### Output yang Ditampilkan:

1. **Live Prediction Analysis**: Harga terbaru, timestamp, prediksi model, dan confidence
2. **Bollinger Bands Analysis**: Posisi harga terhadap berbagai level Bollinger Bands
3. **Bollinger Band Levels**: Detail level BB48 dan jarak harga dari setiap level
4. **Advanced Trading Analytics**: Analisis mendalam yang ditambahkan untuk membantu keputusan trading:
   - Risk-Reward Ratio: Potensi reward vs risk pada trade
   - Expected Return: Return statistik berdasarkan confidence model
   - Next Candle Prediction: Prediksi pergerakan candle selanjutnya
   - Trend Strength: Kekuatan trend berdasarkan ADX
   - Volume Confirmation: Konfirmasi volume relatif terhadap rata-rata
   - Historical Performance: Win rate dan profit rata-rata dari pola serupa
   - Alerts: Peringatan penting terkait kondisi pasar

5. **Trading Recommendations**: Rekomendasi entry, stop loss, take profit, position sizing
6. **Trading Summary**: Ringkasan keseluruhan sinyal dengan reasoning yang jelas

### Catatan Penting:
- Untuk memanfaatkan semua fitur Advanced Analytics, pastikan TA-Lib terinstal di sistem.
- Jika TA-Lib tidak tersedia, sistem akan fallback ke analisis yang lebih sederhana.
- Pada konflik sinyal antara ML dan Bollinger Bands, sistem akan menyelesaikan konflik berdasarkan confidence.

### Instalasi TA-Lib (opsional):
```bash
# Pada Ubuntu/Debian
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# Pada Conda
conda install -c conda-forge ta-lib
```

## 12. Multi-Horizon Direction & Extremes Labeling

Sistem sekarang mendukung beberapa horizon arah sekaligus dan label "extremes" untuk estimasi reward:risk.

### 12.1 Direction Multi-Horizon
Jika `multi_horizon.horizons: [1,5,20]` di config, maka akan dibuat kolom:
- `direction_h1`
- `direction_h5`
- `direction_h20` (sering juga disingkat `direction` lama / base horizon)

Format kelas (multiclass / classification):
0 = DOWN, 1 = SIDEWAYS, 2 = UP  (threshold sideways menggunakan `multi_horizon.thresholds.sideways_pct` atau `target.sideways_threshold_pct`).

### 12.2 Extremes (Regression Targets)
Jika `multi_horizon.enable_extremes = true`, untuk horizon basis (misal 20) akan dibuat:
- `future_max_high_h20`  → Perkiraan maksimum kenaikan relatif (fractional / decimal)
- `future_min_low_h20`   → Perkiraan maksimum penurunan relatif (fractional / decimal, biasanya bernilai negatif)

Definisi (contoh horizon H=20):
```
future_max_high_h20 = (max(high[1..H]) - close[0]) / close[0]
future_min_low_h20  = (min(low[1..H])  - close[0]) / close[0]
```
Catatan: `future_min_low_h20` biasanya negatif (karena low masa depan <= close saat ini). Predictor mengambil nilai absolut untuk menghitung risiko (downside) dalam persen.

### 12.3 Reward:Risk Ratio (R:R)
Ketika kedua extremes tersedia dan diprediksi:
```
upside_pct   = future_max_high_h20_pred * 100
downside_pct = abs(future_min_low_h20_pred) * 100
R:R          = upside_pct / downside_pct
```
Threshold gating ditentukan oleh `multi_horizon.rr_min` (default 1.2). Jika R:R < rr_min DAN sinyal ML utama adalah BUY atau SELL, sinyal dipaksa menjadi HOLD (ditandai `rr_gate_triggered=true` di output JSON).

## 13. Training Multi-Target & Extremes

Gunakan `scripts/train_tree_model.py` (universal). Anda bisa melatih satu target atau banyak sekaligus.

### 13.1 Contoh Melatih Satu Target
```bash
python -m scripts.train_tree_model --timeframe 15m --model LightGBM --target direction_h5 --overwrite
```
`--overwrite` akan menghasilkan nama artefak stabil tanpa timestamp untuk target tersebut (mudah dipakai deployment). Tanpa `--overwrite`, timestamp akan ditambahkan.

### 13.2 Melatih Semua Direction Horizons Sekaligus
```bash
python -m scripts.train_tree_model --timeframe 15m --model LightGBM --multi-horizon-all --overwrite
```
Script otomatis mendeteksi kolom: `direction_h1`, `direction_h5`, `direction_h10`, `direction_h20`, `direction` (horizon base jika ada).

### 13.3 Tambah Extremes Sekaligus
```bash
python -m scripts.train_tree_model \
  --timeframe 15m \
  --model LightGBM \
  --multi-horizon-all \
  --include-extremes \
  --overwrite
```

### 13.4 Melatih Banyak Timeframe Berdasarkan Config
```bash
python -m scripts.train_tree_model --timeframe config --model LightGBM --multi-horizon-all --include-extremes --overwrite
```
`--timeframe config` berarti loop ke setiap timeframe di `config.yaml: timeframes:`.

### 13.5 Deteksi Otomatis Jenis Tugas
- Kolom yang diawali `future_max_high_` atau `future_min_low_` → regression (RMSE dicatat)
- Selain itu (direction_h*, direction) → classification (accuracy dicatat)

### 13.6 Pencegahan Feature Leakage
Trainer otomatis mengecualikan SEMUA label/horizon lain saat melatih satu target:
- Menghapus kolom: `direction`, `direction_h*`, `future_return_pct_h*`, `future_max_high_h*`, `future_min_low_h*`, dsb kecuali target aktif.
- Menghapus kolom meta: timestamp, pair, symbol, timeframe, dsb.
Pastikan tidak menambahkan manual kolom label kembali ke daftar fitur.

## 14. Artefak Model & Registry

Setiap target menghasilkan 4 artefak utama (per model_type & timeframe):
- Model file
- Scaler file
- Features list
- Metadata JSON

Skema nama (contoh LightGBM, timeframe 15m, target `future_max_high_h20`):
- Model (overwrite mode): `models/tree_models/btc_15m_lightgbm_future_max_high_h20.txt`
- Scaler: `models/tree_models/btc_15m_scaler_future_max_high_h20.joblib`
- Features list: `models/tree_models/btc_15m_lightgbm_features_future_max_high_h20.txt`
- Metadata: `models/metadata/btc_15m_lightgbm_future_max_high_h20_metadata.json`

Jika tanpa `--overwrite`, timestamp disisipkan sebelum ekstensi model & scaler.

### 14.1 Registry
Setiap kali model tersimpan, file registry diperbarui:
```
models/metadata/registry_<model_type_lower>_<timeframe>.json
```
Contoh: `registry_lightgbm_15m.json`

Struktur ringkas:
```json
{
  "timeframe": "15m",
  "model_type": "LightGBM",
  "targets": [
    {
      "target_name": "direction_h5",
      "model_file": "btc_15m_lightgbm_direction_h5.txt",
      "scaler_file": "btc_15m_scaler_direction_h5.joblib",
      "features_file": "btc_15m_lightgbm_features_direction_h5.txt",
      "metadata_file": "btc_15m_lightgbm_direction_h5_metadata.json",
      "metric_name": "accuracy",
      "metric_value": 0.6123,
      "task_type": "classification",
      "timestamp": "20250923_151017"
    },
    {
      "target_name": "future_max_high_h20",
      "model_file": "btc_15m_lightgbm_future_max_high_h20.txt",
      "scaler_file": "btc_15m_scaler_future_max_high_h20.joblib",
      "features_file": "btc_15m_lightgbm_features_future_max_high_h20.txt",
      "metadata_file": "btc_15m_lightgbm_future_max_high_h20_metadata.json",
      "metric_name": "rmse",
      "metric_value": 0.00180,
      "task_type": "regression",
      "timestamp": "20250923_151017"
    }
  ]
}
```

Predictor multi-model membaca registry ini untuk memuat semua target relevan.

## 15. Multi-Model Prediction & R:R Gating

`scripts/predict_enhanced.py` kini:
1. Memuat model utama (auto prefer LightGBM jika `--model auto`).
2. Mencoba memuat registry multi-model (`registry_lightgbm_<tf>.json`).
3. Menjalankan inferensi per target (classification & regression).
4. Menghitung `upside_pct`, `downside_pct`, dan `rr` jika kedua extremes tersedia.
5. Menerapkan gating: jika `rr < rr_min` dan sinyal arah (BUY / SELL) → dipaksa HOLD.
6. Menyediakan panel tambahan (rich) atau field JSON `multi_model`.

### 15.1 Prioritas Direction Primary
Urutan preferensi untuk sinyal arah utama: `direction_h5` → `direction` → `direction_h1` (jika tersedia masing-masing). Confidence = maksimum probabilitas kelas.

### 15.2 JSON Output (Mode `--output json --single`)
Contoh (disingkat):
```json
{
  "timestamp": "2025-09-23T15:19:40.123456Z",
  "timeframe": "15m",
  "model_type": "LightGBM",
  "prediction": 2,
  "prediction_original": 2,
  "confidence": 0.8234,
  "overridden": false,
  "prediction_proba": [0.05,0.12,0.83],
  "bb_signals": {"bb_48": "BUY", "bb_96": "NEUTRAL"},
  "price": 61234.5,
  "price_db": 61220.1,
  "price_diff_pct": 0.02,
  "multi_model": {
    "direction_primary": "UP",
    "direction_confidence": 0.8312,
    "upside_pct": 1.2446,
    "downside_pct": 0.9527,
    "rr": 1.3064,
    "targets": {
      "direction_h5": {"prediction": 2, "probs": [0.04,0.13,0.83], "task": "classification"},
      "future_max_high_h20": {"prediction": 0.012446, "task": "regression"},
      "future_min_low_h20": {"prediction": -0.009527, "task": "regression"}
    }
  },
  "rr_gate_triggered": false
}
```

Field penting baru:
- `multi_model.targets` berisi setiap target.
- `upside_pct` / `downside_pct` dalam persen (bukan decimal).
- `rr` = reward:risk ratio (angka > 1 lebih baik).
- `rr_gate_triggered` = true jika sinyal digate jadi HOLD.

### 15.3 Menjalankan Prediction JSON Sekali
```bash
python -m scripts.predict_enhanced --model LightGBM --timeframe 15m --single --output json
```

### 15.4 Continuous Mode Dengan Multi-Model
```bash
python -m scripts.predict_enhanced --model LightGBM --timeframe 15m --continuous --interval 300
```

## 16. Cheat Sheet Konfigurasi Multi-Horizon
Di `config/config.yaml`:
```yaml
multi_horizon:
  horizons: [1,5,20]
  enable_extremes: true
  rr_min: 1.2
  thresholds:
    sideways_pct: 1.0
```
Ubah `rr_min` untuk lebih ketat (misal 1.5) agar hanya sinyal dengan reward lebih tinggi dipertahankan.

## 17. Troubleshooting Multi-Model
| Masalah | Penyebab Umum | Solusi Singkat |
|---------|---------------|----------------|
| Extremes tidak muncul | Belum dilatih / registry tidak memuat extremes | Jalankan training dengan `--include-extremes` |
| `rr` null | Salah satu extremes gagal loading / fitur hilang | Cek log warning "Target ... fitur hilang" |
| Gate terlalu sering aktif | `rr_min` terlalu tinggi | Turunkan `multi_horizon.rr_min` |
| Model lama dipakai | Tidak pakai `--overwrite` (timestamp banyak) | Gunakan `--overwrite` untuk artefak stabil |
| Feature mismatch | Perubahan pipeline tanpa retrain | Retrain target terkait |

## 18. Next Steps (Optional Enhancements)
- Tambah evaluasi out-of-sample per target.
- Logging terstruktur untuk multi-model JSON ke file.
- Integrasi strategi position sizing adaptif berbasis R:R & confidence.
- Menambah unit test untuk verifikasi tidak ada label leakage.

---
Dokumentasi telah diperbarui untuk mendukung multi-horizon + extremes + reward:risk gating.
