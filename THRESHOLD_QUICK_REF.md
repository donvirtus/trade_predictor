# Threshold Modes - Quick Reference Card

## 🎯 **Mode Selection Guide**

```yaml
# PRODUCTION (Recommended) - Perfect Balance
target:
  threshold_mode: "auto_balanced"

# ADVANCED - Dynamic Adaptation  
target:
  threshold_mode: "adaptive"

# DEVELOPMENT - Simple Testing
target:
  threshold_mode: "manual"
```

## 📊 **Performance Matrix**

| Mode | Balance Score | Asset-Specific | Dynamic | Use Case |
|------|---------------|----------------|---------|----------|
| **Manual** | 37-38/100 ❌ | No | No | Development |
| **Auto-Balanced** | **89-96/100** ✅ | Yes | No | **Production** |
| **Adaptive** | 65-82/100 ⚖️ | Yes | Yes | Dynamic Markets |

## ⚙️ **Quick Setup**

### Production Setup (5 minutes)
```bash
# 1. Generate optimal thresholds
python utils/threshold_balancer.py

# 2. Set mode in config.yaml
threshold_mode: "auto_balanced"

# 3. Build datasets
python pipeline/build_dataset.py --pairs btcusdt ethusdt dogeusdt --timeframe config --months 3
```

### Adaptive Setup (10 minutes)
```yaml
target:
  threshold_mode: "adaptive"
  adaptive_thresholds:
    enabled: true
    method: "hybrid"
    auto_update: true
    update_frequency_days: 7
    manual_multipliers:
      "5m": 0.3
      "1h": 1.0  
      "4h": 2.0
```

## 🔧 **Troubleshooting**

| Issue | Solution |
|-------|----------|
| `Auto-balance config not found` | Run `python utils/threshold_balancer.py` |
| `Cache expired` | Normal - auto-refreshes weekly |
| `Poor balance (< 50/100)` | Switch to `auto_balanced` mode |
| `Adaptive not working` | Check cache: `ls data/adaptive_multipliers_cache_*.json` |

## 📈 **Validation Commands**

```bash
# Check balance quality
python -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/db/btc_1h.sqlite')  
df = pd.read_sql('SELECT direction, COUNT(*) FROM features GROUP BY direction', conn)
print(df)
"

# Validate threshold impact
python validate_threshold.py --asset btc --timeframe 1h

# Check current mode
grep "threshold_mode" config/config.yaml
```

---
**📚 Full Guide:** [THRESHOLD_MODES_GUIDE.md](THRESHOLD_MODES_GUIDE.md)  
**🎯 Winner:** Auto-Balanced Mode (92-96/100 balance score)