import os, json, requests
from datetime import datetime, timezone
import pandas as pd

CACHE_DIR = 'data/external_cache'


def fetch_coingecko_snapshot(coin_id: str = 'bitcoin', use_cache: bool = True) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/coingecko_{coin_id}_{datetime.utcnow().date()}.json"
    if use_cache and os.path.exists(cache_file):
        with open(cache_file,'r') as f:
            js = json.load(f)
    else:
        url=("https://api.coingecko.com/api/v3/coins/"+coin_id+
             "?localization=false&tickers=false&market_data=true&community_data=false&developer_data=true&sparkline=false")
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        with open(cache_file,'w') as f:
            json.dump(js,f)
    ts = datetime.utcnow().replace(tzinfo=timezone.utc)
    md = js.get('market_data',{})
    return pd.DataFrame([{ 'date': ts.date(),
                           'cg_market_cap': md.get('market_cap',{}).get('usd'),
                           'cg_volume': md.get('total_volume',{}).get('usd'),
                           'cg_circ_supply': md.get('circulating_supply'),
                           'cg_fdv': md.get('fully_diluted_valuation',{}).get('usd')}])
