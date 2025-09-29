import os, requests, pandas as pd, datetime as dt

BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

def fetch_coinmetrics(asset: str = 'btc', metrics=("AdrActCnt","TxCnt"), start=None, end=None):
    api_key = os.getenv('COINMETRICS_API_KEY')
    if not api_key:
        return pd.DataFrame()
    end = end or dt.datetime.utcnow().date()
    start = start or (end - dt.timedelta(days=30))
    params = {
        'assets': asset,
        'metrics': ','.join(metrics),
        'start': start.isoformat(),
        'end': end.isoformat()
    }
    r = requests.get(BASE, params=params, headers={'Authorization': f'Bearer {api_key}'}, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json().get('data', [])
    rows = []
    for row in data:
        rec = {'date': row['time'][:10]}
        for m in metrics:
            val = row.get(m)
            rec[m] = float(val) if val not in (None,'') else None
        rows.append(rec)
    return pd.DataFrame(rows)
