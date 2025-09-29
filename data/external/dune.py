import os, pandas as pd

# Placeholder Dune integration; real flow: POST execute, poll status, get results

def fetch_dune_query(query_id: int):
    api_key = os.getenv('DUNE_API_KEY')
    if not api_key:
        return pd.DataFrame()
    # TODO: implement real call sequence
    return pd.DataFrame()
