# router.py
from urllib.parse import urlencode
import streamlit as st

DEFAULTS = {
    "fy": "2015-2024",
    "level": "Code3",
    "query": "",
    "admission": "All",
    "sex": "All",
    "metric": "",     # volume|structure|timeliness
    "view": "",       # bar|treemap|compare
    "clicked": "",
    "drop": "-30",
    "rebound": "20",
    "baseline": "2019/20",
}

def _qp_get() -> dict:
    try:
        d = dict(st.query_params)
    except Exception:
        d = st.experimental_get_query_params()
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = (v[0] if v else "")
        else:
            out[k] = "" if v is None else str(v)
    return out

def _qp_set(d: dict) -> None:
    d2 = {k: ("" if v is None else str(v)) for k, v in d.items()}
    try:
        st.query_params.clear()
        st.query_params.update(d2)
    except Exception:
        st.experimental_set_query_params(**d2)

def get_state() -> dict:
    q = _qp_get()
    s = {**DEFAULTS, **{k: q.get(k, DEFAULTS.get(k, "")) for k in DEFAULTS}}
    for k in ("drop", "rebound"):
        try:
            s[k] = int(s[k])
        except Exception:
            s[k] = int(DEFAULTS[k])
    return s

def set_state(**updates) -> None:
    s = get_state()
    for k, v in updates.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    _qp_set(s)

def share_url(**overrides) -> str:
    s = {**get_state(), **overrides}
    q = urlencode({k: ("" if v is None else str(v)) for k, v in s.items()})
    return f"?{q}" if q else "."
