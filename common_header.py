# common_header.py
import streamlit as st

PAGES = [
    ("🔎 Finder",    "pages/01_Finder.py",   "Finder"),
    ("📈 Trends",    "pages/03_Trends.py",   "Trends"),
    ("🧭 Explore",   "pages/06_Explore.py",  "Explore"),
    ("🌡️ Heatmap",  "pages/04_Heatmap.py",  "Heatmap"),
    ("📉 Slopegraph","pages/05_Slopegraph.py","Slopegraph"),
]

def top_nav(active: str, hide_sidebar=True, title_size="1.6rem"):
    st.set_page_config(page_title="Admissions Dashboard", layout="wide",
                       initial_sidebar_state="collapsed")
    if hide_sidebar:
        st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

    st.markdown(f"""
    <style>
      .block-container {{ max-width:1200px; padding: 1.9rem .8rem 1rem; }}
      .stApp h1, h1 {{ font-size:{title_size} !important; font-weight:800 !important; }}
      .toplinks a {{ 
        display:inline-flex; align-items:center; gap:.4rem;
        padding:.45rem .75rem; border:1px solid rgba(0,0,0,.1); border-radius:10px;
        text-decoration:none;
      }}
      .toplinks a[aria-disabled="true"] {{ opacity:.45; pointer-events:none; }}
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns([1.1,1.1,1.2,1.2,1.6,5], gap="small")
    with cols[0]: st.page_link("pages/01_Finder.py",    label="🔎 Finder",    disabled=(active=="Finder"))
    with cols[1]: st.page_link("pages/03_Trends.py",    label="📈 Trends",    disabled=(active=="Trends"))
    with cols[2]: st.page_link("pages/06_Explore.py",   label="🧭 Explore",   disabled=(active=="Explore"))
    with cols[3]: st.page_link("pages/04_Heatmap.py",   label="🌡️ Heatmap",   disabled=(active=="Heatmap"))
    with cols[4]: st.page_link("pages/05_Slopegraph.py",label="📉 Slopegraph",disabled=(active=="Slopegraph"))
