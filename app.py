"""HimaRoad + flood-copilot: one app. Run with: streamlit run app.py"""

import streamlit as st

st.set_page_config(page_title="HimaRoad + flood-copilot", layout="wide")

NAVY, TEAL, SAFFRON, ICE, INK, LINE = "#0E2A3B", "#1F7A8C", "#E3872D", "#EAF3F6", "#1B2329", "#D5DEE2"
st.markdown(
    f"""<style>
    h1, h2, h3, [data-testid="stMetricValue"] {{ font-family: Cambria, Georgia, "Times New Roman", serif !important; color: {NAVY}; }}
    h1 {{ font-size: 2.0rem !important; padding-top: 0.2rem !important; }}
    .kicker {{ color: {TEAL}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: -0.6rem; }}
    .band {{ background: {NAVY}; color: #fff; border-radius: 10px; padding: 14px 22px; margin: 0 0 14px 0;
             display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }}
    .band b {{ font-family: Cambria, Georgia, serif; font-size: 1.35rem; letter-spacing: 0.01em; }}
    .band span {{ color: #BFDCE5; font-size: 0.86rem; }}
    .band .pill {{ background: {SAFFRON}; color: #fff; border-radius: 999px; padding: 3px 12px; font-size: 0.75rem; font-weight: 700; }}
    [data-testid="stMetric"] {{ background: {ICE}; border-left: 4px solid {TEAL}; border-radius: 8px; padding: 10px 14px; }}
    [data-testid="stMetricLabel"] p {{ color: #5B6B73; font-size: 0.82rem; }}
    [data-testid="stMetricValue"] {{ font-size: 2.0rem; }}
    [data-testid="stExpander"] details {{ border: 1px solid {LINE}; border-radius: 8px; }}
    [data-testid="stSidebar"] {{ background: {ICE}; }}
    .stButton button, .stFormSubmitButton button, .stDownloadButton button {{ border-radius: 8px; border: 1px solid {TEAL}; color: {TEAL}; }}
    .stButton button:hover, .stFormSubmitButton button:hover {{ background: {TEAL}; color: #fff; }}
    .card {{ background: {ICE}; border-left: 4px solid {TEAL}; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }}
    .card.warn {{ border-left-color: {SAFFRON}; background: #FDF3E8; }}
    .card b {{ color: {NAVY}; }}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="band"><div><b>HimaRoad + flood-copilot</b><br>'
    '<span>Mapping flood damage and cut-off settlements in Himalayan valleys from space</span></div>'
    '<div class="pill">TRACK B · SITUATION-REPORT COPILOT</div></div>',
    unsafe_allow_html=True,
)

page = st.navigation(
    [
        st.Page("views/dashboard.py", title="Evidence map", default=True),
        st.Page("ui/app.py", title="Ask the copilot", url_path="copilot"),
        st.Page("views/about.py", title="About and limits", url_path="about"),
    ],
    position="top",
)
page.run()
