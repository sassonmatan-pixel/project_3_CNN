"""Fruit Vision - Streamlit interface.

Run from the project root:
    streamlit run streamlit_app.py

This file only handles the *look and interaction*. The model, the preprocessing and the
60% "unknown" rule all live in app/predictor.py (shared with the FastAPI server).
"""
from __future__ import annotations

import hashlib
import html

import streamlit as st

from app.predictor import FruitPredictor, InvalidImageError

st.set_page_config(page_title="Fruit Vision – זיהוי פירות", page_icon="🍎", layout="centered")

HEBREW = {"Apple": "תפוח", "Avocado": "אבוקדו", "Banana": "בננה", "Eggplant": "חציל", "Tomato": "עגבנייה"}
EMOJI = {"Apple": "🍎", "Avocado": "🥑", "Banana": "🍌", "Eggplant": "🍆", "Tomato": "🍅", "unknown": "❓"}

CSS = """
<style>
:root { --green:#16a34a; --green-dark:#15803d; --purple:#7c3aed; --purple-soft:#ede9fe;
        --green-soft:#dcfce7; --muted:#6b7280; }

/* Page background + RTL for the whole app */
.stApp {
  direction: rtl;
  background: radial-gradient(circle at 10% 0%, var(--green-soft) 0, transparent 45%),
              radial-gradient(circle at 100% 20%, var(--purple-soft) 0, transparent 40%), #f4faf6;
}
.block-container { max-width: 640px; padding-top: 2rem; padding-bottom: 3rem; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp div { font-family: "Segoe UI", system-ui, "Arial Hebrew", Arial, sans-serif; }

/* Hero */
.fv-hero { text-align: center; padding: 6px 0 18px; }
.fv-logo { width: 74px; height: 74px; margin: 0 auto 10px; display: grid; place-items: center; font-size: 40px;
  background: linear-gradient(135deg, var(--green), var(--purple)); border-radius: 22px;
  box-shadow: 0 10px 25px rgba(124,58,237,.25); }
.fv-hero h1 { margin: 0; padding: 0; font-size: 2.3rem; font-weight: 800;
  background: linear-gradient(90deg, var(--green-dark), var(--purple));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.fv-hero p { margin: 6px 0 0; color: var(--muted); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; justify-content: center; }
.stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 8px 18px; background: #fff; font-weight: 600; }
.stTabs [aria-selected="true"] { background: var(--purple-soft); color: var(--purple); }

/* Uploader + camera look like cards */
[data-testid="stFileUploader"] section, [data-testid="stCameraInput"] {
  border-radius: 16px; border: 2px dashed #c4b5fd; background: #fff; }
[data-testid="stImage"] img { border-radius: 16px; box-shadow: 0 6px 24px rgba(17,24,39,.10); }

/* Primary button = the big "identify" button */
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  width: 100%; padding: 14px 18px; font-size: 1.15rem; font-weight: 700; color: #fff; border: 0; border-radius: 14px;
  background: linear-gradient(90deg, var(--green), var(--purple)); box-shadow: 0 8px 20px rgba(22,163,74,.30);
  transition: transform .08s ease; }
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); color: #fff; }
.stDownloadButton > button, button[data-testid="stBaseButton-secondary"] { border-radius: 14px; font-weight: 600; width: 100%; }

/* Result card */
.fv-card { background: #fff; border-radius: 18px; padding: 20px; margin-top: 14px; box-shadow: 0 6px 24px rgba(17,24,39,.07);
  animation: fv-pop .35s ease; }
@keyframes fv-pop { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.fv-top { display: flex; align-items: center; gap: 16px; }
.fv-emoji { font-size: 56px; line-height: 1; }
.fv-label { font-size: 1.8rem; font-weight: 800; color: var(--green-dark); }
.fv-label.unknown { color: var(--muted); }
.fv-conf { color: var(--muted); font-size: 1.05rem; }
.fv-conf b { color: var(--purple); font-size: 1.3rem; }
.fv-note { margin-top: 12px; padding: 10px 12px; border-radius: 12px; background: var(--purple-soft); color: #5b21b6; font-size: .95rem; }
.fv-sub { margin: 18px 0 8px; color: var(--muted); font-weight: 600; font-size: 1rem; }
.fv-row { display: grid; grid-template-columns: 78px 1fr 56px; align-items: center; gap: 10px; margin: 9px 0; }
.fv-track { height: 12px; border-radius: 99px; background: #eef2f0; overflow: hidden; direction: ltr; }
.fv-fill { height: 100%; border-radius: 99px; background: #a7d7b8; animation: fv-grow .7s ease; }
.fv-row.top .fv-fill { background: linear-gradient(90deg, var(--green), var(--purple)); }
.fv-row.top { font-weight: 700; }
.fv-pct { direction: ltr; text-align: left; color: var(--muted); font-variant-numeric: tabular-nums; }
@keyframes fv-grow { from { width: 0 !important; } }

/* Loading animation */
.fv-loading { display: flex; align-items: center; justify-content: center; gap: 12px; padding: 18px;
  color: var(--purple); font-weight: 700; }
.fv-spin { width: 24px; height: 24px; border-radius: 50%; border: 3px solid var(--purple-soft);
  border-top-color: var(--purple); animation: fv-rot .8s linear infinite; }
@keyframes fv-rot { to { transform: rotate(360deg); } }

/* Phones */
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }
  .fv-hero h1 { font-size: 1.9rem; }
  .fv-label { font-size: 1.5rem; }
  .fv-emoji { font-size: 46px; }
  .fv-row { grid-template-columns: 66px 1fr 50px; }
}
</style>
"""


@st.cache_resource(show_spinner="טוען את המודל…")
def get_predictor() -> FruitPredictor:
    """Load the CNN once per server process (not on every click)."""
    return FruitPredictor()


def result_card(r: dict) -> str:
    key = "unknown" if r["is_unknown"] else r["prediction"]
    label = "לא ידוע" if r["is_unknown"] else HEBREW.get(r["prediction"], r["prediction"])
    note = ""
    if r["is_unknown"]:
        note = '<div class="fv-note">רמת הביטחון נמוכה מ-60%, לכן התוצאה מוצגת כ"לא ידוע". נסו תמונה ברורה יותר של פרי אחד.</div>'

    rows = []
    for i, (name, p) in enumerate(sorted(r["probabilities"].items(), key=lambda kv: kv[1], reverse=True)):
        pct = p * 100
        rows.append(
            f'<div class="fv-row{" top" if i == 0 else ""}"><span>{html.escape(HEBREW.get(name, name))}</span>'
            f'<div class="fv-track"><div class="fv-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="fv-pct">{pct:.1f}%</span></div>'
        )
    return (
        '<div class="fv-card">'
        f'<div class="fv-top"><div class="fv-emoji">{EMOJI.get(key, "🍽️")}</div><div>'
        f'<div class="fv-label{" unknown" if r["is_unknown"] else ""}">{html.escape(label)}</div>'
        f'<div class="fv-conf">רמת ביטחון: <b>{r["confidence"] * 100:.1f}%</b></div></div></div>'
        f"{note}"
        '<div class="fv-sub">הסתברות לכל פרי</div>' + "".join(rows) + "</div>"
    )


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="fv-hero"><div class="fv-logo">🍎</div><h1>Fruit Vision</h1>'
    "<p>צלמו תמונה והמערכת שלנו תזהה את הפרי</p></div>",
    unsafe_allow_html=True,
)

predictor = get_predictor()

# --- 1. Get an image: camera or gallery/computer -------------------------------------
tab_cam, tab_up = st.tabs(["📷 צילום", "🖼️ גלריה / מחשב"])
with tab_cam:
    cam = st.camera_input("צלמו פרי", label_visibility="collapsed")
with tab_up:
    up = st.file_uploader("בחרו תמונה", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

source = cam or up  # a fresh photo wins; otherwise use the uploaded file
if source is None:
    st.info("העלו תמונה או צלמו כדי להתחיל 👆")
    st.stop()

image_bytes = source.getvalue()
image_id = hashlib.sha1(image_bytes).hexdigest()

# --- 2. Preview + actions -------------------------------------------------------------
st.image(image_bytes, width="stretch")

if st.button("🔍 זהה את הפרי", type="primary", width="stretch"):
    slot = st.empty()
    slot.markdown('<div class="fv-loading"><span class="fv-spin"></span>מנתח את התמונה…</div>', unsafe_allow_html=True)
    try:
        st.session_state["result"] = (image_id, predictor.predict(image_bytes))
        st.session_state.pop("error", None)
    except InvalidImageError:
        st.session_state["error"] = "הקובץ אינו תמונה תקינה. נסו קובץ JPG או PNG אחר."
        st.session_state.pop("result", None)
    slot.empty()

st.download_button(
    "💾 שמור תמונה",
    data=image_bytes,
    file_name="fruit-photo." + ("png" if image_bytes[:4] == b"\x89PNG" else "jpg"),
    mime="image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg",
    width="stretch",
)

# --- 3. Result (only if it belongs to the image currently shown) ------------------------
if "error" in st.session_state:
    st.error(st.session_state["error"])
saved = st.session_state.get("result")
if saved and saved[0] == image_id:
    st.markdown(result_card(saved[1]), unsafe_allow_html=True)
