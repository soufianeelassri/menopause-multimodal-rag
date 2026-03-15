"""MenoGuide Streamlit application entry point.

Provides a 4-tab interface: RAG Chatbot, Symptom Tracker,
PDF Report, and Educational Cards — per paper Section IV-D.
"""

from __future__ import annotations

import gc

import streamlit as st

from src.app.components.sidebar import render_sidebar
from src.app.data.common_data import menopause_stages, wellness_tips
from src.app.pages.chatbot import render_chat_tab
from src.app.pages.educational_cards import render_educational_cards_tab
from src.app.pages.pdf_report import render_pdf_report_tab
from src.app.pages.symptom_tracker import render_symptom_tracker_tab
from src.app.styles.css_styles import load_css_styles
from src.generation.generator import get_llm
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vectorstore import get_vectorstore
from src.utils.helpers import load_assets


# Page configuration
st.set_page_config(
    page_title="MenoGuide",
    page_icon="\U0001f338",
    layout="wide",
)

# Load CSS styles
st.markdown(load_css_styles(), unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Chat"


# Load models and assets
@st.cache_resource(show_spinner=False)
def _initialize_resources() -> dict:
    """Load and cache all required resources."""
    llm = get_llm()
    vectorstore = get_vectorstore()
    bm25 = BM25Index()
    bm25.load()
    assets = load_assets()

    return {
        "llm": llm,
        "vectorstore": vectorstore,
        "bm25_index": bm25,
        "assets": assets,
    }


resources = _initialize_resources()


# Reset chat function
def _reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.chat_history = []
    gc.collect()


# Render sidebar
render_sidebar(wellness_tips, menopause_stages, _reset_chat)

# Create tabs — 4-tab layout per paper Section IV-D
tabs = st.tabs([
    "\U0001f4ac Chat",
    "\U0001f4ca Symptom Tracker",
    "\U0001f4c4 PDF Report",
    "\U0001f4da Information Cards",
])

# Render each tab
with tabs[0]:
    render_chat_tab(
        llm=resources["llm"],
        vectorstore=resources["vectorstore"],
        bm25_index=resources["bm25_index"],
    )

with tabs[1]:
    render_symptom_tracker_tab(llm=resources["llm"])

with tabs[2]:
    render_pdf_report_tab()

with tabs[3]:
    render_educational_cards_tab(
        resource_images=resources["assets"]["resource_images"],
    )
