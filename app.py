import os
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import streamlit as st

from backend.config import BACKEND_URL
from src.api_client import APIClient
from src.chunking import chunk_pages
try:
    from src.citations import format_citations_html, format_citations_markdown
except ImportError:
    import importlib
    import src.citations
    importlib.reload(src.citations)
    from src.citations import format_citations_html, format_citations_markdown
from src.generation import generate_answer
from src.ingestion import extract_text_from_bytes
from src.retrieval import retrieve_context
from src.vector_store import get_vector_store

load_dotenv(override=True)

DOCUMENTS_DIR = Path("data") / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path("data") / "documents_registry.json"

st.set_page_config(
    page_title="Textile Intelligence — Grounded RAG Platform",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&display=swap');

:root {
    --paper: #f7f5ef;
    --card: #ffffff;
    --card-subtle: #fbfaf6;
    --ink: #1a221c;
    --ink-light: #445046;
    --muted: #6e7970;
    --line: #dedad0;
    --line-subtle: #eae7de;
    --forest-dark: #1b3527;
    --forest: #2a4c3a;
    --forest-soft: #385f4b;
    --sage-bg: #edf3ec;
    --sage-border: #b8ccb6;
    --rust: #b3593b;
    --accent: #d4e2d2;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
    z-index: 999990 !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
    pointer-events: none !important;
}

/* Hide only the clutter: Cloud share/star/edit/menu toolbar actions and decoration */
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"],
#MainMenu,
.stAppDeployButton,
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* Force expand sidebar arrow button to be visible, styled, and clickable in all Streamlit versions */
button[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapseButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    background: #ffffff !important;
    border: 1.5px solid #dedad0 !important;
    border-radius: 9px !important;
    padding: 4px 8px !important;
    margin: 8px 0 0 8px !important;
    box-shadow: 0 2px 8px rgba(27, 53, 39, 0.08) !important;
    cursor: pointer !important;
    z-index: 999999 !important;
    transition: all 0.2s ease !important;
}

button[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover {
    background: #edf3ec !important;
    border-color: #b8ccb6 !important;
    transform: scale(1.05) !important;
}

button[data-testid="stExpandSidebarButton"] svg,
button[data-testid="stExpandSidebarButton"] span,
[data-testid="stSidebarCollapsedControl"] svg,
button[data-testid="stSidebarCollapseButton"] svg {
    color: #1b3527 !important;
    fill: #1b3527 !important;
    stroke: #1b3527 !important;
    visibility: visible !important;
    opacity: 1 !important;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', -apple-system, sans-serif;
}

.block-container {
    max-width: 1440px;
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

.stApp h1, .stApp h2, .stApp h3 {
    font-family: 'Playfair Display', Georgia, serif;
    color: var(--forest-dark);
    letter-spacing: -0.02em;
}
.stApp h1 { font-size: clamp(2.2rem, 3.8vw, 3.4rem); font-weight: 600; line-height: 1.1; }
.stApp h2 { font-size: 1.8rem; font-weight: 500; }
.stApp h3 { font-size: 1.35rem; font-weight: 500; }
.stApp p, .stApp label, .stApp span { color: var(--ink); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e4d3c 0%, #243f31 100%) !important;
    border-right: 1px solid #3d634f !important;
}
[data-testid="stSidebar"] > div:first-child { background: transparent !important; }

.brand-card {
    padding: 0.6rem 0.2rem 1.2rem;
    border-bottom: 1px solid #3d634f;
    margin-bottom: 1rem;
}
.brand-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #e5a788;
    text-transform: uppercase;
    font-weight: 600;
}
.brand-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.7rem;
    line-height: 1.05;
    color: #f7f6f0 !important;
    margin: 0.35rem 0 0.2rem;
}
.brand-title span { color: #e5a788; }
.brand-sub {
    font-size: 0.8rem;
    color: #c4d7c9 !important;
    margin: 0;
    line-height: 1.4;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px !important;
    display: flex !important;
    flex-direction: column !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.8rem !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"],
[data-testid="stSidebar"] [role="radiogroup"] > label {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    margin: 3px 0 !important;
    cursor: pointer !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
    backdrop-filter: blur(4px) !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"]:hover,
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.14) !important;
    border-color: rgba(255, 255, 255, 0.28) !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: #ffffff !important;
    border: 1px solid #ded9cd !important;
    border-left: 5px solid #b3593b !important;
    transform: translateX(6px) !important;
    box-shadow: 0 6px 18px rgba(10, 25, 17, 0.25) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #e0ece3 !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) span {
    color: #173323 !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
}

.status-badge-container {
    background: rgba(0, 0, 0, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 0.8rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5ee08b;
    font-weight: 600;
    margin-bottom: 5px;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #5ee08b;
    box-shadow: 0 0 8px #5ee08b;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    margin-top: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: #f7f6f0 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    fill: #f7f6f0 !important;
    color: #f7f6f0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
    padding: 10px 12px !important;
    background: rgba(0, 0, 0, 0.15) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label span {
    color: #e5efe7 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #15271d !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder {
    color: #6d7e72 !important;
    opacity: 0.9 !important;
}
[data-testid="stSidebar"] .stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.hero-box {
    position: relative;
    overflow: hidden;
    padding: 2.4rem 2.6rem;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 18px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 20px rgba(27, 53, 39, 0.04);
}
.hero-box::after {
    content: 'TI';
    position: absolute;
    right: 2rem;
    top: -3.5rem;
    font-family: 'Playfair Display', serif;
    font-size: 15rem;
    line-height: 1;
    color: #f1ede2;
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}
.hero-content { position: relative; z-index: 1; }
.hero-eyebrow {
    display: inline-block;
    color: var(--rust) !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.hero-p {
    color: #4b544d !important;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 740px;
    margin-top: 0.8rem;
}

.arch-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.5rem;
    height: 100%;
    box-sizing: border-box;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.arch-card:hover {
    transform: translateY(-3px);
    border-color: #bccbbd;
    box-shadow: 0 8px 24px rgba(27, 53, 39, 0.06);
}
.arch-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: var(--rust) !important;
    text-transform: uppercase;
    font-weight: 600;
}
.arch-card h3 { font-size: 1.35rem; margin: 0.6rem 0 0.5rem; }
.arch-card p { color: #555f57 !important; font-size: 0.92rem; line-height: 1.6; }

.stButton > button:not([data-testid="stBaseButton-headerNoPadding"]),
.stDownloadButton > button {
    border-radius: 10px !important;
    min-height: 42px !important;
    padding: 8px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #edf4ec 0%, #dbe9da 100%) !important;
    color: #1b3527 !important;
    border: 1px solid #b2c9b1 !important;
    box-shadow: 0 2px 6px rgba(27, 53, 39, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
}
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p {
    color: #1b3527 !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #e4efe2 0%, #cfe2ce 100%) !important;
    border-color: #9ab999 !important;
    box-shadow: 0 5px 14px rgba(27, 53, 39, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    transform: translateY(-1.5px) !important;
}

.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: #ffffff !important;
    color: #2e3a31 !important;
    border: 1px solid #ded9cd !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) * {
    color: #2e3a31 !important;
    font-weight: 600 !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background: #fbf9f4 !important;
    border-color: #b8cbb8 !important;
    transform: translateY(-1.5px) !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stTooltipIcon"],
[data-testid="stTooltipHoverTarget"],
div[data-testid="stTooltipHoverTarget"] button,
button[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stHelpIcon"] {
    all: unset !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 18px !important;
    height: 18px !important;
    padding: 0 !important;
    margin: 0 0 0 6px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    cursor: help !important;
}
[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipHoverTarget"] svg,
button[data-testid="stBaseButton-headerNoPadding"] svg,
[data-testid="stHelpIcon"] svg {
    display: block !important;
    width: 16px !important;
    height: 16px !important;
    fill: #557764 !important;
    color: #557764 !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"] section {
    background: #faf8f2 !important;
    border: 1.5px dashed #c6d3c3 !important;
    border-radius: 12px !important;
    padding: 24px 20px !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #8dae89 !important;
    background: #f4f8f2 !important;
}
[data-testid="stFileUploader"] section button {
    background: #ffffff !important;
    color: #1b3527 !important;
    border: 1px solid #b8ccb6 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    margin: 0.75rem 0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #c9c5b8 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
}

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
}

.prompt-pill-btn > button {
    background: #ffffff !important;
    border: 1px solid #d5d0c2 !important;
    color: #354238 !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    min-height: 36px !important;
    padding: 6px 14px !important;
    text-align: left !important;
    width: 100% !important;
}
.prompt-pill-btn > button:hover {
    background: #f1f7ef !important;
    border-color: #a4c0a1 !important;
    color: #1b3527 !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    min-height: 115px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    color: var(--forest-dark) !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: clamp(1.15rem, 1.55vw, 1.65rem) !important;
    line-height: 1.25 !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

/* ==============================================================================
   Smooth Scrolling, Entrance Animations & Micro-Interactions
   ============================================================================== */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, section.main {
    scroll-behavior: smooth !important;
    -webkit-overflow-scrolling: touch !important;
    overscroll-behavior-y: auto !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

@keyframes fadeSlideUp {
    0% {
        opacity: 0;
        transform: translateY(16px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes gentleCardFloat {
    0% {
        opacity: 0;
        transform: translateY(12px) scale(0.995);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* Staggered entrance animations */
.hero-box {
    animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease, border-color 0.25s ease;
}
.hero-box:hover {
    transform: translateY(-2px);
    border-color: #c9d8c8;
    box-shadow: 0 8px 28px rgba(27, 53, 39, 0.06);
}

[data-testid="stMetric"] {
    animation: fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease, border-color 0.25s ease !important;
}
[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { animation-delay: 0.04s; }
[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { animation-delay: 0.08s; }
[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { animation-delay: 0.12s; }
[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { animation-delay: 0.16s; }

[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    border-color: #b8ccb6 !important;
    box-shadow: 0 10px 26px rgba(27, 53, 39, 0.07) !important;
}

.arch-card {
    animation: fadeSlideUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: transform 0.32s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.32s ease, border-color 0.25s ease !important;
}
.arch-card:hover {
    transform: translateY(-4px) !important;
    border-color: #a4c0a1 !important;
    box-shadow: 0 12px 28px rgba(27, 53, 39, 0.08) !important;
}

[data-testid="stChatMessage"] {
    animation: fadeSlideUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease !important;
}
[data-testid="stChatMessage"]:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 6px 20px rgba(27, 53, 39, 0.04) !important;
}

.prompt-pill-btn {
    animation: fadeSlideUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.prompt-pill-btn:nth-child(1) { animation-delay: 0.05s; }
.prompt-pill-btn:nth-child(2) { animation-delay: 0.1s; }
.prompt-pill-btn:nth-child(3) { animation-delay: 0.15s; }

.prompt-pill-btn > button {
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
.prompt-pill-btn > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 6px 18px rgba(27, 53, 39, 0.08) !important;
    border-color: #a4c0a1 !important;
    color: #1b3527 !important;
}
.prompt-pill-btn > button:active {
    transform: translateY(0) scale(0.99) !important;
}

.stButton > button {
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}


/* ==============================================================================
   Distinct Verified Sources & Evidence Cards Styling
   ============================================================================== */
.citation-evidence-box {
    margin-top: 1.25rem;
    margin-bottom: 0.75rem;
    background: linear-gradient(180deg, #f7f9f6 0%, #edf3eb 100%);
    border: 1px solid #c9d8c7;
    border-left: 4px solid #2a4c3a;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(27, 53, 39, 0.04);
}

.citation-box-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 9px;
    margin-bottom: 11px;
    border-bottom: 1px dashed #cad9c9;
}

.citation-box-title {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #b3593b;
    text-transform: uppercase;
}

.citation-header-icon {
    font-size: 0.95rem;
}

.citation-total-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.70rem;
    font-weight: 600;
    color: #243f31;
    background: #e2ede0;
    border: 1px solid #c2d7bf;
    padding: 2px 9px;
    border-radius: 12px;
}

.citation-items-container {
    display: flex;
    flex-direction: column;
    gap: 9px;
}

.citation-item-card {
    background: #ffffff;
    border: 1px solid #dae5d9;
    border-radius: 7px;
    padding: 10px 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.citation-item-card:hover {
    transform: translateY(-1.5px);
    border-color: #a4c0a2;
    box-shadow: 0 4px 12px rgba(27, 53, 39, 0.06);
}

.citation-item-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
}

.citation-index-badge {
    background: #1b3527;
    color: #ffffff;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    line-height: 1.2;
}

.citation-doc-name {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    color: #1b3527;
    word-break: break-word;
    margin-right: auto;
}

.citation-meta-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    line-height: 1.2;
    white-space: nowrap;
}

.citation-meta-pill.page-pill {
    background: #f0f4ef;
    color: #385f4b;
    border: 1px solid #cedecc;
    font-weight: 500;
}

.citation-meta-pill.rel-pill {
    background: #eaf4e8;
    color: #1e6933;
    border: 1px solid #bde2bf;
    font-weight: 600;
}

.citation-snippet-quote {
    font-family: 'DM Sans', -apple-system, sans-serif;
    font-size: 0.83rem;
    line-height: 1.48;
    color: #4a574d;
    font-style: italic;
    background: #fbfdfa;
    border-left: 3px solid #8eaf8b;
    padding: 7px 12px;
    border-radius: 0 5px 5px 0;
    margin-top: 7px;
    word-break: break-word;
}

[data-testid="stChatMessage"] blockquote {
    background: #f6f9f5 !important;
    border-left: 3px solid #8fae89 !important;
    color: #445348 !important;
    font-style: italic !important;
    padding: 6px 14px !important;
    border-radius: 0 6px 6px 0 !important;
    margin: 8px 0 !important;
}

</style>
""", unsafe_allow_html=True)

vector_store = get_vector_store()


def get_chunks_count_safe(vs, source: Optional[str] = None, document_id: Optional[str] = None) -> int:
    """Safe chunk counting with fallback to collection.get."""
    if hasattr(vs, "get_document_chunks_count"):
        try:
            return vs.get_document_chunks_count(document_id=document_id, source=source)
        except Exception:
            pass
    try:
        where_clause = {"document_id": document_id} if document_id else ({"source": source} if source else None)
        if where_clause:
            res = vs.collection.get(where=where_clause, include=[])
            return len(res.get("ids", []))
    except Exception:
        pass
    return 0


def delete_document_safe(vs, document_id: Optional[str] = None, source: Optional[str] = None) -> int:
    """Safe deletion using metadata filter with fallback to collection.delete."""
    if document_id and hasattr(vs, "delete_by_document_id"):
        try:
            vs.delete_by_document_id(document_id)
        except Exception:
            pass
    elif document_id:
        try:
            vs.collection.delete(where={"document_id": document_id})
        except Exception:
            pass
    if source:
        try:
            vs.delete_document(source)
        except Exception:
            pass
    return vs.count()


def load_documents_registry() -> Dict[str, Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_documents_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"Error saving registry: {e}")


def sync_registry_with_vector_store() -> Dict[str, Dict[str, Any]]:
    registry = load_documents_registry()
    try:
        indexed_sources = vector_store.get_indexed_sources()
    except Exception:
        indexed_sources = []

    registered_filenames = {item["filename"]: doc_id for doc_id, item in registry.items()}
    modified = False

    for source in indexed_sources:
        if source not in registered_filenames:
            doc_id = f"doc_{hashlib.md5(source.encode()).hexdigest()[:10]}"
            chunk_cnt = get_chunks_count_safe(vector_store, source=source)
            found_path = ""
            for possible_path in [
                DOCUMENTS_DIR / source,
                Path("lace_knowledge_base_deliverables") / source,
                Path("data") / source
            ]:
                if possible_path.exists():
                    found_path = str(possible_path)
                    break

            file_sz = os.path.getsize(found_path) if found_path and os.path.exists(found_path) else 0

            registry[doc_id] = {
                "document_id": doc_id,
                "filename": source,
                "file_path": found_path,
                "file_size": file_sz,
                "upload_date": "2026-10-30",
                "chunk_count": chunk_cnt,
                "page_count": "Indexed",
                "status": "Indexed"
            }
            modified = True

    if modified:
        save_documents_registry(registry)

    return registry


def ensure_default_documents_indexed():
    """Auto-seed bundled technical reference PDFs if ChromaDB is empty on fresh cloud deployment."""
    if vector_store.count() == 0:
        deliverables_dir = Path("lace_knowledge_base_deliverables")
        if deliverables_dir.exists():
            pdfs = sorted(deliverables_dir.glob("*.pdf"))
            for pdf_path in pdfs:
                try:
                    pages = extract_text_from_bytes(pdf_path.read_bytes(), pdf_path.name)
                    if pages:
                        doc_id = f"doc_{hashlib.md5(pdf_path.name.encode()).hexdigest()[:10]}"
                        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100, default_source=pdf_path.name)
                        for c in chunks:
                            c["metadata"]["document_id"] = doc_id
                            c["metadata"]["source"] = pdf_path.name
                        vector_store.add_chunks(chunks)
                except Exception as e:
                    print(f"Error bootstrapping {pdf_path.name}: {e}")


ensure_default_documents_indexed()
sync_registry_with_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None

if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None

# Initialize FastAPI client with session token from session state or query param
token_candidate = st.session_state.get("auth_token") or st.query_params.get("token")
api_client = APIClient(base_url=BACKEND_URL, session_token=token_candidate)

# Auto-validate token on page load/refresh
if token_candidate and "user" not in st.session_state:
    ok_me, user_res = api_client.get_me()
    if ok_me and isinstance(user_res, dict):
        st.session_state.auth_token = token_candidate
        st.session_state.user = user_res
        st.session_state.logged_in = True
        st.session_state.is_admin = (user_res.get("role") == "admin")
        st.query_params["token"] = token_candidate
    else:
        st.session_state.auth_token = None
        st.session_state.user = None
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        if "token" in st.query_params:
            del st.query_params["token"]

def display_clean_error(title: str, error_msg: Any):
    """Present a clean, professional user alert with optional collapsible technical details."""
    raw = str(error_msg)
    is_technical = any(k in raw for k in (
        "SSL handshake", "TLSV1_ALERT", "Traceback", "TopologyDescription",
        "configured timeouts", "AutoReconnect", "ServerDescription", "socketTimeoutMS"
    ))
    if is_technical:
        st.error(f"⚠️ **{title}**: Database service is temporarily unreachable.")
        with st.expander("🛠️ View Diagnostic Details", expanded=False):
            st.code(raw, language="text")
            st.caption("Hint: Check MongoDB Atlas Network Access to ensure `0.0.0.0/0` is Active.")
    else:
        st.error(f"⚠️ **{title}**: {raw}")

def render_auth_portal(client: APIClient):
    st.markdown("""
    <div class="hero-box" style="text-align: center; margin: 2rem auto; max-width: 580px;">
        <span class="hero-eyebrow">ENTERPRISE SECURITY & KNOWLEDGE ARCHIVE</span>
        <h1 style="font-size: 2.2rem; margin-top: 0.3rem;">LabhTex Intelligence<span>.</span></h1>
        <p class="hero-p" style="margin: 0 auto; max-width: 460px;">
            Authenticated access for textile engineers, fabric consultants, and authorized quality auditors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register Account"])

        with tab_login:
            with st.form("portal_login_form"):
                st.markdown("##### Member Sign In")
                login_email = st.text_input("Work Email", placeholder="engineer@labhtex.com")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                btn_login = st.form_submit_button("Sign In to Workspace", use_container_width=True)

                if btn_login:
                    if not login_email or not login_password:
                        st.error("Please enter both email and password.")
                    else:
                        ok, res = client.login(login_email, login_password)
                        if ok:
                            st.session_state.auth_token = res["token"]
                            st.session_state.user = res["user"]
                            st.session_state.logged_in = True
                            st.session_state.is_admin = (res["user"].get("role") == "admin")
                            st.query_params["token"] = res["token"]
                            st.rerun()
                        else:
                            display_clean_error("Authentication Failed", res)

        with tab_register:
            with st.form("portal_register_form"):
                st.markdown("##### Account Registration")
                reg_name = st.text_input("Full Name", placeholder="Arjun Sharma")
                reg_email = st.text_input("Work Email", placeholder="arjun@labhtex.com")
                reg_password = st.text_input("Password (min 12 characters)", type="password", placeholder="Create secure password")

                with st.expander("🛡️ Administrator Provisioning (Optional)"):
                    st.caption("Provide the administrator secret key if registering with administrative privileges.")
                    reg_admin_key = st.text_input("Admin Registration Key", type="password", placeholder="Admin registration secret")

                btn_reg = st.form_submit_button("Register Account", use_container_width=True)
                if btn_reg:
                    if not reg_name or not reg_email or not reg_password:
                        st.error("Please fill in all required fields.")
                    elif len(reg_password) < 12:
                        st.error("Password must be at least 12 characters long.")
                    else:
                        ok, res = client.register(reg_name, reg_email, reg_password, admin_key=reg_admin_key if reg_admin_key else None)
                        if ok:
                            role_title = "Administrator" if res.get("role") == "admin" else "Textile Specialist"
                            st.success(f"✓ Account successfully created as **{role_title}**! Please switch to 'Sign In' to log in.")
                        else:
                            display_clean_error("Registration Failed", res)

# Gate unauthenticated sessions
if not st.session_state.get("logged_in", False):
    render_auth_portal(api_client)
    st.stop()

is_openrouter = bool(
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("LLM_PROVIDER") == "openrouter"
    or (os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY").startswith("sk-or-"))
)
active_llm_name = (
    f"OpenRouter ({os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct')})"
    if is_openrouter
    else f"Gemini ({os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')})"
)
api_key_set = bool(
    (os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "your_openrouter_api_key_here")
    or (os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here")
)

with st.sidebar:
    st.markdown("""
    <div class="brand-card">
        <div class="brand-tag">ENTERPRISE RAG PLATFORM</div>
        <div class="brand-title">LabhTex<br>Intelligence<span>.</span></div>
        <p class="brand-sub">Technical Auditing & Knowledge Workspace</p>
    </div>
    """, unsafe_allow_html=True)

    user_info = st.session_state.get("user", {})
    user_name = user_info.get("name", "User")
    user_email = user_info.get("email", "")
    is_admin = (user_info.get("role") == "admin")

    if is_admin:
        st.markdown(f"""
        <div style="background: rgba(42, 76, 58, 0.45); border: 1px solid #5ee08b; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #9fe2b5; letter-spacing: 0.12em; text-transform: uppercase;">Active Identity</div>
            <div style="font-size: 0.88rem; font-weight: 600; color: #ffffff; display: flex; align-items: center; gap: 6px; margin-top: 2px;">
                <span class="pulse-dot"></span> 🛡️ {user_name}
            </div>
            <div style="font-size: 0.72rem; color: #d0edd8; margin-top: 2px;">{user_email} (Admin)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(0,0,0,0.18); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #b8ccb6; letter-spacing: 0.12em; text-transform: uppercase;">Active Identity</div>
            <div style="font-size: 0.88rem; font-weight: 600; color: #f7f6f0; margin-top: 2px;">👤 {user_name}</div>
            <div style="font-size: 0.72rem; color: #a4bea9; margin-top: 2px;">{user_email}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚪 Log Out", use_container_width=True):
        api_client.logout()
        st.session_state.auth_token = None
        st.session_state.user = None
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        if "token" in st.query_params:
            del st.query_params["token"]
        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Conversation Archive (Persistent in MongoDB Atlas)
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:0.68rem;letter-spacing:0.12em;color:#b8ccb6;margin-bottom:6px;text-transform:uppercase;'>CONVERSATIONS</div>", unsafe_allow_html=True)

    if st.button("➕ New Technical Inquiry", use_container_width=True):
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        st.rerun()

    ok_convs, conv_list = api_client.list_conversations()
    if ok_convs and conv_list:
        curr_id = st.session_state.get("current_conv_id")
        for c in conv_list[:12]:
            c_id = c["id"]
            c_title = c.get("title", "Inquiry")
            is_active = (c_id == curr_id)
            c_msgs = c.get("message_count", 0)
            prefix = "🟢 " if is_active else "💬 "
            label = f"{prefix}{c_title[:22]} ({c_msgs})"

            if st.button(label, key=f"conv_select_{c_id}", use_container_width=True):
                st.session_state.current_conv_id = c_id
                ok_m, msgs = api_client.get_messages(c_id)
                if ok_m:
                    st.session_state.messages = msgs
                st.rerun()

        # If an active conversation is selected, provide Rename & Delete
        if curr_id:
            with st.expander("⚙️ Manage Active Chat"):
                rename_title = st.text_input("New Title", key="input_rename_title")
                if st.button("Rename", key="btn_apply_rename", use_container_width=True):
                    if rename_title.strip():
                        api_client.rename_conversation(curr_id, rename_title.strip())
                        st.rerun()
                if st.button("🗑️ Delete Chat", key="btn_del_chat", use_container_width=True):
                    api_client.delete_conversation(curr_id)
                    st.session_state.current_conv_id = None
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No previous chats recorded.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    if st.session_state.is_admin:
        available_pages = ["AI Assistant", "Knowledge Base", "Overview"]
        nav_labels = {
            "AI Assistant": "01 / AI ASSISTANT",
            "Knowledge Base": "02 / KNOWLEDGE BASE",
            "Overview": "03 / OVERVIEW",
        }
    else:
        available_pages = ["AI Assistant", "Overview"]
        nav_labels = {
            "AI Assistant": "01 / AI ASSISTANT",
            "Overview": "02 / OVERVIEW",
        }

    page = st.radio(
        "WORKSPACE NAVIGATION",
        options=available_pages,
        format_func=lambda x: nav_labels.get(x, x),
        label_visibility="collapsed",
    )

    if st.session_state.is_admin:
        st.markdown(f"""
        <div class="status-badge-container">
            <div class="status-pill"><span class="pulse-dot"></span> Pipeline Active</div>
            <div style="font-size:0.75rem; color:#d8eedf; line-height:1.7;">
                • PyMuPDF Extraction<br>
                • all-MiniLM-L6-v2 (384d)<br>
                • ChromaDB Persistent<br>
                • {active_llm_name}
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.15); margin-top: 8px; padding-top: 8px; font-family:'DM Mono',monospace; font-size:0.73rem; line-height: 1.6;">
                <div style="color: #bde4c8;">Vector Database: <strong style="color: #ffffff;">{vector_store.count()} chunks</strong></div>
                <div style="color: #bde4c8;">LLM API: <strong style="color: {'#5ee08b' if api_key_set else '#f3a988'};">{'Connected' if api_key_set else 'Key Missing'}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 10px; padding: 12px 14px; margin-top: 1.2rem;">
            <div style="display: flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 0.74rem; color: #5ee08b; font-weight: 600;">
                <span class="pulse-dot"></span> Approved Knowledge Base Active
            </div>
            <div style="font-size: 0.74rem; color: #c4d7c9; margin-top: 5px; line-height: 1.45;">
                Verified textile reference standards, ISO/ASTM test methods & operations active.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Backend Security Enforcement
if page in ["Knowledge Base"]:
    if not st.session_state.get("is_admin", False):
        st.error("⛔ Unauthorized Access: Administrative authentication required.")
        st.stop()

if page == "Overview":
    st.markdown("""
    <div class="hero-box">
        <div class="hero-content">
            <span class="hero-eyebrow">OVERVIEW & ARCHITECTURAL TELEMETRY</span>
            <h1>Real Database Statistics</h1>
            <p class="hero-p">
                Real-time operational metrics across persistent ChromaDB collections, vector embeddings,
                and indexed technical textile literature.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    registry = sync_registry_with_vector_store()
    total_indexed_sources = len(vector_store.get_indexed_sources())
    total_chunks = vector_store.count()
    collection_name = vector_store.collection_name
    embedding_model = "all-MiniLM-L6-v2 (384-dimensional dense vectors)"
    db_connected = True

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Indexed PDFs", total_indexed_sources)
    with m2:
        st.metric("Total Vector Chunks", total_chunks)
    with m3:
        st.metric("Collection Name", collection_name)
    with m4:
        st.metric("Database Status", "🟢 Connected" if db_connected else "🔴 Offline")

    st.divider()

    col_left, col_right = st.columns([1.5, 1], gap="large")

    with col_left:
        st.subheader("Persistent Vector Collection Inventory")
        if not registry:
            st.info("No documents are currently indexed in ChromaDB.")
        else:
            table_rows = []
            for doc_id, doc in registry.items():
                chunk_cnt = doc.get("chunk_count", 0)
                share = (chunk_cnt / total_chunks * 100) if total_chunks > 0 else 0
                table_rows.append({
                    "Document ID": doc_id,
                    "Filename": doc.get("filename", "Unknown"),
                    "Chunks": chunk_cnt,
                    "Collection Share": f"{share:.1f}%",
                    "Status": doc.get("status", "Indexed")
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("System Infrastructure")
        st.markdown(f"""
        <div class="arch-card">
            <div class="arch-label">RUNTIME SPECIFICATIONS</div>
            <p style="margin-top: 0.8rem; line-height: 1.8;">
                <strong>Vector DB:</strong> ChromaDB Persistent<br>
                <strong>Storage Path:</strong> <code>data/chroma_db</code><br>
                <strong>Documents Path:</strong> <code>data/documents</code><br>
                <strong>Distance Metric:</strong> Cosine Similarity (1 - distance)<br>
                <strong>Embedding Model:</strong> <code>{embedding_model}</code><br>
                <strong>Synthesis Model:</strong> <code>{active_llm_name}</code><br>
                <strong>Chunk Geometry:</strong> 500 chars (100 char overlap)
            </p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Knowledge Base":
    st.markdown("<div class='hero-eyebrow'>ADMINISTRATIVE KNOWLEDGE BASE MANAGEMENT</div>", unsafe_allow_html=True)
    st.title("Document Library & Vector Management")
    st.caption("Upload reference technical PDFs, inspect indexed documents, view chunk status, and safely delete individual documents.")

    with st.expander("📤 Upload & Index Technical PDF Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Select Technical PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload digital PDFs containing textile specifications, standards, or manuals.",
            key="admin_pdf_uploader",
        )

        if uploaded_files:
            st.caption(f"📄 **{len(uploaded_files)} PDF file(s)** staged for processing.")

            if st.button("⚡ Process & Index Selected PDFs into ChromaDB", type="primary"):
                if not st.session_state.get("is_admin", False):
                    st.error("⛔ Unauthorized action: Administrative privileges required.")
                    st.stop()

                registry = sync_registry_with_vector_store()
                registered_filenames = {item["filename"]: doc_id for doc_id, item in registry.items()}

                processed_count = 0
                skipped_count = 0
                progress_bar = st.progress(0, text="Extracting and indexing text...")

                for idx, file in enumerate(uploaded_files):
                    if file.name in registered_filenames:
                        skipped_count += 1
                        progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Already registered: {file.name}")
                        continue

                    try:
                        document_id = f"doc_{uuid.uuid4().hex[:10]}"
                        saved_file_path = DOCUMENTS_DIR / f"{document_id}_{file.name}"
                        saved_file_path.write_bytes(file.getvalue())

                        page_items = extract_text_from_bytes(file.getvalue(), file.name)
                        for item in page_items:
                            item["source"] = file.name

                        if not page_items:
                            st.warning(f"No selectable text found in {file.name} (scanned/bitmap PDF).")
                            status = "No selectable text"
                            chunks = []
                        else:
                            chunks = chunk_pages(page_items, chunk_size=500, chunk_overlap=100, default_source=file.name)
                            for c in chunks:
                                c["metadata"]["document_id"] = document_id
                                c["metadata"]["source"] = file.name

                            vector_store.add_chunks(chunks)
                            status = f"Indexed ({len(chunks)} chunks)"

                        registry[document_id] = {
                            "document_id": document_id,
                            "filename": file.name,
                            "file_path": str(saved_file_path),
                            "file_size": file.size,
                            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "chunk_count": len(chunks),
                            "page_count": len(page_items),
                            "status": status
                        }
                        save_documents_registry(registry)
                        processed_count += 1

                    except Exception as err:
                        st.error(f"Error processing {file.name}: {err}")

                    progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Processed {idx + 1} of {len(uploaded_files)} PDF(s)")

                progress_bar.empty()

                if processed_count:
                    st.success(f"✓ Successfully processed and indexed {processed_count} PDF(s) into ChromaDB.")
                    st.rerun()
                if skipped_count:
                    st.info(f"Skipped {skipped_count} PDF(s) already in the database.")

    st.divider()
    st.subheader("Indexed Documents Management")
    registry = sync_registry_with_vector_store()

    if not registry:
        st.info("No documents are currently registered. Upload reference PDFs above to populate your knowledge base.")
    else:
        if st.session_state.confirm_delete_id:
            del_id = st.session_state.confirm_delete_id
            target_doc = registry.get(del_id)

            if target_doc:
                with st.container(border=True):
                    st.markdown(f"""
                    <div style="color: #b3593b; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px;">
                        ⚠️ Confirm Document Deletion
                    </div>
                    <p style="margin-bottom: 12px; color: #333;">
                        Are you sure you want to permanently delete <strong>{target_doc.get('filename')}</strong> (ID: <code>{del_id}</code>)?<br>
                        This will delete <strong>all {target_doc.get('chunk_count', 0)} associated chunks and embeddings</strong> from ChromaDB using metadata filter <code>document_id='{del_id}'</code>, and delete the stored physical PDF from file storage.<br>
                        <em>Note: The rest of the ChromaDB collection will remain untouched.</em>
                    </p>
                    """, unsafe_allow_html=True)

                    btn_col1, btn_col2, _ = st.columns([1.5, 1, 3])
                    with btn_col1:
                        if st.button("🚨 Confirm Permanent Deletion", key=f"confirm_del_btn_{del_id}", type="primary"):
                            if not st.session_state.get("is_admin", False):
                                st.error("⛔ Unauthorized action: Administrative privileges required.")
                                st.stop()

                            try:
                                filename = target_doc.get("filename", "")
                                chunk_cnt = target_doc.get("chunk_count", 0)

                                delete_document_safe(vector_store, document_id=del_id, source=filename)

                                file_path = target_doc.get("file_path")
                                if file_path and os.path.exists(file_path):
                                    try:
                                        os.remove(file_path)
                                    except Exception:
                                        pass

                                registry.pop(del_id, None)
                                save_documents_registry(registry)

                                st.session_state.confirm_delete_id = None
                                st.success(f"✓ Successfully deleted '{filename}' and purged its {chunk_cnt} chunks from ChromaDB.")
                                st.rerun()

                            except Exception as del_err:
                                st.error(f"Error during document deletion: {del_err}")

                    with btn_col2:
                        if st.button("Cancel", key=f"cancel_del_btn_{del_id}"):
                            st.session_state.confirm_delete_id = None
                            st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        for doc_id, doc in list(registry.items()):
            with st.container(border=True):
                c_main, c_date, c_chunks, c_status, c_action = st.columns([3, 1.6, 1.4, 1.4, 1.2])

                with c_main:
                    st.markdown(f"**📄 {doc.get('filename', 'Unknown')}**")
                    size_kb = doc.get("file_size", 0) / 1024
                    st.caption(f"ID: `{doc_id}` · {size_kb:.1f} KB")

                with c_date:
                    st.markdown("**Upload Date**")
                    st.caption(doc.get("upload_date", "2026-10-30"))

                with c_chunks:
                    st.markdown("**Indexed Chunks**")
                    st.caption(f"**{doc.get('chunk_count', 0)}** chunks")

                with c_status:
                    st.markdown("**Status**")
                    st.caption("🟢 " + doc.get("status", "Indexed"))

                with c_action:
                    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                    if st.button("Delete", key=f"del_request_{doc_id}", use_container_width=True):
                        st.session_state.confirm_delete_id = doc_id
                        st.rerun()

elif page == "AI Assistant":
    st.markdown("<div class='hero-eyebrow'>CONVERSATIONAL CONSULTANT</div>", unsafe_allow_html=True)
    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.title("Ask the archive.")
        if st.session_state.is_admin:
            st.caption("Administrator Consultation Mode · Answers synthesized strictly from retrieved document evidence.")
        else:
            st.caption("Customer Portal · Queries answered strictly from the approved technical textile knowledge base.")
    with action_col:
        if st.session_state.messages:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if vector_store.count() == 0:
        if st.session_state.is_admin:
            st.warning("⚠️ No documents indexed yet. Go to **Knowledge Base** tab to upload and index technical PDFs.")
        else:
            st.warning("⚠️ Knowledge base is currently updating. Please check back shortly.")

    if not st.session_state.messages and vector_store.count() > 0:
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:0.72rem;letter-spacing:0.12em;color:var(--rust);margin:1.5rem 0 0.6rem;text-transform:uppercase;'>TRY AN EXAMPLE TECHNICAL QUERY</div>", unsafe_allow_html=True)
        q_cols = st.columns(3)
        sample_prompts = [
            "What is the standard pneumatic burst strength test parameter for lace under ISO 13938-2?",
            "What are the kinematic differences between Leavers twist looms and Karl Mayer Raschel lace machines?",
            "What are the critical temperature limits and degradation risks for elastane during heat-setting?"
        ]
        for idx, col in enumerate(q_cols):
            with col:
                st.markdown('<div class="prompt-pill-btn">', unsafe_allow_html=True)
                if st.button(sample_prompts[idx], key=f"quick_prompt_{idx}"):
                    st.session_state["submitted_prompt"] = sample_prompts[idx]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("citations_md"):
                with st.expander("📑 Sources & Citations"):
                    st.markdown(message["citations_md"])

            chunks = message.get("retrieved_chunks", [])
            if chunks:
                with st.expander("🔍 View Retrieved Evidence Chunks"):
                    for c in chunks:
                        sim_pct = int(c.get("similarity", 0.0) * 100)
                        st.markdown(f"**[{c.get('source', 'Unknown')}] Page {c.get('page_number', 1)}** (Relevance: `{sim_pct}%`)")
                        st.caption(f"Chunk ID: `{c.get('chunk_id')}`")
                        st.text(c.get("text", ""))
                        st.divider()

    prompt = st.chat_input("Ask a technical question about indexed textile documentation...")
    if "submitted_prompt" in st.session_state:
        prompt = st.session_state.pop("submitted_prompt")

    if prompt:
        # Create conversation in MongoDB if not active
        if not st.session_state.get("current_conv_id"):
            ok_c, new_c = api_client.create_conversation("New Technical Inquiry")
            if ok_c:
                st.session_state.current_conv_id = new_c["id"]

        curr_conv_id = st.session_state.get("current_conv_id")

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing technical response from approved archive..."):
                ok_resp, msg_data = api_client.send_message(curr_conv_id, prompt)

            if ok_resp:
                # Reload messages directly from MongoDB
                ok_m, msgs = api_client.get_messages(curr_conv_id)
                if ok_m:
                    st.session_state.messages = msgs
                st.rerun()
            else:
                display_clean_error("Response Error", msg_data)
