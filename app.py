import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_house_prices.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

# Hero photograph: save your photo as assets/hero_house.jpg (next to app.py)
ASSETS_DIR = BASE_DIR / "assets"
HERO_IMAGE_NAMES = [
    "hero_house.jpg",
    "hero_house.jpeg",
    "hero_house.png",
    "hero_house.webp",
]

# Optional: a hosted image link. Leave as None to use the built-in
# illustration. A file saved as assets/hero_house.jpg always wins.
HERO_IMAGE_URL = None

# Column order used when the model input is built (same order as the
# original app, so the model receives exactly the same DataFrame layout).
FEATURE_ORDER = [
    "Location",
    "Property_Type",
    "Furnishing",
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_Years",
    "Floor",
    "Parking_Spaces",
    "Construction_Quality",
]


st.set_page_config(
    page_title="EstiNest | Property Price Estimator",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------------

STYLES = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,500&family=Fredoka:wght@600;700&family=Dancing+Script:wght@700&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
    color-scheme: light;
    --ivory: #FBFDFF;
    --surface: #FFFFFF;
    --cream: #EAF3FD;
    --accent: #B9D3EE;
    --button: #3F6FB5;
    --button-dark: #2F5A99;
    --heading: #1F3A5F;
    --body: #3F4E63;
    --border: #D6E4F3;
    --sage: #3FA489;
    --blue: #4F86D6;
    --gold: #D98A2B;
    --clay: #7C6FD6;
    --shadow: 0 6px 22px rgba(31, 58, 95, 0.08);
}

html { scroll-behavior: smooth; color-scheme: light; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main { scroll-behavior: smooth; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--ivory) !important;
    color: var(--body);
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
}

.stApp p, .stApp li, .stApp label, .stApp input, .stApp button {
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
#MainMenu,
footer {
    display: none !important;
}

/* Remove the large side padding */
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    max-width: 100% !important;
    padding: 1.25rem 1.5rem 3rem 1.5rem !important;
}

/* ---------- Header / branding ---------- */
.site-header {
    text-align: center;
    background: linear-gradient(135deg, #E6F0FC 0%, #F1EEFD 100%);
    border: 1.5px solid #CBDDF3;
    border-radius: 28px;
    padding: 26px 76px 24px 76px;
    margin-bottom: 28px;
    box-shadow: var(--shadow);
    scroll-margin-top: 20px;
}

.brand {
    font-family: 'Dancing Script', 'Fredoka', cursive;
    font-weight: 700;
    font-size: clamp(60px, 9.5vw, 108px);
    line-height: 1.2;
    letter-spacing: 1px;
    padding: 0 8px 6px 8px;
    color: #2F5A99;
    background: linear-gradient(90deg, #2F5A99 0%, #4F86D6 50%, #7C6FD6 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.tagline {
    font-family: 'DM Sans', sans-serif;
    font-style: italic;
    font-weight: 500;
    font-size: clamp(19px, 2.5vw, 25px);
    color: var(--body);
    margin-top: 2px;
}

/* ---------- Hamburger navigation (pure CSS, no JavaScript) ---------- */
#nav-closed {
    position: fixed;
    top: 0;
    left: 0;
    width: 1px;
    height: 1px;
    pointer-events: none;
}

.hb {
    position: fixed;
    top: 22px;
    right: 24px;
    z-index: 1000002;
    width: 52px;
    height: 52px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-decoration: none !important;
    cursor: pointer;
}

.hb:hover { background: var(--cream); }
.hb:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }

.hb span {
    display: block;
    width: 24px;
    height: 2px;
    margin: 3px 0;
    background: var(--heading);
    border-radius: 2px;
    transition: transform 0.25s ease, opacity 0.25s ease;
}

.hb-close { display: none; }
.hb-close span:nth-child(1) { transform: translateY(8px) rotate(45deg); }
.hb-close span:nth-child(2) { opacity: 0; }
.hb-close span:nth-child(3) { transform: translateY(-8px) rotate(-45deg); }

.nav-panel:target ~ .hb-open { display: none; }
.nav-panel:target ~ .hb-close { display: flex; }

.nav-panel {
    position: fixed;
    inset: 0;
    z-index: 1000001;
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.25s ease, visibility 0.25s ease;
}

.nav-panel:target { visibility: visible; opacity: 1; }

.nav-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(31, 58, 95, 0.14);
}

.nav-drawer {
    position: absolute;
    top: 0;
    right: 0;
    width: min(340px, 86vw);
    height: 100%;
    background: var(--surface);
    border-left: 1px solid var(--border);
    box-shadow: -12px 0 32px rgba(31, 58, 95, 0.08);
    padding: 92px 24px 24px 24px;
    transform: translateX(100%);
    transition: transform 0.3s ease;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.nav-panel:target .nav-drawer { transform: translateX(0); }

.nav-title {
    font-family: 'Fredoka', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--heading);
    padding-bottom: 14px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

.nav-drawer a {
    display: block;
    padding: 14px 16px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    color: var(--heading) !important;
    text-decoration: none !important;
}

.nav-drawer a:hover { background: var(--cream); }

/* ---------- Hero ---------- */
.hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 36px;
    align-items: center;
    background: linear-gradient(115deg, #FFFFFF 0%, #F1F7FD 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 34px;
    box-shadow: var(--shadow);
}

.hero-kicker {
    font-size: 18px;
    font-weight: 700;
    color: var(--clay);
    margin-bottom: 12px;
}

.hero-heading {
    font-family: 'Playfair Display', serif;
    font-size: clamp(38px, 4.8vw, 60px);
    line-height: 1.15;
    font-weight: 700;
    color: var(--heading);
    margin-bottom: 16px;
}

.hero-text {
    font-size: 20.5px;
    line-height: 1.75;
    color: var(--body);
    max-width: 640px;
}

.pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }

.pill {
    padding: 9px 18px;
    border-radius: 999px;
    font-size: 17px;
    font-weight: 600;
    color: var(--heading);
}

.pill.p1 { background: #E6F6F0; border: 1px solid #BFE4D6; }
.pill.p2 { background: #E8F1FC; border: 1px solid #BCD3F0; }
.pill.p3 { background: #EFEDFC; border: 1px solid #D0CBF2; }

.hero-actions { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 26px; }

.stApp a.btn-primary,
.stApp a.btn-secondary {
    display: inline-block;
    padding: 15px 28px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    text-decoration: none !important;
}

.stApp a.btn-primary {
    background: var(--button);
    border: 1px solid var(--button);
    color: #FFFFFF !important;
}

.stApp a.btn-primary:hover { background: var(--button-dark); }

.stApp a.btn-secondary {
    background: var(--surface);
    border: 1px solid var(--accent);
    color: var(--heading) !important;
}

.stApp a.btn-secondary:hover { background: var(--cream); }

.hero-media {
    width: 100%;
    aspect-ratio: 4 / 3;
    max-height: 500px;
    border-radius: 18px;
    overflow: hidden;
    background: linear-gradient(135deg, #E6F0FC, #F1EEFD);
    border: 1px solid var(--border);
}

.hero-media img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.hero-placeholder {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 24px;
    font-size: 17px;
    line-height: 1.7;
    color: var(--body);
}

.hero-placeholder span { font-weight: 700; color: var(--heading); }

/* ---------- Sections ---------- */
.section {
    margin-top: 56px;
    margin-bottom: 22px;
    scroll-margin-top: 24px;
    background: linear-gradient(135deg, #EAF3FD 0%, #F4F1FE 100%);
    border: 1.5px solid #CBDDF3;
    border-left: 8px solid var(--blue);
    border-radius: 20px;
    padding: 22px 28px;
}

.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: clamp(32px, 3.8vw, 44px);
    font-weight: 700;
    color: var(--heading);
    line-height: 1.2;
    margin-bottom: 8px;
}

.section-description {
    font-size: 19px;
    line-height: 1.7;
    color: var(--body);
    max-width: 800px;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.card-grid.four { grid-template-columns: repeat(4, 1fr); }

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--heading);
    margin-bottom: 10px;
}

.card-text {
    font-size: 18px;
    line-height: 1.7;
    color: var(--body);
}

/* How it works: outlined step boxes */
.step {
    background: var(--surface);
    border: 2px solid var(--c);
    border-radius: 20px;
    padding: 24px;
}

.step.s1 { --c: var(--sage); }
.step.s2 { --c: var(--blue); }
.step.s3 { --c: var(--gold); }
.step.s4 { --c: var(--clay); }

.step-number {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: var(--c);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
}

/* About: soft filled boxes with a coloured top edge */
.about-card {
    background: var(--tint);
    border: 1px solid var(--border);
    border-top: 6px solid var(--c);
    border-radius: 16px;
    padding: 26px;
    box-shadow: var(--shadow);
}

.about-card.a1 { --c: var(--sage); --tint: #EDF8F4; }
.about-card.a2 { --c: var(--blue); --tint: #EEF4FD; }
.about-card.a3 { --c: var(--gold); --tint: #F3F1FD; }

/* ---------- Prediction form ---------- */
.stApp div[data-testid="stForm"] {
    background: #F6FAFE !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 24px !important;
    padding: 32px !important;
    box-shadow: var(--shadow);
}

.form-group-title {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--heading);
    padding: 4px 0 4px 14px;
    margin-bottom: 16px;
    border-left: 5px solid var(--c);
}

.form-group-title.g1 { --c: var(--sage); }
.form-group-title.g2 { --c: var(--clay); }

.stApp div[data-testid="stWidgetLabel"] p,
.stApp div[data-testid="stForm"] label p {
    color: var(--heading) !important;
    font-size: 17.5px !important;
    font-weight: 600 !important;
}

/* One pastel-blue look for the selectbox field (new and old Streamlit) */
.stApp [data-testid="stSelectbox"] div:has(> input),
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #EAF3FD !important;
    background-color: #EAF3FD !important;
    border: 1.5px solid #B9D3EE !important;
    border-radius: 16px !important;
    height: 58px !important;
    min-height: 58px !important;
    overflow: hidden;
    box-sizing: border-box;
}

/* Everything inside those fields is transparent so no dark layer shows */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] *,
.stApp [data-testid="stNumberInput"] [data-baseweb="base-input"],
.stApp [data-testid="stNumberInputField"] {
    background: transparent !important;
    background-color: transparent !important;
}

.stApp [data-testid="stSelectbox"] input,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] *,
.stApp [data-testid="stNumberInput"] input {
    color: var(--heading) !important;
    -webkit-text-fill-color: var(--heading) !important;
    font-size: 18px !important;
    font-weight: 500;
}

.stApp [data-testid="stSelectbox"] input,
.stApp [data-testid="stNumberInput"] input {
    background: transparent !important;
    padding-left: 16px !important;
    height: 100% !important;
}

.stApp [data-testid="stSelectbox"] button,
.stApp [data-testid="stSelectbox"] svg {
    background: transparent !important;
    color: var(--button) !important;
    fill: var(--button) !important;
}

.stApp [data-testid="stNumberInputStepDown"],
.stApp [data-testid="stNumberInputStepUp"],
.stApp [data-testid="stNumberInput"] button {
    background: #D6E6F8 !important;
    background-color: #D6E6F8 !important;
    color: var(--heading) !important;
    border: none !important;
}

.stApp [data-testid="stNumberInput"] button svg { fill: var(--heading) !important; }

.stApp [data-testid="stNumberInput"] button:hover {
    background: #BFD6F1 !important;
    background-color: #BFD6F1 !important;
}

.stApp [data-testid="stSelectbox"] div:has(> input):focus-within,
.stApp [data-testid="stNumberInputContainer"]:focus-within,
.stApp [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: var(--button) !important;
    box-shadow: 0 0 0 4px rgba(63, 111, 181, 0.18) !important;
}

/* Dropdown list (opens outside .stApp, so no prefix) */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] *,
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
[role="listbox"],
[role="listbox"] * {
    background-color: #F3F8FE !important;
    color: var(--heading) !important;
    -webkit-text-fill-color: var(--heading) !important;
    font-size: 17px !important;
}

[data-testid="stSelectboxVirtualDropdown"] {
    border: 1.5px solid #B9D3EE !important;
    border-radius: 16px !important;
    overflow: hidden;
}

[role="option"]:hover,
[role="option"]:hover *,
[role="option"][aria-selected="true"],
[role="option"][aria-selected="true"] *,
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li:hover * {
    background-color: #D6E6F8 !important;
}

[data-testid="InputInstructions"] { display: none !important; }

.stApp div[data-testid="stFormSubmitButton"] button {
    background: var(--button) !important;
    border: 1px solid var(--button) !important;
    border-radius: 16px !important;
    min-height: 58px;
    transition: background 0.2s ease;
}

.stApp div[data-testid="stFormSubmitButton"] button p {
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

.stApp div[data-testid="stFormSubmitButton"] button:hover {
    background: var(--button-dark) !important;
    border-color: var(--button-dark) !important;
}

/* ---------- Result (highlighted) ---------- */
.result-box {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #E6F6EE 0%, #F2FAF6 55%, #FFFFFF 100%);
    border: 2px solid #8CCDB5;
    border-radius: 22px;
    padding: 40px 28px 34px 28px;
    text-align: center;
    margin-top: 28px;
    box-shadow: 0 14px 40px rgba(22, 107, 91, 0.16);
    scroll-margin-top: 24px;
}

.result-box::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 8px;
    background: linear-gradient(90deg, #3FA489, #4F86D6, #7C6FD6, #D98A2B);
}

.result-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    background: #CFEEDF;
    color: #0F5A4A;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 14px;
}

.result-caption {
    font-family: 'DM Sans', sans-serif;
    font-size: 25px;
    font-weight: 700;
    color: var(--heading);
}

.result-value {
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(52px, 9vw, 86px);
    font-weight: 800;
    line-height: 1.1;
    color: #0F6B58;
    margin: 10px 0 14px 0;
}

.result-note { font-size: 17px; color: var(--body); }

/* ---------- Footer ---------- */
.footer {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-top: 64px;
    padding: 32px 24px;
    text-align: center;
}

.footer-brand {
    font-family: 'Dancing Script', cursive;
    font-size: 44px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--heading);
}

.footer-tagline {
    font-style: italic;
    font-size: 19px;
    color: var(--button);
    margin: 4px 0 14px 0;
}

.footer-note {
    font-size: 17px;
    line-height: 1.7;
    color: var(--body);
    max-width: 720px;
    margin: 0 auto;
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {
        padding: 1rem 0.9rem 2.5rem 0.9rem !important;
    }
    .hero { grid-template-columns: 1fr; gap: 24px; padding: 22px; }
    .hero-media { max-height: none; }
    .card-grid, .card-grid.four { grid-template-columns: 1fr; }
    .chip-grid { grid-template-columns: repeat(2, 1fr); }
    .stApp div[data-testid="stForm"] { padding: 18px !important; }
    .hb { top: 14px; right: 12px; }
    .site-header { padding: 22px 58px 20px 58px; }
}

@media (max-width: 520px) {
    .chip-grid { grid-template-columns: 1fr; }
}
</style>
"""


def render(html):
    """Render HTML through st.markdown.

    Leading whitespace and blank lines are removed so that Markdown never
    mistakes indented HTML for a code block.
    """
    cleaned = "\n".join(
        line.strip() for line in html.strip().splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


render(STYLES)


# ---------------------------------------------------------------------------
# DATA / MODEL HELPERS (unchanged logic)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def format_inr(amount):
    amount = int(round(amount))
    number = str(abs(amount))

    if len(number) <= 3:
        formatted = number
    else:
        last_three = number[-3:]
        remaining = number[:-3]
        groups = []

        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]

        if remaining:
            groups.insert(0, remaining)

        formatted = ",".join(groups) + "," + last_three

    sign = "-" if amount < 0 else ""
    return f"{sign}₹{formatted}"


def get_numeric_limits(column):
    values = pd.to_numeric(data[column], errors="coerce").dropna()

    minimum = float(values.min())
    maximum = float(values.max())

    if minimum == maximum:
        maximum = minimum + 1

    return minimum, maximum


def get_hero_image_uri():
    """Return the hero photograph as a data URI, or None if it is missing."""
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    for name in HERO_IMAGE_NAMES:
        path = ASSETS_DIR / name

        if path.exists():
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            except OSError:
                return None
            return f"data:{mime_types[path.suffix.lower()]};base64,{encoded}"

    return None


HERO_ILLUSTRATION_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#EAF3FD"/><stop offset="1" stop-color="#F3F0FD"/></linearGradient>
<linearGradient id="table" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#DCE9F8"/><stop offset="1" stop-color="#C8DAF1"/></linearGradient>
<linearGradient id="arrow" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#4F86D6"/><stop offset="1" stop-color="#1F3A5F"/></linearGradient>
</defs>
<rect width="800" height="600" fill="url(#bg)"/>
<circle cx="90" cy="90" r="70" fill="#CFE0F8" opacity="0.55"/>
<circle cx="735" cy="520" r="90" fill="#D9D4F7" opacity="0.5"/>
<circle cx="620" cy="70" r="26" fill="#FFFFFF" opacity="0.7"/>
<rect x="0" y="425" width="800" height="175" fill="url(#table)"/>
<rect x="0" y="425" width="800" height="4" fill="#FFFFFF" opacity="0.8"/>
<ellipse cx="220" cy="438" rx="170" ry="14" fill="#1F3A5F" opacity="0.13"/>
<rect x="290" y="172" width="28" height="70" rx="3" fill="#C9634A"/>
<rect x="100" y="262" width="240" height="168" fill="#FFFFFF" stroke="#C9DAEF" stroke-width="3"/>
<rect x="220" y="265" width="118" height="163" fill="#E3ECF8" opacity="0.7"/>
<polygon points="78,272 220,158 362,272" fill="#E27D60"/>
<polygon points="220,158 362,272 220,272" fill="#B9533C" opacity="0.35"/>
<rect x="78" y="268" width="284" height="12" rx="4" fill="#C9634A"/>
<circle cx="220" cy="226" r="17" fill="#CFE3FA" stroke="#9FC0E8" stroke-width="4"/>
<rect x="120" y="305" width="52" height="48" rx="4" fill="#CFE3FA" stroke="#9FC0E8" stroke-width="4"/>
<line x1="146" y1="305" x2="146" y2="353" stroke="#FFFFFF" stroke-width="4"/>
<rect x="268" y="305" width="52" height="48" rx="4" fill="#CFE3FA" stroke="#9FC0E8" stroke-width="4"/>
<line x1="294" y1="305" x2="294" y2="353" stroke="#FFFFFF" stroke-width="4"/>
<rect x="198" y="350" width="46" height="80" rx="5" fill="#4F86D6"/>
<circle cx="234" cy="392" r="3.5" fill="#FFFFFF"/>
<polyline points="406,338 476,293 516,318 596,233 636,256 708,146" fill="none" stroke="#1F3A5F" stroke-opacity="0.13" stroke-width="22" stroke-linejoin="round" stroke-linecap="round" transform="translate(6 9)"/>
<polyline points="406,338 476,293 516,318 596,233 636,256 708,146" fill="none" stroke="url(#arrow)" stroke-width="22" stroke-linejoin="round" stroke-linecap="round"/>
<polygon points="728,116 730,160 686,130" fill="#1F3A5F" stroke="#1F3A5F" stroke-width="10" stroke-linejoin="round"/>
<ellipse cx="415" cy="506" rx="48" ry="12" fill="#1F3A5F" opacity="0.13"/>
<ellipse cx="415" cy="506" rx="42" ry="13" fill="#C98F2B"/>
<rect x="373" y="498" width="84" height="8" fill="#D9A23A"/>
<ellipse cx="415" cy="498" rx="42" ry="13" fill="#F3C24F"/>
<ellipse cx="415" cy="492" rx="42" ry="13" fill="#C98F2B"/>
<rect x="373" y="484" width="84" height="8" fill="#D9A23A"/>
<ellipse cx="415" cy="484" rx="42" ry="13" fill="#F3C24F"/>
<ellipse cx="415" cy="478" rx="42" ry="13" fill="#C98F2B"/>
<rect x="373" y="470" width="84" height="8" fill="#D9A23A"/>
<ellipse cx="415" cy="470" rx="42" ry="13" fill="#F3C24F"/>
<g transform="rotate(-5 560 450)">
<rect x="478" y="360" width="190" height="180" rx="22" fill="#1F3A5F" opacity="0.13" transform="translate(8 10)"/>
<rect x="478" y="350" width="190" height="190" rx="22" fill="#FFFFFF" stroke="#C9DAEF" stroke-width="3"/>
<rect x="496" y="368" width="154" height="46" rx="10" fill="#CFE3FA"/>
<text x="640" y="402" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="27" font-weight="700" fill="#1F3A5F">65,00,000</text>
<g fill="#EAF3FD" stroke="#C9DAEF" stroke-width="2">
<rect x="496" y="428" width="30" height="28" rx="7"/><rect x="534" y="428" width="30" height="28" rx="7"/><rect x="572" y="428" width="30" height="28" rx="7"/><rect x="610" y="428" width="30" height="28" rx="7"/>
<rect x="496" y="464" width="30" height="28" rx="7"/><rect x="534" y="464" width="30" height="28" rx="7"/><rect x="572" y="464" width="30" height="28" rx="7"/><rect x="610" y="464" width="30" height="28" rx="7"/>
<rect x="496" y="500" width="30" height="28" rx="7"/><rect x="534" y="500" width="30" height="28" rx="7"/><rect x="572" y="500" width="30" height="28" rx="7"/>
</g>
<rect x="610" y="500" width="30" height="28" rx="7" fill="#E27D60"/>
</g>
</svg>
"""


def get_illustration_uri():
    """Built-in house, rising-arrow and calculator illustration (data URI)."""
    encoded = base64.b64encode(HERO_ILLUSTRATION_SVG.strip().encode("utf-8"))
    return "data:image/svg+xml;base64," + encoded.decode("utf-8")


try:
    data = load_data()
    model = load_model()

except FileNotFoundError as error:
    st.error(f"Project file not found: {error.filename}")
    st.info(
        "Check that your cleaned dataset is inside the data folder "
        "and your trained model is inside the models folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# HAMBURGER NAVIGATION
# ---------------------------------------------------------------------------

render("""
<div id="nav-closed"></div>
<div id="nav-panel" class="nav-panel">
<a class="nav-backdrop" href="#nav-closed" target="_self" aria-label="Close menu"></a>
<nav class="nav-drawer">
<div class="nav-title">Menu</div>
<a href="#home" target="_self">Home</a>
<a href="#price-prediction" target="_self">Price Prediction</a>
<a href="#how-it-works" target="_self">How It Works</a>
<a href="#about-estinest" target="_self">About EstiNest</a>
</nav>
</div>
<a class="hb hb-open" href="#nav-panel" target="_self" aria-label="Open menu"><span></span><span></span><span></span></a>
<a class="hb hb-close" href="#nav-closed" target="_self" aria-label="Close menu"><span></span><span></span><span></span></a>
""")


# ---------------------------------------------------------------------------
# 1. TOP BRANDING
# ---------------------------------------------------------------------------

render("""
<div id="home" class="site-header">
<div class="brand">EstiNest</div>
<div class="tagline">Smarter Home. Smarter Estimates.</div>
</div>
""")


# ---------------------------------------------------------------------------
# 2. HERO SECTION
# ---------------------------------------------------------------------------

hero_uri = get_hero_image_uri() or HERO_IMAGE_URL or get_illustration_uri()

hero_media = (
    f'<img src="{hero_uri}" referrerpolicy="no-referrer" '
    'alt="House with a rising price arrow, calculator and coins">'
)

render(f"""
<div class="hero">
<div>
<div class="hero-kicker">A smarter way to understand property value</div>
<div class="hero-heading">Your property.<br>Its estimated value.</div>
<div class="hero-text">
Discover an estimated house price using property details
and machine learning. Enter the details below and let
EstiNest do the calculation.
</div>
<div class="pill-row">
<span class="pill p1">Quick estimate</span>
<span class="pill p2">Price in Indian Rupees</span>
<span class="pill p3">Data-driven</span>
</div>
<div class="hero-actions">
<a class="btn-primary" href="#price-prediction" target="_self">Estimate Your Property</a>
<a class="btn-secondary" href="#how-it-works" target="_self">How It Works</a>
</div>
</div>
<div class="hero-media">{hero_media}</div>
</div>
""")


# ---------------------------------------------------------------------------
# 3. PRICE PREDICTION
# ---------------------------------------------------------------------------

render("""
<div id="price-prediction" class="section">
<div class="section-heading">Estimate Your Property</div>
<div class="section-description">
Tell us a little about the property to get your estimated price.
</div>
</div>
""")


with st.form("property_prediction_form"):

    left_col, right_col = st.columns(2, gap="large")

    input_values = {}

    location_options = sorted(
        data["Location"].dropna().astype(str).unique().tolist()
    )
    property_options = sorted(
        data["Property_Type"].dropna().astype(str).unique().tolist()
    )
    furnishing_options = sorted(
        data["Furnishing"].dropna().astype(str).unique().tolist()
    )

    with left_col:

        st.markdown(
            '<div class="form-group-title g1">Property Information</div>',
            unsafe_allow_html=True
        )

        input_values["Location"] = st.selectbox(
            "Location",
            location_options
        )

        input_values["Property_Type"] = st.selectbox(
            "Property Type",
            property_options
        )

        area_min, area_max = get_numeric_limits("Area_sqft")

        input_values["Area_sqft"] = st.number_input(
            "Area (sq. ft.)",
            min_value=0.0,
            max_value=max(area_max, 1.0),
            value=min(1200.0, max(area_max, 1.0)),
            step=50.0
        )

        bedroom_min, bedroom_max = get_numeric_limits("Bedrooms")

        input_values["Bedrooms"] = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=max(int(bedroom_max), 1),
            value=min(2, max(int(bedroom_max), 1)),
            step=1
        )

        bathroom_min, bathroom_max = get_numeric_limits("Bathrooms")

        input_values["Bathrooms"] = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=max(int(bathroom_max), 1),
            value=min(2, max(int(bathroom_max), 1)),
            step=1
        )

    with right_col:

        st.markdown(
            '<div class="form-group-title g2">Condition &amp; Amenities</div>',
            unsafe_allow_html=True
        )

        input_values["Furnishing"] = st.selectbox(
            "Furnishing",
            furnishing_options
        )

        quality_min, quality_max = get_numeric_limits(
            "Construction_Quality"
        )

        input_values["Construction_Quality"] = st.number_input(
            "Construction Quality",
            min_value=quality_min,
            max_value=quality_max,
            value=quality_min,
            step=1.0
        )

        age_min, age_max = get_numeric_limits("Age_Years")

        input_values["Age_Years"] = st.number_input(
            "Property Age (years)",
            min_value=0,
            max_value=max(int(age_max), 1),
            value=min(5, max(int(age_max), 1)),
            step=1
        )

        floor_min, floor_max = get_numeric_limits("Floor")

        input_values["Floor"] = st.number_input(
            "Floor",
            min_value=0,
            max_value=max(int(floor_max), 1),
            value=min(1, max(int(floor_max), 1)),
            step=1
        )

        parking_min, parking_max = get_numeric_limits("Parking_Spaces")

        input_values["Parking_Spaces"] = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=max(int(parking_max), 1),
            value=min(1, max(int(parking_max), 1)),
            step=1
        )

    submitted = st.form_submit_button(
        "Estimate Property Price",
        use_container_width=True
    )


if submitted:

    if input_values["Area_sqft"] <= 0:
        st.error("Please enter a property area greater than zero.")

    else:
        # Same column order as the original app.
        ordered_values = {name: input_values[name] for name in FEATURE_ORDER}
        input_df = pd.DataFrame([ordered_values])

        try:
            prediction = float(model.predict(input_df)[0])

            if prediction <= 0:
                st.error("The model returned an invalid price estimate.")

            else:
                st.session_state["last_prediction"] = prediction

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ---------------------------------------------------------------------------
# 4. ESTIMATED PRICE RESULT
# ---------------------------------------------------------------------------

if "last_prediction" in st.session_state:

    prediction = st.session_state["last_prediction"]

    render(f"""
    <div id="result" class="result-box">
    <div class="result-badge">Estimate ready</div>
    <div class="result-caption">Your Estimated Property Price</div>
    <div class="result-value">{format_inr(prediction)}</div>
    <div class="result-note">
    This is a machine learning estimate, not a formal property valuation.
    </div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 5. HOW IT WORKS (outlined step boxes)
# ---------------------------------------------------------------------------

render("""
<div id="how-it-works" class="section">
<div class="section-heading">How EstiNest Works</div>
<div class="section-description">
From property details to a price estimate in four simple steps.
</div>
</div>
<div class="card-grid four">
<div class="step s1">
<div class="step-number">1</div>
<div class="card-title">Enter details</div>
<div class="card-text">
Choose the location, property type, size, rooms, age, floor,
parking, furnishing and construction quality.
</div>
</div>
<div class="step s2">
<div class="step-number">2</div>
<div class="card-title">Prepare the data</div>
<div class="card-text">
Your inputs are arranged into the same format the model was
trained on.
</div>
</div>
<div class="step s3">
<div class="step-number">3</div>
<div class="card-title">Predict the price</div>
<div class="card-text">
The trained regression model estimates the property price from
the details you entered.
</div>
</div>
<div class="step s4">
<div class="step-number">4</div>
<div class="card-title">Read your estimate</div>
<div class="card-text">
The estimated price appears in Indian Rupees, right below the
form.
</div>
</div>
</div>
""")


# ---------------------------------------------------------------------------
# 6. ABOUT ESTINEST (soft filled boxes)
# ---------------------------------------------------------------------------

render("""
<div id="about-estinest" class="section">
<div class="section-heading">About EstiNest</div>
<div class="section-description">
A machine learning project focused on house price estimation.
</div>
</div>
<div class="card-grid">
<div class="about-card a1">
<div class="card-title">Our Purpose</div>
<div class="card-text">
To explore how property characteristics relate to house prices
and generate useful price estimates.
</div>
</div>
<div class="about-card a2">
<div class="card-title">Our Approach</div>
<div class="card-text">
We use data preprocessing, regression algorithms and model
evaluation to estimate property prices.
</div>
</div>
<div class="about-card a3">
<div class="card-title">Our Technology</div>
<div class="card-text">
Python, Pandas, Scikit-learn, Joblib and Streamlit are used to
build this application.
</div>
</div>
</div>
""")


# ---------------------------------------------------------------------------
# 7. FOOTER
# ---------------------------------------------------------------------------

render("""
<div class="footer">
<div class="footer-brand">EstiNest</div>
<div class="footer-tagline">Smarter Home. Smarter Estimates.</div>
<div class="footer-note">
Predictions are estimates based on the available dataset and should
not be treated as guaranteed market prices or a formal property valuation.
</div>
</div>
""")