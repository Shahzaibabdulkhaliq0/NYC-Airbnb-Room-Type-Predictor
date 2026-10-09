from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model_Pipeline_Final.pkl"

# NYC subway-inspired colours for the room types (any other class gets a fallback colour).
CLASS_COLOURS = {"Entire home/apt": "#ffc629", "Private room": "#ee352e", "Shared room": "#00a651"}
CLASS_ICONS = {"Entire home/apt": "🏠", "Private room": "🛏️", "Shared room": "👥"}
FALLBACK_COLOURS = ["#ffc629", "#ee352e", "#00a651", "#3b82f6"]

st.set_page_config(
    page_title="StayWise NYC | Room Type Classifier",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# "NYC night" theme: near-black with taxi yellow and subway-line colours, flat square cards, bold uppercase headings.
# Three cards side by side, sized to fit one screen with no scrolling.
st.markdown(
    """
    <style>
    html{font-size:15px}
    .stApp{background:radial-gradient(circle at 85% -10%,#2a2008 0,#101010 38%,#080808 100%);color:#f3f0e8;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}
    [data-testid="stHeader"],footer,#MainMenu,[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
    .block-container,[data-testid="stMainBlockContainer"]{padding:.7rem 1.4rem 0 1.4rem!important;max-width:100%!important}
    [data-testid="stVerticalBlock"]{gap:.5rem}
    [data-testid="stHorizontalBlock"]{gap:.9rem}
    h1,h2,h3,h4,p,label,span,li{color:#f3f0e8}
    [data-testid="stWidgetLabel"] p{font-size:.72rem;color:#a39d8e;margin-bottom:0;text-transform:uppercase;letter-spacing:.8px;font-weight:700}

    /* Dark inputs: light text on dark boxes so everything is visible */
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#1a1a1a!important;border-color:#383838;border-radius:5px;min-height:2.3rem}
    div[data-baseweb="select"] *,div[data-baseweb="input"] *{color:#f6f3ea!important}
    div[data-baseweb="select"] svg{fill:#c9c3b3!important}
    input{color:#f6f3ea!important;-webkit-text-fill-color:#f6f3ea!important;background:transparent!important}
    [data-testid="stNumberInput"] button{background:#262626!important;color:#f6f3ea!important;border-color:#383838!important}
    [data-testid="stNumberInput"] button svg{fill:#f6f3ea!important}
    [data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"] ul{background:#1a1a1a!important}
    [data-baseweb="popover"] li,[data-baseweb="popover"] li *{color:#f6f3ea!important;background:transparent}
    [data-baseweb="popover"] li:hover,[data-baseweb="popover"] li[aria-selected="true"]{background:#33290a!important}

    /* Flat square cards with a yellow top edge */
    div[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid #2c2c2c;border-top:3px solid #ffc629;border-radius:6px;background:#141414;padding:.8rem .95rem;box-shadow:0 10px 30px rgba(0,0,0,.5)}
    .topbar{display:flex;justify-content:space-between;align-items:center;padding-bottom:.45rem;margin-bottom:.15rem;border-bottom:1px solid #2c2c2c}
    .brand{display:flex;align-items:center;gap:.6rem;font-size:1.2rem;font-weight:900;letter-spacing:2px;text-transform:uppercase}
    .brand small{font-size:.68rem;font-weight:600;color:#a39d8e;letter-spacing:1.5px;margin-left:.4rem}
    .bullet{width:30px;height:30px;border-radius:50%;background:#ffc629;color:#111;display:inline-flex;align-items:center;justify-content:center;font-weight:900;font-size:.95rem;letter-spacing:0}
    .pill{border:1px solid #383838;border-radius:4px;padding:.2rem .7rem;font-size:.68rem;color:#ffc629;letter-spacing:1px;text-transform:uppercase;font-weight:700}
    .sec{font-size:.9rem;font-weight:900;margin:0 0 .1rem 0;letter-spacing:1.2px;text-transform:uppercase}
    .cap{font-size:.74rem;color:#a39d8e;margin:0 0 .3rem 0}
    .tag{display:inline-block;padding:.18rem .55rem;border-radius:3px;background:rgba(255,198,41,.12);color:#ffc629;font-weight:800;font-size:.64rem;letter-spacing:1.2px;text-transform:uppercase}
    .center{display:flex;flex-direction:column;align-items:center;text-align:center}
    .bullet-big{width:76px;height:76px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2.1rem;margin:.6rem 0 .45rem;color:#111;font-weight:900}
    .rname{font-size:1.3rem;font-weight:900;line-height:1.15;text-transform:uppercase;letter-spacing:.8px}
    .pct{font-size:2.7rem;font-weight:900;line-height:1.05;margin-top:.3rem;letter-spacing:-1px}
    .sub{font-size:.66rem;color:#a39d8e;letter-spacing:1.5px;text-transform:uppercase;margin:.1rem 0 .45rem}
    .note{font-size:.74rem;color:#a39d8e;margin-top:.3rem}
    .bar{margin:.55rem 0}
    .bl{display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:.22rem;text-transform:uppercase;letter-spacing:.6px}
    .bt{height:10px;border-radius:2px;background:#262626;overflow:hidden;width:100%}
    .bf{height:100%;border-radius:2px}
    .chips{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-top:.7rem}
    .chip{border:1px solid #2c2c2c;border-left:3px solid #ffc629;border-radius:3px;background:#1a1a1a;padding:.4rem .6rem}
    .chip-l{font-size:.6rem;color:#a39d8e;letter-spacing:1.2px;text-transform:uppercase;font-weight:800}
    .chip-v{font-size:.85rem;font-weight:700;margin-top:.1rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .disc{font-size:.68rem;color:#7d7869;margin-top:.7rem}
    div.stButton>button,div[data-testid="stFormSubmitButton"] button{border:0;border-radius:5px;min-height:2.6rem;color:#111!important;font-weight:900;letter-spacing:1.5px;text-transform:uppercase;background:#ffc629;box-shadow:0 6px 18px rgba(255,198,41,.25);transition:transform .16s,box-shadow .16s}
    div.stButton>button *,div[data-testid="stFormSubmitButton"] button *{color:#111!important}
    div.stButton>button:hover,div[data-testid="stFormSubmitButton"] button:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(255,198,41,.4);background:#ffd25a}

    /* Animations (only the result area, so they play when you press Predict) */
    @keyframes grow{from{width:0}}
    @keyframes popIn{0%{opacity:0;transform:scale(.5)}70%{transform:scale(1.12)}100%{opacity:1;transform:scale(1)}}
    @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
    .bf{animation:grow 1s ease-out}
    .bullet-big{animation:popIn .7s ease both}
    .pct{animation:fadeUp .7s .25s ease both}
    .rname{animation:fadeUp .7s .15s ease both}
    .chip{animation:fadeUp .6s .5s ease both}
    @media (prefers-reduced-motion: reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified_at: float):
    # Loads the saved full preprocessing + classifier pipeline. No training happens in this app.
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=UserWarning)
        return joblib.load(path)


st.markdown(
    '<div class="topbar"><div class="brand"><span class="bullet">S</span>StayWise NYC<small>Room type classifier</small></div>'
    '<div class="pill">● Saved pipeline ready</div></div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Saved model file not found. Keep `Model_Pipeline_Final.pkl` in the same folder as `app.py`.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    required_features = list(model.feature_names_in_)
    preprocessor = model.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]
    category_options = {
        name: [str(item) for item in categories]
        for name, categories in zip(["neighbourhood_group", "neighbourhood"], cat_encoder.categories_)
    }
    model_classes = [str(value) for value in model.classes_]
except Exception as exc:
    st.error("The saved model could not be loaded. Check that the package versions match the environment used to save it.")
    st.exception(exc)
    st.stop()

# Keep the prediction when Streamlit reruns after the form is submitted.
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

input_card, result_card, detail_card = st.columns([1.25, 1, 1], vertical_alignment="top")

# ---------- Card 1: inputs ----------
with input_card:
    with st.container(border=True, key="input_panel"):
        st.markdown('<div class="sec">Listing details</div><div class="cap">Enter the listing, then classify the room type.</div>', unsafe_allow_html=True)
        with st.form("listing_prediction_form", border=False):
            a1, a2 = st.columns(2)
            neighbourhood_group = a1.selectbox("Borough", category_options["neighbourhood_group"], index=min(2, len(category_options["neighbourhood_group"]) - 1))
            neighbourhood = a2.selectbox("Neighbourhood", category_options["neighbourhood"], index=(category_options["neighbourhood"].index("Chelsea") if "Chelsea" in category_options["neighbourhood"] else 0))

            b1, b2 = st.columns(2)
            price = b1.number_input("Nightly price ($)", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)
            minimum_nights = b2.number_input("Minimum nights", min_value=1, max_value=3650, value=3, step=1)

            c1, c2 = st.columns(2)
            latitude = c1.number_input("Latitude", min_value=40.0, max_value=41.0, value=40.745, step=0.001, format="%.6f")
            longitude = c2.number_input("Longitude", min_value=-75.0, max_value=-73.0, value=-73.985, step=0.001, format="%.6f")

            d1, d2 = st.columns(2)
            number_of_reviews = d1.number_input("Total reviews", min_value=0, max_value=10000, value=10, step=1)
            reviews_per_month = d2.number_input("Reviews per month", min_value=0.0, max_value=1000.0, value=1.0, step=0.5)

            e1, e2 = st.columns(2)
            availability_365 = e1.number_input("Availability / year", min_value=0, max_value=365, value=180, step=1)
            calculated_host_listings_count = e2.number_input("Host listings count", min_value=1, max_value=10000, value=1, step=1)

            submitted = st.form_submit_button("Predict room type →", use_container_width=True)

    if submitted:
        input_values = {
            "neighbourhood_group": neighbourhood_group,
            "neighbourhood": neighbourhood,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "price": float(price),
            "minimum_nights": int(minimum_nights),
            "number_of_reviews": int(number_of_reviews),
            "reviews_per_month": float(reviews_per_month),
            "calculated_host_listings_count": int(calculated_host_listings_count),
            "availability_365": int(availability_365),
        }
        try:
            input_frame = pd.DataFrame([{feature: input_values[feature] for feature in required_features}], columns=required_features)
            prediction = str(model.predict(input_frame)[0])
            probabilities = model.predict_proba(input_frame)[0] if hasattr(model, "predict_proba") else None
            probability_map = ({name: float(prob) for name, prob in zip(model_classes, probabilities)} if probabilities is not None else {})
            st.session_state.last_prediction = {
                "prediction": prediction,
                "probabilities": probability_map,
                "inputs": input_values,
            }
        except Exception as exc:
            st.error("Prediction failed. Check the saved model and package versions.")
            st.exception(exc)

result = st.session_state.last_prediction

# ---------- Card 2: main result ----------
with result_card:
    with st.container(border=True, key="result_panel"):
        st.markdown('<div class="sec">Prediction</div><div class="cap">Most likely room type for this listing.</div>', unsafe_allow_html=True)
        if result:
            winner = result["prediction"]
            probs = result["probabilities"]
            colour = CLASS_COLOURS.get(winner, FALLBACK_COLOURS[0])
            icon = CLASS_ICONS.get(winner, "🏙️")
            confidence = probs.get(winner) if probs else None
            if confidence is not None:
                st.markdown(
                    f'<div class="center"><span class="tag">Most likely class</span>'
                    f'<div class="bullet-big" style="background:{colour};box-shadow:0 0 24px {colour}66">{icon}</div>'
                    f'<div class="rname">{winner}</div>'
                    f'<div class="pct" style="color:{colour}">{confidence:.0%}</div><div class="sub">confidence</div>'
                    f'<div class="bt"><div class="bf" style="width:{confidence*100:.1f}%;background:{colour}"></div></div>'
                    f'<div class="note">Model probability for the predicted class</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="center"><span class="tag">Most likely class</span>'
                    f'<div class="bullet-big" style="background:{colour}">{icon}</div><div class="rname">{winner}</div>'
                    f'<div class="note">The model did not provide probability scores.</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="center"><span class="tag">Ready for input</span>'
                '<div class="bullet-big" style="background:#2a2a2a;color:#777;animation:none">?</div>'
                '<div class="rname">Awaiting input</div><div class="note">Fill in the listing and press “Predict room type”.</div></div>',
                unsafe_allow_html=True,
            )

# ---------- Card 3: probabilities and listing summary ----------
with detail_card:
    with st.container(border=True, key="detail_panel"):
        st.markdown('<div class="sec">Class probabilities</div><div class="cap">How confident the model is in each room type.</div>', unsafe_allow_html=True)
        bars = ""
        if result and result["probabilities"]:
            ordered = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
        else:
            ordered = [(name, 0.0) for name in model_classes]
        position = 0
        for label, value in ordered:
            bar_colour = CLASS_COLOURS.get(label, FALLBACK_COLOURS[position % len(FALLBACK_COLOURS)])
            bars += (
                f'<div class="bar"><div class="bl"><span>{label}</span><b>{value:.1%}</b></div>'
                f'<div class="bt"><div class="bf" style="width:{value*100:.1f}%;background:{bar_colour}"></div></div></div>'
            )
            position += 1
        st.markdown(bars, unsafe_allow_html=True)

        if result:
            data = result["inputs"]
            st.markdown(
                '<div class="chips">'
                f'<div class="chip"><div class="chip-l">Borough</div><div class="chip-v">{data["neighbourhood_group"]}</div></div>'
                f'<div class="chip"><div class="chip-l">Neighbourhood</div><div class="chip-v">{data["neighbourhood"]}</div></div>'
                f'<div class="chip"><div class="chip-l">Price / night</div><div class="chip-v">${data["price"]:,.0f}</div></div>'
                f'<div class="chip"><div class="chip-l">Min nights</div><div class="chip-v">{data["minimum_nights"]}</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="chips"><div class="chip"><div class="chip-l">Output classes</div><div class="chip-v">' + " · ".join(model_classes) + '</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="disc">Educational demo. Probabilities are model estimates, not a guarantee of the actual room type.</div>', unsafe_allow_html=True)
