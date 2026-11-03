"""
PropVista India - Smart Property Investment Intelligence
Run:  streamlit run app.py      (needs cleaned_housing_data.csv -> run clean_data.py first)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CREATOR_NAME = "Your Name"          # <<< PUT YOUR NAME HERE
APP_TITLE = "PropVista India"
TAGLINE = "Smart Property Investment Intelligence: from 2.5 lakh listings to one confident decision"

st.set_page_config(page_title=f"{APP_TITLE} | Real Estate Investment Advisor", page_icon="🏙️", layout="wide")

# ------------------------------------------------------------------ STYLE
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#12002b 0%,#2e0854 45%,#4a1a7a 100%);background-attachment:fixed;}
[data-testid="stHeader"]{background:transparent;}
html,body,.stMarkdown p,.stMarkdown li,label,.stSelectbox,.stMultiSelect{font-size:1.15rem !important;color:#f3e8ff;}
h1{font-size:3.4rem !important;font-weight:800 !important;color:#fff !important;letter-spacing:.5px;}
h2{font-size:2.5rem !important;font-weight:750 !important;color:#e9d5ff !important;border-bottom:3px solid #a855f7;padding-bottom:6px;margin-top:1.5rem !important;}
h3{font-size:1.9rem !important;color:#d8b4fe !important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b0019 0%,#2a0a4d 100%);border-right:2px solid #7c3aed;}
[data-testid="stSidebar"] .stButton>button{width:100%;text-align:left;font-size:1.15rem;font-weight:600;padding:.75rem 1rem;border-radius:14px;
  border:1px solid #5b21b6;background:rgba(124,58,237,.12);color:#e9d5ff;transition:all .25s;}
[data-testid="stSidebar"] .stButton>button:hover{transform:translateX(6px);border-color:#c084fc;background:rgba(168,85,247,.35);color:#fff;}
[data-testid="stSidebar"] button[kind="primary"],[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(90deg,#7c3aed,#c026d3) !important;color:#fff !important;border:none !important;box-shadow:0 0 18px rgba(192,38,211,.6);}
.brand{text-align:center;padding:10px 0 4px 0;}
.brand .b1{font-size:2rem;font-weight:800;color:#fff;} .brand .b2{font-size:.95rem;color:#c084fc;letter-spacing:2px;}
.hero{background:linear-gradient(120deg,#6d28d9,#a21caf,#4c1d95);background-size:200% 200%;animation:flow 8s ease infinite;
  padding:42px 38px;border-radius:24px;box-shadow:0 10px 40px rgba(124,58,237,.55);margin-bottom:22px;}
@keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero .t{font-size:3.6rem;font-weight:900;color:#fff;line-height:1.1;} .hero .s{font-size:1.5rem;color:#f5d0fe;margin-top:10px;}
.kpi{background:rgba(255,255,255,.07);border:1px solid #7c3aed;border-radius:18px;padding:18px 12px;text-align:center;backdrop-filter:blur(6px);}
.kpi .kv{font-size:2.3rem;font-weight:800;color:#e9d5ff;} .kpi .kl{font-size:1.1rem;color:#c084fc;font-weight:600;} .kpi .ks{font-size:.9rem;color:#a78bfa;}
.card{background:rgba(255,255,255,.06);border-left:6px solid #c026d3;border-radius:14px;padding:18px 22px;margin:10px 0;font-size:1.15rem;line-height:1.65;}
.card h4{font-size:1.5rem;color:#f0abfc;margin:0 0 6px 0;}
.insight{background:rgba(168,85,247,.16);border-left:6px solid #e879f9;border-radius:12px;padding:14px 20px;margin:6px 0 26px 0;font-size:1.15rem;line-height:1.6;}
.footer{text-align:center;margin-top:50px;padding:22px;border-top:2px solid #7c3aed;font-size:1.3rem;color:#d8b4fe;}
.footer b{color:#fff;font-size:1.6rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ DATA
@st.cache_data(show_spinner="Loading 250,000 listings...")
def load():
    d = pd.read_csv("cleaned_housing_data.csv")
    d["Good_Investment_%"] = d["Good_Investment"] * 100
    return d

df_all = load()

FEATS_CAT = ["City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility", "Parking_Space",
             "Security", "Facing", "Owner_Type", "Availability_Status"]
FEATS_NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property", "Floor_No", "Total_Floors",
             "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count"]


@st.cache_resource(show_spinner="Training investment models (one-time, ~20 sec)...")
def train_models():
    s = load().sample(40000, random_state=42)
    X = s[FEATS_CAT + FEATS_NUM]
    pre = lambda: ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATS_CAT), ("num", "passthrough", FEATS_NUM)])
    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, s["Good_Investment"], s["Future_Price_5Y_Lakhs"], test_size=0.2, random_state=42)
    clf = Pipeline([("p", pre()), ("m", RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=42))]).fit(Xtr, ytr)
    reg = Pipeline([("p", pre()), ("m", RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=42))]).fit(Xtr, rtr)
    pc, pr = clf.predict(Xte), reg.predict(Xte)
    metrics = {"Accuracy": accuracy_score(yte, pc), "F1": f1_score(yte, pc), "ROC-AUC": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]),
               "R2": r2_score(rte, pr), "RMSE": float(np.sqrt(mean_squared_error(rte, pr))), "MAE": mean_absolute_error(rte, pr)}
    names = clf.named_steps["p"].get_feature_names_out()
    imp = {}
    for n, v in zip(names, clf.named_steps["m"].feature_importances_):
        raw = n.split("__", 1)[1]
        base = next(c for c in FEATS_CAT + FEATS_NUM if raw == c or raw.startswith(c + "_"))
        imp[base] = imp.get(base, 0) + v
    imp = pd.Series(imp).sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    return clf, reg, metrics, imp

# ------------------------------------------------------------------ HELPERS
PURPLES = ["#7c3aed", "#a855f7", "#c084fc", "#d8b4fe", "#c026d3", "#6d28d9", "#e879f9"]


def style(fig, h=430):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.04)",
                      font=dict(size=15, color="#f3e8ff"), height=h, margin=dict(l=10, r=10, t=55, b=10),
                      title_font_size=21, colorway=PURPLES)
    return fig


def show(fig, h=430, key=None):
    st.plotly_chart(style(fig, h), use_container_width=True, key=key)


def insight(text):
    st.markdown(f'<div class="insight">💡 <b>Insight:</b> {text}</div>', unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="kv">{value}</div><div class="kl">{label}</div><div class="ks">{sub}</div></div>', unsafe_allow_html=True)


def gap(s):
    """Text describing highest / lowest group and the spread."""
    return (f"highest is <b>{s.idxmax()}</b> ({s.max():,.1f}), lowest is <b>{s.idxmin()}</b> ({s.min():,.1f}) - "
            f"a gap of only <b>{(s.max() / s.min() - 1):.1%}</b>")


def footer():
    st.markdown(f'<div class="footer">🏙️ {APP_TITLE} - Real Estate Investment Advisor<br>Created with 💜 by <b>{CREATOR_NAME}</b></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ SIDEBAR: NAVIGATION + FILTERS
PAGES = [("🏠", "Home"), ("🔎", "Key Findings"), ("📊", "Live Dashboards"), ("❓", "20 Questions Answered"),
         ("🤖", "Investment Advisor"), ("🔮", "Future Growth"), ("💼", "Recommendations & Conclusion")]
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.markdown(f'<div class="brand"><div class="b1">🏙️ {APP_TITLE}</div><div class="b2">INVEST SMARTER</div></div>', unsafe_allow_html=True)
st.sidebar.markdown("### 🧭 Navigate")
for i, (ic, name) in enumerate(PAGES):
    if st.sidebar.button(f"{ic}  {name}", key=f"nav{i}", type="primary" if st.session_state.page == name else "secondary"):
        st.session_state.page = name
        st.rerun()
page = st.session_state.page

st.sidebar.markdown("### 🎛️ Live Filters")
st.sidebar.caption("Filters apply to every dashboard, chart and insight. Leave empty = all.")
f_state = st.sidebar.multiselect("State", sorted(df_all.State.unique()))
cities = sorted(df_all[df_all.State.isin(f_state)].City.unique()) if f_state else sorted(df_all.City.unique())
f_city = st.sidebar.multiselect("City", cities)
f_type = st.sidebar.multiselect("Property type", sorted(df_all.Property_Type.unique()))
f_bhk = st.sidebar.multiselect("BHK", sorted(df_all.BHK.unique()))
f_avail = st.sidebar.multiselect("Availability", sorted(df_all.Availability_Status.unique()))
f_price = st.sidebar.slider("Price range (Lakhs)", 10, 500, (10, 500))
f_size = st.sidebar.slider("Size range (sq ft)", 500, 5000, (500, 5000), step=50)

d = df_all
if f_state: d = d[d.State.isin(f_state)]
if f_city: d = d[d.City.isin(f_city)]
if f_type: d = d[d.Property_Type.isin(f_type)]
if f_bhk: d = d[d.BHK.isin(f_bhk)]
if f_avail: d = d[d.Availability_Status.isin(f_avail)]
d = d[d.Price_in_Lakhs.between(*f_price) & d.Size_in_SqFt.between(*f_size)]
st.sidebar.success(f"✅ {len(d):,} of {len(df_all):,} listings selected")

if len(d) < 50:
    st.warning("Too few listings match these filters. Please widen the filters in the sidebar.")
    footer()
    st.stop()


def sample(n=5000):
    return d.sample(min(n, len(d)), random_state=7)


# ================================================================== PAGE: HOME
def page_home():
    st.markdown(f'<div class="hero"><div class="t">🏙️ {APP_TITLE}</div><div class="s">{TAGLINE}</div></div>', unsafe_allow_html=True)
    c = st.columns(5)
    kpi(c[0], "Listings analysed", f"{len(d):,}", f"{d.City.nunique()} cities | {d.State.nunique()} states")
    kpi(c[1], "Average price", f"₹{d.Price_in_Lakhs.mean():.0f} L", f"median ₹{d.Price_in_Lakhs.median():.0f} L")
    kpi(c[2], "Avg price / sq ft", f"₹{d.Price_per_SqFt_INR.mean():,.0f}", "cleaned & recomputed")
    kpi(c[3], "Good investments", f"{d.Good_Investment.mean():.1%}", "by domain scoring rule")
    kpi(c[4], "Avg 5-yr gain", f"{(d.Future_Price_5Y_Lakhs.sum() / d.Price_in_Lakhs.sum() - 1):.0%}", f"≈ {d['Annual_Growth_Rate_%'].mean():.1f}% per year")

    st.markdown("## 📖 Project Overview & Story")
    st.markdown(
        "Buying property is the biggest financial decision most families and investors ever make, yet most listings only show a price and a "
        "photo. **PropVista India** turns a dataset of **250,000 property listings across 20+ states and 40+ cities** into a decision engine. "
        "We cleaned the data, engineered investment-focused features (price per sq ft, infrastructure score, school density, amenity count), "
        "defined a transparent *Good Investment* rule, projected prices five years ahead, and trained machine-learning models that "
        "answer two questions for any property: **\"Is this a good investment?\"** and **\"What will it be worth in 5 years?\"**")
    st.markdown("## 🎯 Business Problem")
    card("The challenge", "Investors, buyers and real-estate platforms lack a fast, data-backed way to compare thousands of listings. Decisions are driven "
         "by gut feeling, brokers' claims or a single metric like price. This leads to over-paying, ignoring infrastructure quality and "
         "missing properties with strong appreciation potential.")
    st.markdown("## 🧪 Project Objectives & Analyses")
    a, b = st.columns(2)
    with a:
        card("1️⃣ Clean & prepare", "Remove duplicates, handle missing values, fix impossible floor numbers, recompute price per sq ft, flag outliers.")
        card("2️⃣ Explore (EDA)", "Answer 20 business questions on price, size, location, correlations, amenities and ownership.")
        card("3️⃣ Engineer features", "Price per sq ft, Infrastructure Score, School Density, Amenity Count, Age & Size categories.")
    with b:
        card("4️⃣ Classify", "Predict <b>Good Investment (Yes/No)</b> with a Random Forest and show confidence scores.")
        card("5️⃣ Forecast", "Estimate the <b>price after 5 years</b> with a regression model plus scenario analysis.")
        card("6️⃣ Deliver", "Interactive dashboards, advisor tool, recommendations - all filterable live from the sidebar.")
    st.info("👈 Use the **navigation panel** on the left to explore. Every chart is interactive: hover, zoom, click legends and use the live filters.")


# ================================================================== PAGE: KEY FINDINGS
def page_findings():
    st.markdown("# 🔎 Key Findings")
    st.caption("All numbers below recalculate live from your sidebar filters.")
    corr_sp = d.Size_in_SqFt.corr(d.Price_per_SqFt_INR)
    corr_sz = d.Size_in_SqFt.corr(d.Price_in_Lakhs)
    pps_state = d.groupby("State").Price_per_SqFt_INR.mean()
    good_city = d.groupby("City").Good_Investment.mean().sort_values(ascending=False)
    tr = d.groupby("Public_Transport_Accessibility").Good_Investment.mean()
    am = d.groupby("Amenity_Count").Price_per_SqFt_INR.mean()
    gr = d.groupby("City")["Annual_Growth_Rate_%"].mean().sort_values(ascending=False)
    findings = [
        ("💰 Prices are spread evenly - there is no 'cheap' or 'expensive' band",
         f"Prices run from ₹{d.Price_in_Lakhs.min():.0f} L to ₹{d.Price_in_Lakhs.max():.0f} L with a mean of ₹{d.Price_in_Lakhs.mean():.0f} L and median ₹{d.Price_in_Lakhs.median():.0f} L. "
         "The distribution is almost flat, so budgets of every size find options."),
        ("📐 Bigger does not mean pricier - small homes carry the premium per sq ft",
         f"Size and total price are practically uncorrelated (r = {corr_sz:.2f}), but size and price per sq ft are strongly negative (r = {corr_sp:.2f}). "
         "Compact homes cost the most per sq ft; large homes give the most space per rupee."),
        ("📍 Location gaps are small in this market",
         f"Average price per sq ft by state: {gap(pps_state)} (₹ per sq ft). Investors should pick on property fundamentals, not just the city name."),
        ("🛋️ Furnishing, parking, facing and amenity count barely move price",
         f"Price per sq ft changes only {(am.max() / am.min() - 1):.1%} from fewest to most amenities. These features are lifestyle choices, not value multipliers."),
        ("🚇 Transport access is the strongest lever for investment quality",
         "Share of Good Investments: " + ", ".join(f"<b>{k}</b> {v:.1%}" for k, v in tr.items()) + " transport access. Connectivity is a core part of our scoring rule."),
        ("🏆 Best cities for investment share",
         "Top 3: " + ", ".join(f"<b>{c}</b> ({v:.1%})" for c, v in good_city.head(3).items()) + f". Lowest: <b>{good_city.index[-1]}</b> ({good_city.iloc[-1]:.1%})."),
        ("🔮 Five-year outlook",
         f"Modelled average growth is {d['Annual_Growth_Rate_%'].mean():.1f}% a year: ₹{d.Price_in_Lakhs.mean():.0f} L today → ₹{d.Future_Price_5Y_Lakhs.mean():.0f} L in 5 years. "
         f"Fastest-growing city: <b>{gr.index[0]}</b> ({gr.iloc[0]:.2f}%/yr)."),
        ("🧹 Data quality",
         "No duplicates or missing values were found; however about 46% of raw rows had a floor number above the building height "
         "and were corrected. Price per sq ft was recomputed in rupees because the raw column was rounded to 2 decimals of a lakh."),
    ]
    for t, b in findings:
        card(t, b)
    footer()


# ================================================================== PAGE: DASHBOARDS
def page_dashboards():
    st.markdown("# 📊 Live Interactive Dashboards")
    st.caption("Change filters in the sidebar - every chart and insight updates instantly.")
    c = st.columns(5)
    kpi(c[0], "Listings", f"{len(d):,}")
    kpi(c[1], "Avg price", f"₹{d.Price_in_Lakhs.mean():.1f} L")
    kpi(c[2], "Avg ₹/sq ft", f"₹{d.Price_per_SqFt_INR.mean():,.0f}")
    kpi(c[3], "Good investment", f"{d.Good_Investment.mean():.1%}")
    kpi(c[4], "Avg infra score", f"{d.Infrastructure_Score.mean():.1f}/10")

    # 1 Location heatmap
    st.markdown("## 🗺️ 1. Location × Property-Type Heatmap")
    n = st.slider("Number of cities (ranked by avg price per sq ft)", 5, d.City.nunique(), min(15, d.City.nunique()), key="hm")
    top = d.groupby("City").Price_per_SqFt_INR.mean().nlargest(n).index
    hm = d[d.City.isin(top)].pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt_INR", aggfunc="mean").loc[top]
    show(px.imshow(hm, text_auto=",.0f", color_continuous_scale="Purples", aspect="auto", title="Average ₹ per sq ft"), 80 + 32 * n)
    insight(f"Costliest city-type combination is <b>{hm.stack().idxmax()[0]} - {hm.stack().idxmax()[1]}</b> (₹{hm.stack().max():,.0f}/sq ft); cheapest among the shown is "
            f"<b>{hm.stack().idxmin()[0]} - {hm.stack().idxmin()[1]}</b> (₹{hm.stack().min():,.0f}). Colours differ only slightly: the market is price-uniform, so look beyond city names.")

    # 2 Treemap
    st.markdown("## 🌳 2. Market Map - State → City (size = listings, colour = ₹/sq ft)")
    tm = d.groupby(["State", "City"]).agg(Listings=("ID", "count"), PPS=("Price_per_SqFt_INR", "mean")).reset_index()
    show(px.treemap(tm, path=["State", "City"], values="Listings", color="PPS", color_continuous_scale="Purples"), 520)
    st2 = d.groupby("State").Price_per_SqFt_INR.mean()
    insight(f"The market is broad: <b>{d.State.nunique()}</b> states and <b>{d.City.nunique()}</b> cities. State with highest ₹/sq ft: <b>{st2.idxmax()}</b> (₹{st2.max():,.0f}); lowest: <b>{st2.idxmin()}</b> (₹{st2.min():,.0f}).")

    # 3 Explorer
    st.markdown("## 🧩 3. Build Your Own Chart")
    e1, e2, e3, e4 = st.columns(4)
    grp = e1.selectbox("Group by", ["Property_Type", "BHK", "State", "City", "Furnished_Status", "Facing", "Owner_Type", "Availability_Status",
                                    "Public_Transport_Accessibility", "Age_Category", "Size_Category", "Floor_Position", "Amenity_Count"])
    met = e2.selectbox("Metric", ["Price_per_SqFt_INR", "Price_in_Lakhs", "Infrastructure_Score", "Annual_Growth_Rate_%", "Good_Investment_%", "Future_Price_5Y_Lakhs"])
    agg = e3.selectbox("Aggregation", ["mean", "median", "max", "min"])
    kind = e4.radio("Chart", ["Bar", "Line"], horizontal=True)
    zoom = st.checkbox("🔍 Zoom the y-axis (magnify small differences)")
    g = d.groupby(grp)[met].agg(agg).reset_index()
    g = g.sort_values(grp) if grp in ("BHK", "Amenity_Count") else g.sort_values(met, ascending=False).head(25)
    g[grp] = g[grp].astype(str)
    fig = px.bar(g, x=grp, y=met, color=met, color_continuous_scale="Purples", text_auto=".3s") if kind == "Bar" else px.line(g, x=grp, y=met, markers=True)
    if zoom:
        fig.update_yaxes(range=[g[met].min() * 0.98, g[met].max() * 1.01])
    show(fig, 470)
    s = g.set_index(grp)[met]
    insight(f"{agg.title()} <b>{met.replace('_', ' ')}</b> by <b>{grp.replace('_', ' ')}</b>: top group <b>{s.idxmax()}</b> ({s.max():,.2f}), bottom group <b>{s.idxmin()}</b> ({s.min():,.2f}); "
            f"relative spread {(s.max() / s.min() - 1):.1%}. {'Use the zoom box to see small gaps clearly.' if not zoom else 'Note: zoomed axis exaggerates visual differences.'}")

    # 4 Scatter
    st.markdown("## 🔵 4. Size vs Price (coloured by investment label)")
    sm = sample().assign(Label=lambda x: np.where(x.Good_Investment == 1, "Good Investment", "Not Good"))
    show(px.scatter(sm, x="Size_in_SqFt", y="Price_in_Lakhs", color="Label", opacity=.55, hover_data=["City", "Property_Type", "BHK"],
                    color_discrete_map={"Good Investment": "#e879f9", "Not Good": "#6d28d9"}), 470)
    insight(f"Correlation between size and price is <b>{d.Size_in_SqFt.corr(d.Price_in_Lakhs):.3f}</b> - a shapeless cloud. Good investments (pink) appear at every size and price, "
            f"showing that value comes from price-per-sq-ft and infrastructure, not from the size of the home.")

    # 5 Distribution
    st.markdown("## 📦 5. Price per Sq Ft Distribution by Property Type")
    show(px.box(sample(20000), x="Property_Type", y="Price_per_SqFt_INR", color="Property_Type"), 450)
    tp = d.groupby("Property_Type").Price_per_SqFt_INR.median()
    insight(f"Median ₹/sq ft - {', '.join(f'{k}: ₹{v:,.0f}' for k, v in tp.items())}. Long upper whiskers are small, high-priced units (outliers: {d.Outlier_Price_per_SqFt_INR.mean():.1%} of listings), "
            "which are the ones investors should negotiate hardest on.")

    # 6 Sunburst
    st.markdown("## ☀️ 6. Ownership & Availability Structure (click to drill down)")
    sb = d.groupby(["Owner_Type", "Availability_Status", "Property_Type"]).size().reset_index(name="Listings")
    show(px.sunburst(sb, path=["Owner_Type", "Availability_Status", "Property_Type"], values="Listings", color_discrete_sequence=PURPLES), 520)
    ot = d.Owner_Type.value_counts(normalize=True)
    insight(f"Supply is balanced: " + ", ".join(f"{k} {v:.1%}" for k, v in ot.items()) + f". {d.Availability_Status.value_counts(normalize=True).iloc[0]:.0%} of listings are "
            f"<b>{d.Availability_Status.value_counts().index[0].replace('_', ' ')}</b>. Buyers are never short of choices from any seller type.")

    # 7 Infrastructure
    st.markdown("## 🏗️ 7. Infrastructure Score → Investment Quality & Growth")
    ib = d.assign(Band=pd.cut(d.Infrastructure_Score, [-0.1, 3, 4, 5, 6, 10], labels=["0-3", "3-4", "4-5", "5-6", "6-10"]).astype(str))
    gi = ib.groupby("Band").agg(Good=("Good_Investment_%", "mean"), Growth=("Annual_Growth_Rate_%", "mean"), PPS=("Price_per_SqFt_INR", "mean")).reset_index().sort_values("Band")
    fig = go.Figure()
    fig.add_bar(x=gi.Band, y=gi.Good, name="% Good investments", marker_color="#a855f7")
    fig.add_scatter(x=gi.Band, y=gi.Growth, name="Avg growth %/yr", yaxis="y2", mode="lines+markers", line=dict(color="#f0abfc", width=4))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Growth %/yr"), yaxis_title="% Good", xaxis_title="Infrastructure score band", title="Better infrastructure = better odds")
    show(fig, 450)
    insight(f"Properties scoring 6+ are good investments <b>{gi.Good.iloc[-1]:.0f}%</b> of the time versus <b>{gi.Good.iloc[0]:.0f}%</b> in the 0-3 band, and grow "
            f"{gi.Growth.iloc[-1]:.1f}% vs {gi.Growth.iloc[0]:.1f}% a year. Meanwhile ₹/sq ft barely changes (₹{gi.PPS.min():,.0f}-₹{gi.PPS.max():,.0f}) - <b>infrastructure is under-priced by the market</b>, which is the opportunity.")

    # 8 Best cities
    st.markdown("## 🏅 8. Cities Ranked by Share of Good Investments")
    gc = d.groupby("City").agg(Good=("Good_Investment_%", "mean"), Growth=("Annual_Growth_Rate_%", "mean")).reset_index().sort_values("Good", ascending=False).head(15)
    show(px.bar(gc, x="City", y="Good", color="Growth", color_continuous_scale="Purples", text_auto=".1f", labels={"Good": "% Good investments"}), 450)
    insight(f"<b>{gc.City.iloc[0]}</b> leads with {gc.Good.iloc[0]:.1f}% good investments. The top 15 sit between {gc.Good.min():.1f}% and {gc.Good.max():.1f}%: differences are modest, so screen by property, not city.")

    # 9 3D
    st.markdown("## 🧊 9. 3-D Investment Space (rotate me!)")
    s3 = sample(3000).assign(Label=lambda x: np.where(x.Good_Investment == 1, "Good", "Not Good"))
    show(px.scatter_3d(s3, x="Size_in_SqFt", y="Price_in_Lakhs", z="Infrastructure_Score", color="Label", opacity=.7,
                       color_discrete_map={"Good": "#e879f9", "Not Good": "#5b21b6"}), 600)
    insight("Good investments (pink) cluster in the upper layers of the cube - high infrastructure score - and are scattered across size and price. "
            "Rotate and zoom to confirm that infrastructure, not size, separates winners from the rest.")

    # 10 Correlation
    st.markdown("## 🔗 10. Correlation Matrix")
    cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score", "Future_Price_5Y_Lakhs"]
    show(px.imshow(d[cols].corr(), text_auto=".2f", color_continuous_scale="PRGn", zmin=-1, zmax=1, aspect="auto"), 560)
    insight("Most raw features are independent (near 0). Strong links are mechanical: Price ↔ Future Price, Size ↔ ₹/sq ft (negative), Infrastructure ↔ schools/hospitals/amenities (by construction). "
            "Independent features mean a model needs several of them together to judge a property.")
    footer()


# ================================================================== PAGE: 20 QUESTIONS
def qblock(n, title, figs, text, h=400):
    with st.expander(f"Q{n}. {title}", expanded=True):
        figs = figs if isinstance(figs, list) else [figs]
        cols = st.columns(len(figs))
        for i, (c, f) in enumerate(zip(cols, figs)):
            with c:
                st.plotly_chart(style(f, h), use_container_width=True, key=f"q{n}_{i}")
        insight(text)


def page_questions():
    st.markdown("# ❓ The 20 Business Questions - Answered")
    st.caption("Each answer has an interactive chart and a plain-language insight, recalculated for your filters.")
    P = "Price_per_SqFt_INR"
    tabs = st.tabs(["💰 Price & Size (1-5)", "📍 Location (6-10)", "🔗 Relationships (11-15)", "🏢 Investment & Amenities (16-20)"])

    with tabs[0]:
        qblock(1, "What is the distribution of property prices?", px.histogram(d, x="Price_in_Lakhs", nbins=50, marginal="box", title="Price (Lakhs)"),
               f"Prices span ₹{d.Price_in_Lakhs.min():.0f}-{d.Price_in_Lakhs.max():.0f} L; mean ₹{d.Price_in_Lakhs.mean():.1f} L, median ₹{d.Price_in_Lakhs.median():.1f} L. "
               "Nearly flat distribution: no dominant price band, so every budget has plenty of options.")
        qblock(2, "What is the distribution of property sizes?", px.histogram(d, x="Size_in_SqFt", nbins=50, marginal="box", title="Size (sq ft)"),
               f"Sizes range {d.Size_in_SqFt.min()}-{d.Size_in_SqFt.max()} sq ft, median {d.Size_in_SqFt.median():,.0f}. Evenly spread - compact to luxury homes are all well represented.")
        t = d.groupby("Property_Type")[P].mean()
        qblock(3, "How does price per sq ft vary by property type?", px.box(sample(20000), x="Property_Type", y=P, color="Property_Type", title="₹ per sq ft"),
               f"Average ₹/sq ft: {gap(t)}. Property type is almost irrelevant to the per-sq-ft rate; choose by lifestyle and yield.")
        qblock(4, "Is there a relationship between property size and price?", px.scatter(sample(4000), x="Size_in_SqFt", y="Price_in_Lakhs", opacity=.4, color="BHK", title="Size vs Price"),
               f"Correlation = <b>{d.Size_in_SqFt.corr(d.Price_in_Lakhs):.3f}</b> → no relationship. As a result price per sq ft falls with size (r = {d.Size_in_SqFt.corr(d[P]):.2f}): "
               "big homes are the better value per rupee.")
        qblock(5, "Are there outliers in price per sq ft or property size?", [px.box(d, y=P, title="₹ per sq ft"), px.box(d, y="Size_in_SqFt", title="Size (sq ft)")],
               f"Price/sq ft has <b>{int(d.Outlier_Price_per_SqFt_INR.sum()):,}</b> IQR outliers ({d.Outlier_Price_per_SqFt_INR.mean():.1%}) - small units priced high; size has <b>{int(d.Outlier_Size_in_SqFt.sum())}</b>. "
               "Outliers were flagged (not deleted) so models can learn from them.")

    with tabs[1]:
        t = d.groupby("State")[P].mean().sort_values()
        qblock(6, "What is the average price per sq ft by state?", px.bar(t.reset_index(), x=P, y="State", orientation="h", color=P, color_continuous_scale="Purples"),
               f"{gap(t)} (₹/sq ft). Differences between states are small.", 650)
        t = d.groupby("City").Price_in_Lakhs.mean().sort_values()
        qblock(7, "What is the average property price by city?", px.bar(t.reset_index(), x="Price_in_Lakhs", y="City", orientation="h", color="Price_in_Lakhs", color_continuous_scale="Purples"),
               f"Costliest: <b>{t.index[-1]}</b> (₹{t.iloc[-1]:.1f} L); cheapest: <b>{t.index[0]}</b> (₹{t.iloc[0]:.1f} L). All cities sit in a narrow band.", 900)
        t = d.groupby("Locality").Age_of_Property.median()
        qblock(8, "What is the median age of properties by locality?", px.histogram(t.reset_index(), x="Age_of_Property", nbins=20, title="Median age across localities"),
               f"Across {len(t)} localities the median age ranges {t.min():.0f}-{t.max():.0f} years (typical {t.median():.0f}). Oldest locality: <b>{t.idxmax()}</b>; youngest: <b>{t.idxmin()}</b>.")
        ct = pd.crosstab(d.City, d.BHK, normalize="index") * 100
        qblock(9, "How is BHK distributed across cities?", px.imshow(ct, text_auto=".0f", color_continuous_scale="Purples", aspect="auto", title="% of listings by BHK"),
               f"Each BHK type holds roughly 20% share in every city (max deviation {abs(ct - 20).max().max():.1f} points). Product mix is uniform nationwide.", 900)
        top5 = d.groupby("Locality")[P].mean().nlargest(5)
        sub = d[d.Locality.isin(top5.index)].assign(Period=lambda x: pd.cut(x.Year_Built, [1989, 1995, 2000, 2005, 2010, 2015, 2023], labels=["1990-95", "1996-00", "2001-05", "2006-10", "2011-15", "2016-23"]))
        tr = sub.groupby(["Period", "Locality"], observed=True)[P].mean().reset_index()
        qblock(10, "What are the price trends for the top 5 most expensive localities?", px.line(tr, x="Period", y=P, color="Locality", markers=True),
               "Top 5 localities (avg ₹/sq ft): " + ", ".join(f"<b>{k}</b> ₹{v:,.0f}" for k, v in top5.items()) + ". Price by build period zig-zags without a steady rise, so location label alone does not guarantee appreciation.")

    with tabs[2]:
        cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", P, "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score"]
        cm = d[cols].corr()
        pr = cm.where(~np.eye(len(cm), dtype=bool)).stack().abs().sort_values(ascending=False)
        qblock(11, "How are numeric features correlated?", px.imshow(cm, text_auto=".2f", color_continuous_scale="PRGn", zmin=-1, zmax=1, aspect="auto"),
               f"Strongest pair: <b>{pr.index[0][0]} ↔ {pr.index[0][1]}</b> (|r| = {pr.iloc[0]:.2f}). Excluding derived pairs, raw features are essentially independent.", 560)
        t = d.groupby("Nearby_Schools")[P].mean()
        qblock(12, "How do nearby schools relate to price per sq ft?", px.line(t.reset_index(), x="Nearby_Schools", y=P, markers=True),
               f"Correlation {d.Nearby_Schools.corr(d[P]):.3f}; price varies just {(t.max() / t.min() - 1):.1%} across the school range. Schools are not priced in → a buying opportunity.")
        t = d.groupby("Nearby_Hospitals")[P].mean()
        qblock(13, "How do nearby hospitals relate to price per sq ft?", px.line(t.reset_index(), x="Nearby_Hospitals", y=P, markers=True),
               f"Correlation {d.Nearby_Hospitals.corr(d[P]):.3f}; variation {(t.max() / t.min() - 1):.1%}. Healthcare access is also not reflected in price.")
        t = d.groupby("Furnished_Status").Price_in_Lakhs.mean()
        qblock(14, "How does price vary by furnished status?", px.bar(t.reset_index(), x="Furnished_Status", y="Price_in_Lakhs", color="Furnished_Status", text_auto=".1f"),
               f"Average price: {gap(t)} (₹ L). Furnishing earns no premium - do not overpay for furniture.")
        t = d.groupby("Facing")[P].mean()
        qblock(15, "How does price per sq ft vary by facing direction?", px.bar(t.reset_index(), x="Facing", y=P, color="Facing", text_auto=",.0f"),
               f"{gap(t)} (₹/sq ft). Facing direction has virtually no effect on price here.")

    with tabs[3]:
        t = d.Owner_Type.value_counts()
        qblock(16, "How many properties belong to each owner type?", px.pie(t.reset_index(), names="Owner_Type", values="count", hole=.45),
               ", ".join(f"<b>{k}</b>: {v:,} ({v / len(d):.1%})" for k, v in t.items()) + ". A three-way even split: buyers can deal directly or via brokers/builders.")
        t = d.Availability_Status.value_counts()
        qblock(17, "How many properties are available under each availability status?", px.pie(t.reset_index(), names="Availability_Status", values="count", hole=.45),
               ", ".join(f"<b>{k.replace('_', ' ')}</b>: {v:,} ({v / len(d):.1%})" for k, v in t.items()) + ". Half the supply is under construction - lower entry risk for patient investors with higher modelled growth.")
        t = d.groupby("Parking_Space")[["Price_in_Lakhs", P]].mean().reset_index()
        qblock(18, "Does parking space affect property price?", [px.bar(t, x="Parking_Space", y="Price_in_Lakhs", color="Parking_Space", text_auto=".1f", title="Avg price (L)"),
                                                                    px.bar(t, x="Parking_Space", y=P, color="Parking_Space", text_auto=",.0f", title="Avg ₹/sq ft")],
               f"Price difference between parking and no-parking is {(t.Price_in_Lakhs.max() / t.Price_in_Lakhs.min() - 1):.1%} - parking is not priced as an extra.")
        am = pd.DataFrame([{"Amenity": a, "Premium %": (d.loc[d.Amenities.str.contains(a), P].mean() / d.loc[~d.Amenities.str.contains(a), P].mean() - 1) * 100}
                           for a in ["Gym", "Pool", "Garden", "Clubhouse", "Playground"]])
        bc = d.groupby("Amenity_Count")[P].mean().reset_index()
        qblock(19, "How do amenities affect price per sq ft?", [px.bar(am, x="Amenity", y="Premium %", color="Amenity", text_auto=".2f", title="Price/sq ft premium with amenity (%)"),
                                                                 px.line(bc, x="Amenity_Count", y=P, markers=True, title="By amenity count")],
               f"Premium of each amenity ranges {am['Premium %'].min():+.2f}% to {am['Premium %'].max():+.2f}%; correlation with amenity count is {d.Amenity_Count.corr(d[P]):.3f}. Amenities do not command a price premium - great value for buyers.")
        t = d.groupby("Public_Transport_Accessibility").agg(Good=("Good_Investment_%", "mean"), PPS=(P, "mean")).reset_index()
        qblock(20, "How does public transport accessibility relate to price per sq ft or investment potential?",
               [px.bar(t, x="Public_Transport_Accessibility", y="PPS", color="Public_Transport_Accessibility", text_auto=",.0f", title="Avg ₹/sq ft"),
                px.bar(t, x="Public_Transport_Accessibility", y="Good", color="Public_Transport_Accessibility", text_auto=".1f", title="% Good investments")],
               "Price per sq ft is nearly the same across access levels, but good-investment share is " + ", ".join(f"{r.Public_Transport_Accessibility} <b>{r.Good:.1f}%</b>" for r in t.itertuples())
               + ". Transport is under-priced yet drives investment potential - the clearest value signal in the data.")
    footer()


# ================================================================== PAGE: ADVISOR
def page_advisor():
    st.markdown("# 🤖 Investment Advisor")
    st.caption("Enter a property → get a verdict, a 5-year price forecast, model confidence and what drives the decision.")
    clf, reg, metrics, imp = train_models()
    city_state = df_all.drop_duplicates("City").set_index("City").State
    city_med = df_all.groupby("City").Price_per_SqFt_INR.median()
    infra_med, max_am = df_all.Infrastructure_Score.median(), df_all.Amenity_Count.max()

    with st.form("prop"):
        a, b, c = st.columns(3)
        city = a.selectbox("City", sorted(df_all.City.unique()))
        ptype = b.selectbox("Property type", sorted(df_all.Property_Type.unique()))
        bhk = c.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
        size = a.number_input("Size (sq ft)", 500, 5000, 1500, step=50)
        price = b.number_input("Asking price (Lakhs)", 10.0, 500.0, 150.0, step=5.0)
        year = c.number_input("Year built", 1990, 2025, 2015)
        furn = a.selectbox("Furnished status", sorted(df_all.Furnished_Status.unique()))
        floor = b.number_input("Floor number", 0, 30, 5)
        tfloor = c.number_input("Total floors", 1, 30, 12)
        sch = a.slider("Nearby schools", 0, 10, 6)
        hos = b.slider("Nearby hospitals", 0, 10, 6)
        trn = c.selectbox("Public transport accessibility", ["Low", "Medium", "High"], index=2)
        park = a.selectbox("Parking space", ["Yes", "No"])
        sec = b.selectbox("Security", ["Yes", "No"])
        face = c.selectbox("Facing", sorted(df_all.Facing.unique()))
        own = a.selectbox("Owner type", sorted(df_all.Owner_Type.unique()))
        av = b.selectbox("Availability", sorted(df_all.Availability_Status.unique()))
        am_sel = c.multiselect("Amenities", ["Gym", "Pool", "Garden", "Clubhouse", "Playground"], default=["Gym", "Garden"])
        go_btn = st.form_submit_button("🚀 Analyse this property", use_container_width=True)

    if not go_btn:
        st.info("Fill the form and press **Analyse this property**.")
    else:
        age = 2025 - year
        tfloor = max(tfloor, floor)
        namen = max(len(am_sel), 1)
        X = pd.DataFrame([{"City": city, "Property_Type": ptype, "Furnished_Status": furn, "Public_Transport_Accessibility": trn, "Parking_Space": park,
                           "Security": sec, "Facing": face, "Owner_Type": own, "Availability_Status": av, "BHK": bhk, "Size_in_SqFt": size,
                           "Price_in_Lakhs": price, "Age_of_Property": age, "Floor_No": floor, "Total_Floors": tfloor, "Nearby_Schools": sch,
                           "Nearby_Hospitals": hos, "Amenity_Count": namen}])
        prob = clf.predict_proba(X)[0, 1]
        Xt = reg.named_steps["p"].transform(X)
        tree_preds = np.array([t.predict(Xt)[0] for t in reg.named_steps["m"].estimators_])
        fut, lo, hi = tree_preds.mean(), np.percentile(tree_preds, 5), np.percentile(tree_preds, 95)

        ts = {"Low": 1, "Medium": 2, "High": 3}[trn]
        infra = (0.30 * sch / 10 + 0.30 * hos / 10 + 0.25 * (ts - 1) / 2 + 0.15 * namen / max_am) * 10
        pps = price * 1e5 / size
        checks = [(f"Price/sq ft ₹{pps:,.0f} ≤ {city} median ₹{city_med[city]:,.0f}", pps <= city_med[city]),
                  (f"Infrastructure score {infra:.1f} ≥ market median {infra_med:.1f}", infra >= infra_med),
                  ("Medium / High public transport", ts >= 2), ("Ready to move", av == "Ready_to_Move"), ("Age ≤ 15 years", age <= 15)]
        score = sum(ok for _, ok in checks)
        growth = (0.05 + 0.03 * infra / 10 + 0.015 * (ts - 1) / 2 + 0.01 * (age <= 10) + 0.01 * (av == "Under_Construction"))

        verdict = prob >= 0.5
        st.markdown(f"## {'✅ GOOD INVESTMENT' if verdict else '⚠️ NOT A STRONG INVESTMENT'}")
        g1, g2, g3 = st.columns([1.2, 1, 1])
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number={"suffix": "%"}, title={"text": "Model confidence: Good Investment"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e879f9"},
                                            "steps": [{"range": [0, 50], "color": "#3b0a6b"}, {"range": [50, 100], "color": "#5b21b6"}]}))
        with g1:
            show(fig, 300)
        kpi(g2, "Estimated price after 5 years", f"₹{fut:.0f} L", f"90% range ₹{lo:.0f}-{hi:.0f} L")
        kpi(g3, "Expected gain", f"₹{fut - price:.0f} L", f"{(fut / price - 1):.0%} total | {growth * 100:.1f}% per year")
        insight(f"The model is <b>{prob:.0%}</b> confident this is a good investment. At an asking price of ₹{price:.0f} L the estimated value in 2030 is <b>₹{fut:.0f} L</b> "
                f"(range ₹{lo:.0f}-{hi:.0f} L from the spread of 80 decision trees). The rule-based score is <b>{score}/5</b> (≥4 = Good).")
        l, r = st.columns(2)
        with l:
            st.markdown("### 📋 Rule-based checklist")
            for t, ok in checks:
                st.markdown(f"{'✅' if ok else '❌'} {t}")
        with r:
            yrs = list(range(2025, 2031))
            fig = go.Figure(go.Scatter(x=yrs, y=[price * (1 + growth) ** i for i in range(6)], mode="lines+markers", line=dict(color="#e879f9", width=4), name="Projected value"))
            fig.update_layout(title="Projected value path (₹ Lakhs)")
            show(fig, 320)

    st.markdown("## 📈 Model Performance (hold-out test set)")
    m = st.columns(6)
    for col, (k, v) in zip(m, metrics.items()):
        kpi(col, k, f"{v:.3f}" if k in ("Accuracy", "F1", "ROC-AUC", "R2") else f"{v:.2f}", "classification" if k in ("Accuracy", "F1", "ROC-AUC") else "regression")
    show(px.bar(imp.tail(12), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Purples", title="Feature importance (Good-Investment model)"), 480)
    insight(f"Top drivers: <b>{imp.Feature.iloc[-1]}</b>, <b>{imp.Feature.iloc[-2]}</b> and <b>{imp.Feature.iloc[-3]}</b>. Because the label is built from domain rules the classifier scores high; "
            "on real transaction history the same pipeline can be retrained with true appreciation data.")

    st.markdown("## 🏆 Top 10 Recommended Listings (within your sidebar filters)")
    cols = ["ID", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Infrastructure_Score", "Investment_Score", "Future_Price_5Y_Lakhs", "Expected_Gain_Lakhs"]
    st.dataframe(d.sort_values(["Investment_Score", "Expected_Gain_Lakhs"], ascending=False).head(10)[cols], use_container_width=True, hide_index=True)
    insight("Listings are ranked by investment score first, then by expected 5-year gain. Narrow the sidebar filters (city, BHK, budget) to shortlist your own top 10.")
    footer()


# ================================================================== PAGE: FUTURE
def page_future():
    st.markdown("# 🔮 Future Growth & Price Forecast")
    st.caption("5-year projections from the growth model (5% base + infrastructure, transport, newness and construction premiums).")
    scen = st.radio("Scenario", ["Conservative (-2 pts)", "Base case", "Optimistic (+2 pts)"], index=1, horizontal=True)
    adj = {"Conservative (-2 pts)": -2, "Base case": 0, "Optimistic (+2 pts)": 2}[scen]
    cg = d.groupby("City").agg(Now=("Price_in_Lakhs", "mean"), G=("Annual_Growth_Rate_%", "mean"), Good=("Good_Investment_%", "mean")).reset_index()
    cg["G"] = cg.G + adj
    cg["In5y"] = cg.Now * (1 + cg.G / 100) ** 5
    cg["Gain %"] = (cg.In5y / cg.Now - 1) * 100
    long = pd.concat([cg.assign(Year=2025 + i, Value=cg.Now * (1 + cg.G / 100) ** i)[["City", "Year", "Value"]] for i in range(6)])

    c = st.columns(4)
    kpi(c[0], "Avg price today", f"₹{d.Price_in_Lakhs.mean():.0f} L")
    kpi(c[1], "Avg price in 2030", f"₹{(d.Price_in_Lakhs * (1 + (d['Annual_Growth_Rate_%'] + adj) / 100) ** 5).mean():.0f} L", scen.split(" (")[0])
    kpi(c[2], "Avg annual growth", f"{d['Annual_Growth_Rate_%'].mean() + adj:.2f}%")
    kpi(c[3], "Best city gain", f"{cg['Gain %'].max():.0f}%", cg.loc[cg["Gain %"].idxmax(), "City"])

    st.markdown("## 📈 Price Trajectory by City")
    default = list(cg.nlargest(5, "G").City)
    sel = st.multiselect("Choose cities to compare", sorted(cg.City), default=default)
    if sel:
        show(px.line(long[long.City.isin(sel)], x="Year", y="Value", color="City", markers=True, labels={"Value": "Avg price (₹ Lakhs)"}), 450)
        best = cg[cg.City.isin(sel)].sort_values("Gain %", ascending=False).iloc[0]
        insight(f"Under the <b>{scen.split(' (')[0].lower()}</b> scenario, <b>{best.City}</b> grows fastest among your selection: ₹{best.Now:.0f} L → ₹{best.In5y:.0f} L (+{best['Gain %']:.0f}%) by 2030.")

    st.markdown("## 🎬 Animated Price Race (press ▶ Play)")
    top10 = cg.nlargest(10, "In5y").City
    an = long[long.City.isin(top10)]
    show(px.bar(an, x="City", y="Value", animation_frame="Year", color="City", range_y=[0, an.Value.max() * 1.1], labels={"Value": "Avg price (₹ Lakhs)"}), 480)
    insight("Watch prices compound year by year. Compounding at 7-9% a year adds roughly 40-55% in five years - time in the market matters more than the exact city.")

    st.markdown("## 🏁 Top 10 Cities by Expected 5-Year Gain")
    t10 = cg.nlargest(10, "Gain %")
    show(px.bar(t10, x="City", y="Gain %", color="Good", color_continuous_scale="Purples", text_auto=".1f", labels={"Good": "% good inv."}), 440)
    insight(f"Leaders: " + ", ".join(f"<b>{r['City']}</b> ({r['Gain %']:.0f}%)" for _, r in t10.head(3).iterrows()) + ". Gaps are small (all cities within a few points), so combine growth with entry price and infrastructure.")

    st.markdown("## 🏗️ Growth by Availability & Transport")
    gg = d.assign(G=d["Annual_Growth_Rate_%"] + adj).groupby(["Availability_Status", "Public_Transport_Accessibility"]).G.mean().reset_index()
    show(px.bar(gg, x="Public_Transport_Accessibility", y="G", color="Availability_Status", barmode="group", text_auto=".2f", labels={"G": "Annual growth %"}), 430)
    insight(f"Highest growth: <b>{gg.loc[gg.G.idxmax(), 'Availability_Status'].replace('_', ' ')}</b> properties with <b>{gg.loc[gg.G.idxmax(), 'Public_Transport_Accessibility']}</b> transport "
            f"({gg.G.max():.2f}%/yr) vs lowest {gg.G.min():.2f}%/yr. Under-construction, well-connected projects offer the strongest appreciation.")
    st.warning("Forecasts are model-based estimates from a single-year snapshot, not guarantees. Validate with local market data before investing.")
    footer()


# ================================================================== PAGE: RECOMMENDATIONS
def page_reco():
    st.markdown("# 💼 Business Recommendations")
    recs = [("1. Buy infrastructure, not postcodes", "Transport, schools and hospitals are not priced into ₹/sq ft yet they decide investment quality. Shortlist properties with Medium/High transport and an infrastructure score above the market median."),
            ("2. Prefer larger homes for value", "Price per sq ft drops as size rises while total price does not. Investors get more area per rupee in larger units."),
            ("3. Never pay extra for furnishing, parking, facing or amenity count", "None of these carry a measurable premium in the data. Negotiate them as free extras instead of paying for them."),
            ("4. Look at under-construction, well-connected projects", "They show the highest modelled growth. Pair with a trusted builder and a possession-date check."),
            ("5. Target new and recent buildings", "Age up to 15 years scores better on the investment rule and has lower maintenance risk."),
            ("6. Real-estate platforms: add an 'Investment Score' badge", "Show Good Investment / 5-year price estimate on each listing to build trust and increase conversion."),
            ("7. Developers: build where infrastructure exists", "Projects near transport and social infrastructure achieve the highest investor demand and appreciation."),
            ("8. Diversify across cities", "City-level growth gaps are small, so spreading across 2-3 cities reduces risk without sacrificing return.")]
    for t, b in recs:
        card(t, b)
    st.markdown("## 🚀 Future Scope")
    card("Next steps", "Add true multi-year transaction history to learn real appreciation; include crime-rate, rental-yield and metro-expansion data; "
         "deploy on Streamlit Cloud with MLflow tracking; add a rental-income and EMI calculator; build a city-level geographic map with coordinates.")
    st.markdown("## 🏁 Conclusion")
    card("Summary", f"Across <b>{len(d):,}</b> listings, the Indian housing market in this dataset is remarkably uniform on price: location, furnishing, facing, parking and amenities barely change "
         f"₹/sq ft. What does differentiate properties is <b>infrastructure and connectivity</b>, and the market is not charging for it. PropVista turns that gap into action: a transparent "
         f"Good-Investment rule ({d.Good_Investment.mean():.1%} of listings qualify), a classifier and a 5-year price forecaster with confidence scores, and interactive dashboards "
         f"that let any investor, builder or platform decide in minutes. Average prices are projected to rise from ₹{d.Price_in_Lakhs.mean():.0f} L to ₹{d.Future_Price_5Y_Lakhs.mean():.0f} L in five years.")
    st.balloons() if st.button("🎉 Celebrate the project") else None
    footer()


{"Home": page_home, "Key Findings": page_findings, "Live Dashboards": page_dashboards, "20 Questions Answered": page_questions,
 "Investment Advisor": page_advisor, "Future Growth": page_future, "Recommendations & Conclusion": page_reco}[page]()
if page == "Home":
    footer()
