import re
import time
from html import escape as esc
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Persona Builder", page_icon="📊", layout="wide")

PAL = ["#4F46E5", "#F5B800", "#10B981", "#F472B6", "#22D3EE", "#F97316", "#8B5CF6", "#64748B"]
px.defaults.color_discrete_sequence = PAL

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F5F6FA; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E7E9F0; }
.hero { background: linear-gradient(120deg, #1E1B4B 0%, #4F46E5 100%); border-radius: 20px;
        padding: 28px 32px; margin-bottom: 14px; }
.hero-t { color: #FFFFFF; font-size: 28px; font-weight: 700; letter-spacing: -0.3px; }
.hero-s { color: #E0E7FF; font-size: 15px; margin-top: 6px; }
.hero-bar { width: 56px; height: 4px; background: #F5B800; border-radius: 4px; margin-bottom: 12px; }
.chip { display: inline-block; background: #FFFFFF; border: 1px solid #E7E9F0; color: #334155;
        border-radius: 999px; padding: 4px 12px; font-size: 12.5px; margin: 0 8px 8px 0; }
.kpi { background: #FFFFFF; border-radius: 16px; padding: 16px 18px; border: 1px solid #E7E9F0;
       box-shadow: 0 2px 10px rgba(15,23,42,0.05); }
.kpi-l { color: #64748B; font-size: 12.5px; font-weight: 500; text-transform: uppercase; letter-spacing: .4px; }
.kpi-v { color: #0F172A; font-size: 30px; font-weight: 700; margin-top: 2px; }
.kpi-s { color: #64748B; font-size: 12.5px; }
.ins { background: #FFFFFF; border-radius: 16px; padding: 18px; border: 1px solid #E7E9F0;
       box-shadow: 0 2px 10px rgba(15,23,42,0.05); height: 100%; }
.ins-b { color: #4F46E5; font-size: 32px; font-weight: 700; }
.ins-t { color: #334155; font-size: 14px; margin-top: 4px; line-height: 1.45; }
.seg { background: #FFFFFF; border-radius: 16px; padding: 16px 18px; border: 1px solid #E7E9F0;
       box-shadow: 0 2px 10px rgba(15,23,42,0.05); margin-bottom: 14px; min-height: 190px; }
.seg-n { color: #0F172A; font-size: 15px; font-weight: 600; }
.seg-m { color: #64748B; font-size: 12.5px; margin: 4px 0 8px 0; }
.seg-k { color: #334155; font-size: 13px; line-height: 1.6; }
.seg-r { color: #334155; font-size: 13px; background: #F5F6FA; border-radius: 10px; padding: 8px 10px; margin-top: 10px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
[class*="st-key-card_"] { background: #FFFFFF; border-radius: 16px; padding: 16px 18px;
       border: 1px solid #E7E9F0; box-shadow: 0 2px 10px rgba(15,23,42,0.05); margin-bottom: 14px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
.stTabs [data-baseweb="tab"] { background: #FFFFFF; border-radius: 10px; padding: 8px 16px;
       border: 1px solid #E7E9F0; height: auto; }
.stTabs [aria-selected="true"] { background: #4F46E5; color: #FFFFFF; border-color: #4F46E5; }
.stTabs [aria-selected="true"] p { color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
h2, h3 { color: #0F172A; letter-spacing: -0.2px; }
</style>
""", unsafe_allow_html=True)

NONE = "(none)"
SUM_MNT = "Sum of all Mnt* columns"
SUM_NUM = "Sum of Num* purchase columns"
MAX_ROWS = 50000
_c = [0]

TEMPLATE_BLANK = "Customer_ID,Age,Income,Total_Spend,Orders,Recency_Days,Response\n"
TEMPLATE_EXAMPLE = TEMPLATE_BLANK + (
    "C001,34,52000,1250,14,12,1\n"
    "C002,45,78000,2890,22,5,0\n"
    "C003,29,31000,310,4,95,0\n"
    "C004,52,95000,4120,31,3,1\n"
    "C005,41,46000,780,9,40,0\n")

REQ_MD = """
**Data requirements**
- **File:** a `.csv` file with column names in the first row. Files above 50,000 rows are sampled.
- **Rows:** one row per customer, not one row per transaction. Use at least 20 rows; 200 or more gives more reliable segments.
- **Columns:** at least 2 numeric columns with different values. Extra columns are welcome. Text columns and ID columns are ignored.
- **Numbers:** write them as `1250` or `$1,250`. Blank cells are filled with the median. Dates are not read, so convert them to days since last purchase first.
- **Response column (optional):** use 1 for responded or churned and 0 for did not.
- **Privacy:** use anonymised data. This app does not save uploaded files.

**What each column unlocks**

| Column | Unlocks |
|---|---|
| Total_Spend | Spend insights, segment names, spend share |
| Orders | Purchase frequency and RFM |
| Recency_Days | RFM tiers |
| Income | Income-based segment names |
| Response | Response or churn chart |

Your column names can differ. The app guesses them, and you can fix the guesses under "Match your columns".
"""

def render_requirements(prefix):
    st.markdown(REQ_MD)
    st.download_button("Download blank template (CSV)", TEMPLATE_BLANK,
                       file_name="customer_template_blank.csv", mime="text/csv", key=f"{prefix}_blank")
    st.download_button("Download example with 5 sample rows", TEMPLATE_EXAMPLE,
                       file_name="customer_template_example.csv", mime="text/csv", key=f"{prefix}_example")

def read_this(items):
    st.markdown("**How to read this**\n" + "\n".join(f"- {x}" for x in items))

def card():
    _c[0] += 1
    return st.container(key=f"card_{_c[0]}")

def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Inter, Arial, sans-serif", color="#0F172A", size=13),
                      legend=dict(orientation="h", y=-0.25, x=0, title_text=""),
                      xaxis=dict(gridcolor="#EEF0F5", zeroline=False),
                      yaxis=dict(gridcolor="#EEF0F5", zeroline=False))
    fig.update_layout(title=dict(font=dict(size=16)))
    return fig

def show(fig, where=st, h=380):
    where.plotly_chart(style(fig, h), theme=None)

def kpi(col, label, value, sub="", accent="#4F46E5"):
    col.markdown(f'<div class="kpi" style="border-top:4px solid {accent}"><div class="kpi-l">{esc(str(label))}</div>'
                 f'<div class="kpi-v">{esc(str(value))}</div><div class="kpi-s">{esc(str(sub))}</div></div>',
                 unsafe_allow_html=True)

def insight(col, big, text):
    col.markdown(f'<div class="ins"><div class="ins-b">{esc(str(big))}</div><div class="ins-t">{esc(str(text))}</div></div>',
                 unsafe_allow_html=True)

def read_any(f):
    last = None
    for enc in ["utf-8", "latin-1"]:
        try:
            f.seek(0)
            return pd.read_csv(f, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last = e
    raise last

MONEY = re.compile(r"^[\s$\u20ac\u00a3\u20b9]*-?[\d,]*\.?\d+[\s%]*$")

def clean_numeric(d):
    d = d.copy()
    d.columns = [str(c).strip() for c in d.columns]
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_datetime64_any_dtype(d[c]):
            s = d[c].dropna().astype(str)
            if len(s) and s.str.match(MONEY).mean() >= 0.9:
                d[c] = pd.to_numeric(d[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
                                     .replace("", np.nan), errors="coerce")
    return d

def is_id_like(name):
    return bool(re.search(r"(^|[^a-z])id($|[^a-z])|^unnamed|index", name.lower()))

def guess(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None

def sil_score(X, labels):
    if len(X) > 5000:
        return silhouette_score(X, labels, sample_size=5000, random_state=42)
    return silhouette_score(X, labels)

@st.cache_data
def load_sample(path):
    return pd.read_csv(path)

@st.cache_data
def cluster(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    return km.labels_, km.cluster_centers_, sil_score(X, km.labels_)

@st.cache_data
def k_scores(X):
    ks = list(range(2, 9))
    inertia, sils = [], []
    for i in ks:
        m = KMeans(n_clusters=i, n_init=10, random_state=42).fit(X)
        inertia.append(m.inertia_)
        sils.append(sil_score(X, m.labels_))
    return ks, inertia, sils

@st.cache_data
def pca_coords(X):
    return PCA(n_components=2, random_state=42).fit_transform(X)

def quality(s):
    return "Strong" if s >= 0.5 else "Good" if s >= 0.35 else "Fair" if s >= 0.25 else "Weak"

def score_5(series, higher_is_better=True):
    s = series if higher_is_better else -series
    return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

def rfm_tier(r, fm):
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3:
        return "Loyal Customers"
    if r >= 4 and fm < 3:
        return "Promising / New"
    if r <= 2 and fm >= 3:
        return "At Risk"
    return "Needs Attention"

st.markdown('<div class="hero"><div class="hero-bar"></div><div class="hero-t">Customer Segmentation and Persona Builder</div>'
            '<div class="hero-s">Turn any customer file into segments, RFM tiers and AI-written marketing personas.</div></div>',
            unsafe_allow_html=True)

st.sidebar.markdown("### Setup")
st.sidebar.markdown("**1. Data**")
up = st.sidebar.file_uploader("Upload a customer CSV", type="csv",
                              help="One row per customer. Any column names work. Leave empty to use the sample data.")
with st.sidebar.expander("Data requirements and template"):
    render_requirements("sb")
try:
    raw = read_any(up) if up else load_sample("data/ifood_df.csv")
except Exception:
    st.error("Could not read this file. Please upload a valid CSV with one row per customer.")
    st.stop()

df = clean_numeric(raw)
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
    st.sidebar.caption(f"Large file: using a random sample of {MAX_ROWS:,} rows.")

num_cols = [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0
            and df[c].nunique() > 1 and not is_id_like(c)]
if len(df) < 20 or len(num_cols) < 2:
    st.error("The file needs at least 20 rows and 2 numeric columns with different values. "
             "Open 'Data requirements and template' in the sidebar for help.")
    st.stop()

mnt = [c for c in df.columns if c.startswith("Mnt") and c not in ("MntTotal", "MntRegularProds")
       and pd.api.types.is_numeric_dtype(df[c])]
numc = [c for c in df.columns if c.startswith("Num") and c != "NumWebVisitsMonth"
        and pd.api.types.is_numeric_dtype(df[c])]

spend_opts = [NONE] + ([SUM_MNT] if mnt else []) + num_cols
freq_opts = [NONE] + ([SUM_NUM] if numc else []) + num_cols
rec_opts = [NONE] + num_cols
inc_opts = [NONE] + num_cols
binary_cols = [c for c in num_cols if set(df[c].dropna().unique()) <= {0, 1}]
resp_opts = [NONE] + binary_cols

def pick(options, preferred, keywords):
    if preferred in options:
        return options.index(preferred)
    g = guess([o for o in options if o != NONE], keywords)
    return options.index(g) if g else 0

with st.sidebar.expander("2. Match your columns", expanded=bool(up)):
    st.caption("We guessed these. Fix any that look wrong, or choose (none).")
    spend_col = st.selectbox("Spend / revenue per customer", spend_opts,
                             index=pick(spend_opts, SUM_MNT, ["spend", "revenue", "sales", "monetary", "amount", "ltv"]))
    freq_col = st.selectbox("Number of purchases / orders", freq_opts,
                            index=pick(freq_opts, SUM_NUM, ["frequency", "orders", "transactions", "purchases", "visits"]))
    rec_col = st.selectbox("Days since last purchase", rec_opts,
                           index=pick(rec_opts, "Recency", ["recency", "days_since", "last_purchase", "dayssince"]))
    inc_col = st.selectbox("Income", inc_opts, index=pick(inc_opts, "Income", ["income", "salary"]))
    resp_col = st.selectbox("Campaign response / churn (0 or 1)", resp_opts,
                            index=pick(resp_opts, "Response", ["response", "churn", "converted", "target"]))

if spend_col == SUM_MNT:
    df["Spend"] = df[mnt].sum(axis=1)
elif spend_col != NONE:
    df["Spend"] = df[spend_col]
if freq_col == SUM_NUM:
    df["Frequency"] = df[numc].sum(axis=1)
elif freq_col != NONE:
    df["Frequency"] = df[freq_col]
if rec_col != NONE:
    df["Recency_"] = df[rec_col]
if inc_col != NONE:
    df["Income"] = df[inc_col]
if resp_col != NONE:
    df["Response"] = df[resp_col]

has_spend, has_freq = spend_col != NONE, freq_col != NONE
has_rec, has_inc, has_resp = rec_col != NONE, inc_col != NONE, resp_col != NONE

feat_options = [c for c in dict.fromkeys(
    [c for c in ["Income", "Spend", "Frequency", "Recency_"] if c in df.columns] + num_cols)
    if c != "Response" and df[c].nunique() > 1]
role_feats = [c for c in ["Income", "Spend", "Frequency", "Recency_"] if c in feat_options]
if len(role_feats) < 2:
    cont = [c for c in feat_options if df[c].nunique() > 10]
    role_feats = (cont or feat_options)[:4]

st.sidebar.markdown("**3. Grouping**")
preset = st.sidebar.radio("Group customers by", ["Recommended columns", "All numeric columns"])
feats = role_feats if preset == "Recommended columns" else feat_options[:12]
with st.sidebar.expander("Advanced: choose features yourself"):
    custom = st.multiselect("Custom features", feat_options, default=feats)
    if len(custom) >= 2:
        feats = custom
k = st.sidebar.slider("Number of segments (k)", 2, 8, 3,
                      help="3 is a good starting point. See the Segments tab for how to choose.")
st.sidebar.markdown("**4. AI (optional)**")
key = st.sidebar.text_input("Gemini API key", type="password", help="Free key from Google AI Studio.")

if len(feats) < 2:
    st.warning("Select at least 2 features.")
    st.stop()
if len(df) <= k:
    st.error("Need more rows than segments.")
    st.stop()

Xraw = df[feats].fillna(df[feats].median())
scaler = StandardScaler().fit(Xraw)
X = scaler.transform(Xraw)
labels, centers, sil = cluster(X, k)
df["Segment"] = labels.astype(str)

extra = [c for c in ["Spend", "Frequency", "Income", "Recency_"] if c in df.columns]
cols = list(dict.fromkeys(feats + extra))
profile = df.groupby("Segment")[cols].mean().round(1)
profile["Size"] = df["Segment"].value_counts()
profile["Share %"] = (profile["Size"] / len(df) * 100).round(1)
avg = df[cols].mean()
sd = df[cols].std().replace(0, 1)

def name_segment(i):
    r = profile.loc[i]
    if has_spend:
        rich = r["Income"] >= avg["Income"] if has_inc else True
        big = r["Spend"] >= avg["Spend"]
        busy = r["Frequency"] >= avg["Frequency"] if has_freq else big
        if big and rich:
            return "Premium Big Spenders", "Offer loyalty perks, premium bundles and early access."
        if big:
            return "Value-Driven Spenders", "Use bundles and targeted discounts to protect spend."
        if busy:
            return "Frequent Light Spenders", "Encourage bigger baskets with bundles and minimum-spend offers."
        return "Low-Engagement Shoppers", "Re-engage with low-cost reminders and first-purchase incentives."
    z = ((r[feats] - avg[feats]) / sd[feats]).sort_values(key=lambda s: s.abs(), ascending=False).head(2)
    label = " & ".join(f"{'High' if v > 0 else 'Low'} {f}" for f, v in z.items())
    return label, f"Tailor messaging to this group's defining traits: {label.lower()}."

names = {i: name_segment(i) for i in profile.index}
profile.insert(0, "Name", [f"Segment {i}: {names[i][0]}" for i in profile.index])
profile["Recommendation"] = [names[i][1] for i in profile.index]
df["Segment Name"] = df["Segment"].map(lambda s: profile.loc[s, "Name"])
seg_color = {n: PAL[j % len(PAL)] for j, n in enumerate(profile["Name"])}

source = esc(up.name) if up else "Sample dataset (iFood-style)"
st.markdown(f'<span class="chip">Data: {source}</span><span class="chip">{len(df):,} customers</span>'
            f'<span class="chip">{len(feats)} grouping features</span><span class="chip">{k} segments</span>',
            unsafe_allow_html=True)
if not up:
    st.caption("You are viewing the sample dataset. To analyse your own file, open the box below, "
               "download the template and upload it in the sidebar.")
with st.expander("Using your own data? Requirements and downloadable template"):
    render_requirements("main")

if has_spend:
    seg_spend = df.groupby("Segment")["Spend"].sum()
    spend_share = (seg_spend / seg_spend.sum() * 100).round(0)

rfm_ready = has_rec and has_freq and has_spend
if rfm_ready:
    rfm = pd.DataFrame({
        "R": score_5(df["Recency_"], higher_is_better=False),
        "F": score_5(df["Frequency"]),
        "M": score_5(df["Spend"]),
    })
    rfm["FM"] = (rfm["F"] + rfm["M"]) / 2
    df["RFM Tier"] = [rfm_tier(r, fm) for r, fm in zip(rfm["R"], rfm["FM"])]
overall_resp = df["Response"].mean() * 100 if has_resp else None

LABELS = {"Recency_": "Days since last purchase", "Frequency": "Number of purchases"}

def compare_text(v, a):
    if not a:
        return "no baseline to compare"
    d = (v / a - 1) * 100
    if abs(d) < 5:
        return "in line with the overall average"
    return f"{abs(d):.0f}% {'above' if d > 0 else 'below'} the overall average"

def segment_insights(i):
    r = profile.loc[i]
    out = [f"**Size:** {int(r['Size']):,} customers, {r['Share %']}% of the base."]
    if has_spend:
        out.append(f"**Value:** drives about {int(spend_share[i])}% of total spend.")
    for f in feats[:4]:
        out.append(f"**{LABELS.get(f, f)}:** {r[f]:,.1f}, {compare_text(r[f], avg[f])}.")
    if has_resp:
        rr = df.loc[df["Segment"] == i, "Response"].mean() * 100
        out.append(f"**Response:** {rr:.1f}% against {overall_resp:.1f}% overall.")
    if rfm_ready:
        mix = df.loc[df["Segment"] == i, "RFM Tier"].value_counts(normalize=True)
        out.append(f"**RFM:** mostly {mix.index[0]} ({mix.iloc[0] * 100:.0f}% of this segment).")
    out.append(f"**Suggested action:** {r['Recommendation']}")
    return out

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Segments", "RFM Tiers", "Customer Lookup", "AI Personas"])

with tab1:
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Customers", f"{len(df):,}", "rows analysed", "#4F46E5")
    kpi(k2, "Segments", k, "groups found", "#F5B800")
    kpi(k3, "Cluster quality", f"{sil:.2f}", f"{quality(sil)} separation", "#10B981")
    if has_spend:
        kpi(k4, "Avg spend", f"{df['Spend'].mean():,.0f}", "per customer", "#F472B6")
    elif has_resp:
        kpi(k4, "Response rate", f"{overall_resp:.1f}%", "overall", "#F472B6")
    else:
        kpi(k4, "Features", len(feats), "used for grouping", "#F472B6")

    st.markdown("### Key insights")
    i1, i2, i3 = st.columns(3)
    big_i = profile["Size"].idxmax()
    insight(i1, f"{profile.loc[big_i, 'Share %']}%", f"of customers sit in {profile.loc[big_i, 'Name']}, the largest segment.")
    if has_spend:
        top_i = spend_share.idxmax()
        insight(i2, f"{int(spend_share[top_i])}%", f"of total spend comes from {profile.loc[top_i, 'Name']}.")
    else:
        insight(i2, str(k), "segments found. Add a spend column to see value by segment.")
    if has_resp:
        rr_all = df.groupby("Segment")["Response"].mean() * 100
        best = rr_all.idxmax()
        insight(i3, f"{rr_all[best]:.1f}%", f"response rate in {profile.loc[best, 'Name']}, against {overall_resp:.1f}% overall.")
    else:
        insight(i3, quality(sil), f"cluster separation (silhouette score {sil:.2f}).")

    st.markdown("### Customer map")
    coords = pca_coords(X)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Segment": df["Segment Name"].values})
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Segment", color_discrete_map=seg_color,
                     opacity=0.7, title="Customers projected onto 2 dimensions")
    with card():
        show(fig, h=460)
    read_this(["Each dot is one customer. Dots close together have similar behaviour.",
               "Colours are segments. Clear colour groups mean well-separated segments."])

with tab2:
    st.markdown("### Segment profiles")
    ui = st.columns(2)
    for n, i in enumerate(profile.index):
        with ui[n % 2]:
            with card():
                nm = profile.loc[i, "Name"]
                st.markdown(f'<div class="seg-n"><span class="dot" style="background:{seg_color[nm]}"></span>{esc(nm)}</div>',
                            unsafe_allow_html=True)
                st.markdown("\n".join(f"- {x}" for x in segment_insights(i)))
    st.markdown("### Segment table")
    st.dataframe(profile.drop(columns=["Recommendation"]), width="stretch")
    if has_resp:
        rr = (df.groupby("Segment Name")["Response"].mean() * 100).reset_index()
        rr.columns = ["Segment", "Response %"]
        figr = px.bar(rr, x="Segment", y="Response %", color="Segment", color_discrete_map=seg_color,
                      title="Response rate by segment")
        with card():
            show(figr, h=360)
    st.markdown("### Choosing the number of segments")
    ks, inertia, sils = k_scores(X)
    c1, c2 = st.columns(2)
    f1 = px.line(x=ks, y=inertia, markers=True, title="Elbow: lower is tighter",
                 labels={"x": "k", "y": "Inertia"})
    f2 = px.line(x=ks, y=sils, markers=True, title="Silhouette: higher is cleaner",
                 labels={"x": "k", "y": "Silhouette"})
    with c1:
        with card():
            show(f1, h=320)
    with c2:
        with card():
            show(f2, h=320)
    read_this(["Pick a k where the elbow flattens and the silhouette is high.",
               "Choose the k that gives segments you can act on, not only the best score."])

with tab3:
    st.markdown("### RFM tiers")
    if not rfm_ready:
        st.info("RFM needs spend, purchase count and days-since-last-purchase columns. "
                "Match them in the sidebar under 'Match your columns'.")
    else:
        order = ["Champions", "Loyal Customers", "Promising / New", "Needs Attention", "At Risk"]
        tc = df["RFM Tier"].value_counts().reindex(order).fillna(0).reset_index()
        tc.columns = ["Tier", "Customers"]
        figt = px.bar(tc, x="Tier", y="Customers", color="Tier", title="Customers per RFM tier")
        with card():
            show(figt, h=360)
        mixdf = pd.crosstab(df["Segment Name"], df["RFM Tier"])
        st.markdown("### RFM tiers inside each segment")
        st.dataframe(mixdf, width="stretch")
        st.markdown("### Average behaviour per tier")
        st.dataframe(df.groupby("RFM Tier")[["Recency_", "Frequency", "Spend"]].mean().round(1), width="stretch")
        read_this(["R, F and M are scored 1 to 5 from recency, frequency and spend.",
                   "Champions buy recently and often. At Risk customers used to buy but have gone quiet."])

with tab4:
    st.markdown("### Customer lookup")
    row = st.number_input("Row number (0 is the first customer)", min_value=0, max_value=len(df) - 1, value=0, step=1)
    cust = df.iloc[int(row)]
    seg_id = cust["Segment"]
    st.markdown(f"**Segment:** {profile.loc[seg_id, 'Name']}")
    if rfm_ready:
        st.markdown(f"**RFM tier:** {cust['RFM Tier']}")
    cmp_df = pd.DataFrame({"Customer": cust[cols].astype(float),
                           "Segment average": profile.loc[seg_id, cols].astype(float),
                           "Overall average": avg[cols].astype(float)}).round(1)
    st.dataframe(cmp_df, width="stretch")
    st.markdown(f"**Suggested action:** {profile.loc[seg_id, 'Recommendation']}")

with tab5:
    st.markdown("### AI personas")
    st.caption("Add a free Gemini API key in the sidebar, then generate marketing personas for each segment.")
    GEMINI_MODEL = "gemini-2.5-flash"
    if not key:
        st.info("Enter your Gemini API key in the sidebar (section 4) to enable this tab.")
    elif st.button("Generate personas"):
        summary = profile.drop(columns=["Recommendation"]).to_string()
        prompt = ("You are a marketing analyst. Below is a table of customer segments with average values. "
                  "For each segment write a short persona: a name, who they are, what they value, and two "
                  "marketing actions. Use plain language and only the numbers given.\n\n" + summary)
        try:
            from google import genai
            client = genai.Client(api_key=key)
            with st.spinner("Writing personas..."):
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            st.markdown(resp.text)
        except Exception as e:
            st.error(f"The AI request failed: {e}")
