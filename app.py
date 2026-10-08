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
        if d[c].dtype == object:
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
    st.error("The file needs at least 20 rows and 2 numeric columns with different values.")
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Segments", "RFM Tiers", "Customer Lookup", "AI Personas"])

with tab1:
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Customers", f"{len(df):,}", "rows analysed", "#4F46E5")
    kpi(k2, "Segments", k, "groups found", "#F5B800")
    kpi(k3, "Cluster quality", f"{sil:.2f}", f"{quality(sil)} separation", "#10B981")
    if has_spend:
        kpi(k4, "Average spend", f"{df['Spend'].mean():,.0f}", "per customer", "#F472B6")
    else:
        kpi(k4, "Features used", len(feats), "for grouping", "#F472B6")
    st.write("")

    big = profile["Size"].idxmax()
    i1, i2, i3 = st.columns(3)
    if has_spend:
        seg_spend = df.groupby("Segment")["Spend"].sum()
        spend_share = (seg_spend / seg_spend.sum() * 100).round(0)
        top = spend_share.idxmax()
        insight(i1, f"{int(spend_share[top])}% of spend",
                f"comes from {profile.loc[top, 'Name']}, only {profile.loc[top, 'Share %']}% of customers.")
        insight(i2, f"{profile.loc[big, 'Share %']}% of customers",
                f"are in the largest group, {profile.loc[big, 'Name']}, which brings about {int(spend_share[big])}% of spend.")
        lo, hi = profile["Spend"].min(), profile["Spend"].max()
        insight(i3, f"{hi / lo:.0f}x gap" if lo > 0 else f"{hi:,.0f} top",
                f"between the highest and lowest average spend ({hi:,.0f} vs {lo:,.0f}).")
    else:
        small = profile["Size"].idxmin()
        insight(i1, f"{profile.loc[big, 'Share %']}%", f"of customers are in {profile.loc[big, 'Name']}, the largest group.")
        insight(i2, f"{profile.loc[small, 'Share %']}%", f"of customers are in {profile.loc[small, 'Name']}, the smallest group.")
        insight(i3, "Tip", "Match a spend column in the sidebar to unlock spend-based insights.")
    st.write("")

    with card():
        coords = pca_coords(X)
        plot_df = df.assign(PC1=coords[:, 0], PC2=coords[:, 1])
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Segment Name", color_discrete_map=seg_color,
                         title="Customer map: each dot is a customer, closer dots are more similar")
        show(fig, st, 460)
    st.download_button("Download customers with segments (CSV)",
                       df.drop(columns=["Segment"]).to_csv(index=False),
                       file_name="customers_with_segments.csv", mime="text/csv")

with tab2:
    st.markdown("### Your segments")
    metric_cols = [c for c in ["Spend", "Income", "Frequency"] if c in profile.columns] or feats[:3]
    rows = list(profile.index)
    for start in range(0, len(rows), 3):
        cs = st.columns(3)
        for col, i in zip(cs, rows[start:start + 3]):
            r = profile.loc[i]
            lines = "<br>".join(f"{esc(m)}: <b>{r[m]:,.1f}</b>" for m in metric_cols)
            col.markdown(
                f'<div class="seg"><div class="seg-n"><span class="dot" style="background:{seg_color[r["Name"]]}"></span>'
                f'{esc(r["Name"])}</div><div class="seg-m">{int(r["Size"]):,} customers | {r["Share %"]}%</div>'
                f'<div class="seg-k">{lines}</div><div class="seg-r">{esc(r["Recommendation"])}</div></div>',
                unsafe_allow_html=True)

    ycol = "Spend" if has_spend else feats[0]
    c1, c2 = st.columns(2)
    with c1:
        with card():
            show(px.bar(profile.reset_index(), x="Name", y=ycol, color="Name", color_discrete_map=seg_color,
                        title=f"Average {ycol} per segment").update_layout(showlegend=False, xaxis_title=None))
    with c2:
        with card():
            show(px.pie(profile.reset_index(), names="Name", values="Size", color="Name",
                        color_discrete_map=seg_color, title="Share of customers", hole=0.5))
    if has_resp:
        with card():
            resp = df.groupby("Segment Name")["Response"].mean().mul(100).round(1).reset_index()
            show(px.bar(resp, x="Segment Name", y="Response", color="Segment Name", color_discrete_map=seg_color,
                        title="Response / churn rate (%) by segment").update_layout(showlegend=False, xaxis_title=None), st, 320)
    with st.expander("See the full segment table"):
        st.dataframe(profile, column_config={
            "Share %": st.column_config.ProgressColumn("Share %", min_value=0, max_value=100, format="%.1f%%")})
    with st.expander("Not sure how many segments? (elbow and silhouette)"):
        st.caption("Pick the k where the elbow bends and the silhouette stays high while segments stay useful.")
        ks, inertia, sils = k_scores(X)
        e1, e2 = st.columns(2)
        show(px.line(x=ks, y=inertia, markers=True, labels={"x": "k", "y": "Inertia"}, title="Elbow curve"), e1, 320)
        show(px.line(x=ks, y=sils, markers=True, labels={"x": "k", "y": "Silhouette"}, title="Silhouette by k"), e2, 320)

with tab3:
    st.markdown("### RFM tiers")
    st.caption("Each customer is scored 1-5 on Recency, Frequency and Monetary value, then given a simple tier label.")
    if not (has_rec and has_freq and has_spend):
        st.info("RFM needs three columns matched in the sidebar: days since last purchase, number of purchases and spend.")
    else:
        rfm = pd.DataFrame({
            "R": score_5(df["Recency_"], higher_is_better=False),
            "F": score_5(df["Frequency"]),
            "M": score_5(df["Spend"]),
        })
        rfm["FM"] = (rfm["F"] + rfm["M"]) / 2
        df["RFM Tier"] = [rfm_tier(r, fm) for r, fm in zip(rfm["R"], rfm["FM"])]
        tier = (df.groupby("RFM Tier")
                  .agg(Customers=("Spend", "size"), AvgSpend=("Spend", "mean"), TotalSpend=("Spend", "sum"))
                  .round(0).reset_index())
        tier["Share of spend %"] = (tier["TotalSpend"] / tier["TotalSpend"].sum() * 100).round(1)
        x1, x2 = st.columns(2)
        with x1:
            with card():
                show(px.bar(tier, x="RFM Tier", y="Customers", color="RFM Tier",
                            title="Customers per tier").update_layout(showlegend=False, xaxis_title=None), st, 340)
        with x2:
            with card():
                show(px.bar(tier, x="RFM Tier", y="Share of spend %", color="RFM Tier",
                            title="Share of spend per tier").update_layout(showlegend=False, xaxis_title=None), st, 340)
        with card():
            st.markdown(
                "- **Champions:** bought recently, often and spend a lot. Reward them.\n"
                "- **Loyal Customers:** steady buyers. Upsell and keep engaged.\n"
                "- **Promising / New:** bought recently but low value so far. Nurture.\n"
                "- **At Risk:** good value but have not bought recently. Win them back.\n"
                "- **Needs Attention:** low value and not recent. Low-cost reminders only.")
        with st.expander("See tier table and overlap with segments"):
            st.dataframe(tier)
            st.caption("How the K-Means segments overlap with RFM tiers (customer counts):")
            st.dataframe(pd.crosstab(df["Segment Name"], df["RFM Tier"]))

with tab4:
    st.markdown("### Where would a new customer fit?")
    st.caption("Enter customer details and press the button. Defaults are the dataset medians.")
    vals = {}
    with card():
        with st.form("lookup"):
            cols_in = st.columns(min(len(feats), 3))
            for i, f in enumerate(feats):
                lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
                vals[f] = cols_in[i % len(cols_in)].number_input(f, min_value=lo, max_value=hi, value=med)
            st.form_submit_button("Find segment")
    new = pd.DataFrame([vals])[feats]
    z = scaler.transform(new)
    seg_new = str(int(np.argmin(((centers - z) ** 2).sum(axis=1))))
    with card():
        st.markdown(f"#### This customer fits: {profile.loc[seg_new, 'Name']}")
        st.write(profile.loc[seg_new, "Recommendation"])
        comp = pd.DataFrame({"This customer": new.iloc[0], "Segment average": profile.loc[seg_new, feats]}).reset_index()
        comp = comp.rename(columns={"index": "Feature"}).melt(id_vars="Feature", var_name="Who", value_name="Value")
        show(px.bar(comp, x="Feature", y="Value", color="Who", barmode="group",
                    title="This customer vs their segment average"), st, 340)

with tab5:
    st.markdown("### AI persona builder")
    st.caption("Needs a free Gemini API key from Google AI Studio, entered in the sidebar.")
    with card():
        seg = st.selectbox("Choose a segment", profile.index.tolist(),
                           format_func=lambda s: profile.loc[s, "Name"])
        go = st.button("Generate persona")
        if go:
            if not key:
                st.info("Add a Gemini API key in the sidebar first.")
            else:
                from google import genai
                client = genai.Client(api_key=key.strip())
                stats = profile.loc[seg].drop(["Name", "Recommendation"]).to_dict()
                prompt = f"""You are a senior marketing strategist. Segment averages: {stats}.
Dataset averages for comparison: {avg.round(1).to_dict()}.
Create: 1) a persona name and one-line description, 2) motivations and pain points,
3) best channels, 4) three campaign ideas, 5) one risk. Use only the numbers provided;
do not invent data. Keep it under 200 words, in bullets."""
                text, last_error = None, ""
                with st.spinner("Writing persona..."):
                    for model_name in ["gemini-flash-latest", "gemini-flash-lite-latest"]:
                        for attempt in range(3):
                            try:
                                out = client.models.generate_content(model=model_name, contents=prompt)
                                text = out.text
                                break
                            except Exception as e:
                                last_error = str(e)
                                time.sleep(2 * (attempt + 1))
                        if text:
                            break
                if text:
                    st.markdown(text)
                else:
                    st.warning("Gemini is busy right now. Please try again in a few minutes.")
                    st.caption(last_error[:200])
