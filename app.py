import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Persona Builder", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #FFF9E0 0%, #FFE98A 100%); }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-radius: 0 24px 24px 0; }
[data-testid="stMetric"] {
    background-color: #FFFFFF; border-radius: 16px; padding: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background-color: #FFFFFF; border-radius: 16px; padding: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
}
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)
px.defaults.color_discrete_sequence = ["#F5B800", "#10B981", "#6C63FF", "#F472B6", "#22D3EE", "#F97316"]

st.title("Customer Segmentation and Persona Builder")
st.caption("Upload any customer CSV, match a few columns, and get segments, RFM tiers and AI personas.")

NONE = "(none)"
SUM_MNT = "Sum of all Mnt* columns"
SUM_NUM = "Sum of Num* purchase columns"
MAX_ROWS = 50000

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

def guess(cols, keywords, binary_only=None):
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None

@st.cache_data
def load_sample(path):
    return pd.read_csv(path)

@st.cache_data
def k_scores(X):
    ks = list(range(2, 9))
    inertia, sils = [], []
    for i in ks:
        m = KMeans(n_clusters=i, n_init=10, random_state=42).fit(X)
        inertia.append(m.inertia_)
        sils.append(sil_score(X, m.labels_))
    return ks, inertia, sils

def sil_score(X, labels):
    if len(X) > 5000:
        return silhouette_score(X, labels, sample_size=5000, random_state=42)
    return silhouette_score(X, labels)

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

st.sidebar.header("1. Data")
up = st.sidebar.file_uploader("Upload your own customer CSV (optional)", type="csv",
                              help="One row per customer. Any column names work; you can match them below.")
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

st.sidebar.header("3. Grouping")
preset = st.sidebar.radio("Group customers by", ["Recommended columns", "All numeric columns"])
feats = role_feats if preset == "Recommended columns" else feat_options[:12]
with st.sidebar.expander("Advanced: choose features yourself"):
    custom = st.multiselect("Custom features", feat_options, default=feats)
    if len(custom) >= 2:
        feats = custom
k = st.sidebar.slider("Number of segments (k)", 2, 8, 3,
                      help="3 is a good starting point. See the Segments tab for how to choose.")
key = st.sidebar.text_input("Gemini API key (for AI personas)", type="password")

if len(feats) < 2:
    st.warning("Select at least 2 features.")
    st.stop()
if len(df) <= k:
    st.error("Need more rows than segments.")
    st.stop()

Xraw = df[feats].fillna(df[feats].median())
scaler = StandardScaler().fit(Xraw)
X = scaler.transform(Xraw)
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
df["Segment"] = km.labels_.astype(str)
sil = sil_score(X, km.labels_)

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Segments", "RFM Tiers", "Customer Lookup", "AI Personas"])

with tab1:
    with st.expander("How to use this dashboard", expanded=False):
        st.markdown(
            "1. Upload a customer CSV in the sidebar, or use the sample data.\n"
            "2. Check the column matching, then pick how to group customers.\n"
            "3. Read the takeaways, then explore the other tabs."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Segments", k)
    c3.metric("Silhouette score", f"{sil:.2f}", help="Higher means cleaner, more separate segments. Above 0.4 is decent.")
    if has_spend:
        c4.metric("Average spend", f"{df['Spend'].mean():,.0f}")
    else:
        c4.metric("Features used", len(feats))

    st.subheader("Key takeaways")
    big = profile["Size"].idxmax()
    if has_spend:
        seg_spend = df.groupby("Segment")["Spend"].sum()
        spend_share = (seg_spend / seg_spend.sum() * 100).round(0)
        top = spend_share.idxmax()
        st.markdown(
            f"- **{profile.loc[top, 'Name']}** is {profile.loc[top, 'Share %']}% of customers but about "
            f"{int(spend_share[top])}% of total spend.\n"
            f"- The largest group is **{profile.loc[big, 'Name']}** ({profile.loc[big, 'Share %']}% of customers), "
            f"contributing about {int(spend_share[big])}% of spend.\n"
            f"- Average spend ranges from {profile['Spend'].min():,.0f} to {profile['Spend'].max():,.0f} across segments."
        )
    else:
        small = profile["Size"].idxmin()
        st.markdown(
            f"- The largest group is **{profile.loc[big, 'Name']}** ({profile.loc[big, 'Share %']}% of customers).\n"
            f"- The smallest group is **{profile.loc[small, 'Name']}** ({profile.loc[small, 'Share %']}% of customers).\n"
            "- Match a spend column in the sidebar to unlock spend-based insights."
        )

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = df.assign(PC1=coords[:, 0], PC2=coords[:, 1])
    st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="Segment Name",
                    title="Segment map: each dot is a customer, closer dots are more similar"))
    st.download_button("Download customers with segments (CSV)",
                       df.drop(columns=["Segment"]).to_csv(index=False),
                       file_name="customers_with_segments.csv", mime="text/csv")

with tab2:
    st.subheader("Segment profiles")
    st.caption("Averages per segment. Size is the number of customers.")
    st.dataframe(profile)
    a, b = st.columns(2)
    ycol = "Spend" if has_spend else feats[0]
    a.plotly_chart(px.bar(profile.reset_index(), x="Name", y=ycol, title=f"Average {ycol} per segment"))
    b.plotly_chart(px.pie(profile.reset_index(), names="Name", values="Size", title="Share of customers"))
    if has_resp:
        resp = df.groupby("Segment Name")["Response"].mean().mul(100).round(1).reset_index()
        st.plotly_chart(px.bar(resp, x="Segment Name", y="Response", title="Response / churn rate (%) by segment"))
    with st.expander("How to choose k (elbow and silhouette)"):
        st.caption("Look for where the elbow bends and the silhouette stays high while segments stay useful.")
        ks, inertia, sils = k_scores(X)
        st.plotly_chart(px.line(x=ks, y=inertia, markers=True,
                        labels={"x": "k", "y": "Inertia"}, title="Elbow curve"))
        st.plotly_chart(px.line(x=ks, y=sils, markers=True,
                        labels={"x": "k", "y": "Silhouette"}, title="Silhouette score by k"))

with tab3:
    st.subheader("RFM tiers")
    st.caption("RFM scores each customer 1-5 on Recency, Frequency and Monetary value, then labels them with a tier.")
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
        x1.plotly_chart(px.bar(tier, x="RFM Tier", y="Customers", title="Customers per tier"))
        x2.plotly_chart(px.bar(tier, x="RFM Tier", y="Share of spend %", title="Share of spend per tier"))
        st.dataframe(tier)
        st.markdown(
            "- **Champions:** bought recently, often and spend a lot. Reward them.\n"
            "- **Loyal Customers:** steady buyers. Upsell and keep engaged.\n"
            "- **Promising / New:** bought recently but low value so far. Nurture.\n"
            "- **At Risk:** good value but have not bought recently. Win them back.\n"
            "- **Needs Attention:** low value and not recent. Low-cost reminders only."
        )
        st.caption("How the K-Means segments overlap with RFM tiers (customer counts):")
        st.dataframe(pd.crosstab(df["Segment Name"], df["RFM Tier"]))

with tab4:
    st.subheader("Which segment would a new customer fall into?")
    st.caption("Enter customer details. Defaults are the dataset medians.")
    vals = {}
    cols_in = st.columns(min(len(feats), 3))
    for i, f in enumerate(feats):
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        vals[f] = cols_in[i % len(cols_in)].number_input(f, min_value=lo, max_value=hi, value=med)
    new = pd.DataFrame([vals])[feats]
    seg_new = str(km.predict(scaler.transform(new))[0])
    st.success(f"This customer fits **{profile.loc[seg_new, 'Name']}**")
    st.write(profile.loc[seg_new, "Recommendation"])
    comp = pd.DataFrame({"This customer": new.iloc[0], "Segment average": profile.loc[seg_new, feats]}).reset_index()
    comp = comp.rename(columns={"index": "Feature"}).melt(id_vars="Feature", var_name="Who", value_name="Value")
    st.plotly_chart(px.bar(comp, x="Feature", y="Value", color="Who", barmode="group",
                    title="This customer vs their segment average"))

with tab5:
    st.subheader("AI persona builder")
    st.caption("Needs a free Gemini API key from Google AI Studio, entered in the sidebar.")
    seg = st.selectbox("Choose a segment", profile.index.tolist(),
                       format_func=lambda s: profile.loc[s, "Name"])
    if st.button("Generate persona"):
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
