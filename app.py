import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

px.defaults.color_discrete_sequence = ["#F5B800", "#10B981", "#6C63FF", "#F472B6", "#22D3EE", "#F97316"]

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
st.title("Customer Segmentation and Persona Builder")
st.caption("Group customers by behaviour, see who matters most, and get AI-written marketing personas.")

@st.cache_data
def load(path):
    return pd.read_csv(path)

@st.cache_data
def k_scores(X):
    ks = list(range(2, 9))
    inertia, sils = [], []
    for i in ks:
        m = KMeans(n_clusters=i, n_init=10, random_state=42).fit(X)
        inertia.append(m.inertia_)
        sils.append(silhouette_score(X, m.labels_))
    return ks, inertia, sils

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

up = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
df = pd.read_csv(up) if up else load("data/ifood_df.csv")

mnt = [c for c in df.columns if c.startswith("Mnt") and c not in ("MntTotal", "MntRegularProds")]
num = [c for c in df.columns if c.startswith("Num") and c != "NumWebVisitsMonth"]
df["Spend"] = df[mnt].sum(axis=1)
df["Frequency"] = df[num].sum(axis=1)
df["Recency_"] = df["Recency"] if "Recency" in df.columns else 0

feat_options = [c for c in ["Income", "Age", "Spend", "Frequency", "Recency_",
                            "NumWebVisitsMonth", "NumDealsPurchases"] if c in df.columns]

presets = {
    "Value and engagement (recommended)": ["Income", "Spend", "Frequency"],
    "Spending behaviour": ["Spend", "Frequency", "NumDealsPurchases"],
    "Full customer profile": feat_options,
}
presets = {n: [c for c in cols if c in feat_options] for n, cols in presets.items()}

st.sidebar.header("Settings")
preset = st.sidebar.radio("What should we group customers by?", list(presets.keys()))
feats = presets[preset]
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

Xraw = df[feats].fillna(df[feats].median())
scaler = StandardScaler().fit(Xraw)
X = scaler.transform(Xraw)
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
df["Segment"] = km.labels_.astype(str)
sil = silhouette_score(X, km.labels_)

extra = [c for c in ["Spend", "Frequency", "Income", "NumDealsPurchases"] if c in df.columns]
cols = list(dict.fromkeys(feats + extra))
profile = df.groupby("Segment")[cols].mean().round(1)
profile["Size"] = df["Segment"].value_counts()
profile["Share %"] = (profile["Size"] / len(df) * 100).round(1)
avg = df[cols].mean()

def name_segment(r):
    rich = r["Income"] >= avg["Income"] if "Income" in cols else True
    big = r["Spend"] >= avg["Spend"]
    deals = r["NumDealsPurchases"] >= avg["NumDealsPurchases"] if "NumDealsPurchases" in cols else False
    if big and rich:
        return "Premium Big Spenders", "Offer loyalty perks, premium bundles and early access."
    if big:
        return "Value-Driven Spenders", "Use bundles and targeted discounts to protect spend."
    if deals:
        return "Deal Seekers", "Run time-limited offers, but watch margins."
    return "Low-Engagement Shoppers", "Re-engage with low-cost reminders and first-purchase incentives."

names = {i: name_segment(profile.loc[i]) for i in profile.index}
profile.insert(0, "Name", [f"Segment {i}: {names[i][0]}" for i in profile.index])
profile["Recommendation"] = [names[i][1] for i in profile.index]
df["Segment Name"] = df["Segment"].map(lambda s: profile.loc[s, "Name"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Segments", "RFM Tiers", "Customer Lookup", "AI Personas"])

with tab1:
    with st.expander("How to use this dashboard", expanded=False):
        st.markdown(
            "1. Pick how to group customers in the sidebar.\n"
            "2. Read the key takeaways below.\n"
            "3. Open the other tabs for details, RFM tiers, lookup of a new customer and AI personas."
        )
        c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Segments", k)
    c3.metric("Silhouette score", f"{sil:.2f}", help="Higher means cleaner, more separate segments. Above 0.4 is decent.")
    c4.metric("Average spend", f"{df['Spend'].mean():,.0f}")

    seg_spend = df.groupby("Segment")["Spend"].sum()
    spend_share = (seg_spend / seg_spend.sum() * 100).round(0)
    top = spend_share.idxmax()
    big = profile["Size"].idxmax()
    st.subheader("Key takeaways")
    st.markdown(
        f"- **{profile.loc[top, 'Name']}** is {profile.loc[top, 'Share %']}% of customers but about "
        f"{int(spend_share[top])}% of total spend.\n"
        f"- The largest group is **{profile.loc[big, 'Name']}** ({profile.loc[big, 'Share %']}% of customers), "
        f"contributing about {int(spend_share[big])}% of spend.\n"
        f"- Average spend ranges from {profile['Spend'].min():,.0f} to {profile['Spend'].max():,.0f} across segments."
    )

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = df.assign(PC1=coords[:, 0], PC2=coords[:, 1])
    st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="Segment Name",
                    title="Segment map: each dot is a customer, closer dots are more similar"))
    st.download_button("Download customers with segments (CSV)",
                       df.drop(columns=["Recency_"]).to_csv(index=False),
                       file_name="customers_with_segments.csv", mime="text/csv")

with tab2:
    st.subheader("Segment profiles")
    st.caption("Averages per segment. Size is the number of customers.")
    st.dataframe(profile)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(profile.reset_index(), x="Name", y="Spend", title="Average spend per segment"))
    b.plotly_chart(px.pie(profile.reset_index(), names="Name", values="Size", title="Share of customers"))
    if "Response" in df.columns:
        resp = df.groupby("Segment Name")["Response"].mean().mul(100).round(1).reset_index()
        st.plotly_chart(px.bar(resp, x="Segment Name", y="Response",
                        title="Last campaign response rate (%) by segment"))
    with st.expander("How to choose k (elbow and silhouette)"):
        st.caption("Look for where the elbow bends and the silhouette stays high while segments stay useful.")
        ks, inertia, sils = k_scores(X)
        st.plotly_chart(px.line(x=ks, y=inertia, markers=True,
                        labels={"x": "k", "y": "Inertia"}, title="Elbow curve"))
        st.plotly_chart(px.line(x=ks, y=sils, markers=True,
                        labels={"x": "k", "y": "Silhouette"}, title="Silhouette score by k"))

with tab3:
    st.subheader("RFM tiers")
    st.caption("RFM scores each customer 1-5 on Recency (how recently they bought), Frequency (how often) "
               "and Monetary value (how much). The tier is a simple label built from those scores.")
    if "Recency" not in df.columns:
        st.info("This dataset has no Recency column, so RFM tiers are unavailable.")
    else:
        rfm = pd.DataFrame({
            "R": score_5(df["Recency"], higher_is_better=False),
            "F": score_5(df["Frequency"]),
            "M": score_5(df["Spend"]),
        })
        rfm["FM"] = (rfm["F"] + rfm["M"]) / 2
        rfm["Tier"] = [rfm_tier(r, fm) for r, fm in zip(rfm["R"], rfm["FM"])]
        df["RFM Tier"] = rfm["Tier"]
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
        cross = pd.crosstab(df["Segment Name"], df["RFM Tier"])
        st.caption("How the K-Means segments overlap with RFM tiers (customer counts):")
        st.dataframe(cross)

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
