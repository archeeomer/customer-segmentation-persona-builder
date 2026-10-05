import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Persona Builder", layout="wide")
st.title("Customer Segmentation and Persona Builder")

@st.cache_data
def load(path):
    return pd.read_csv(path)

up = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
df = pd.read_csv(up) if up else load("data/ifood_df.csv")

mnt = [c for c in df.columns if c.startswith("Mnt") and c not in ("MntTotal", "MntRegularProds")]
num = [c for c in df.columns if c.startswith("Num") and c != "NumWebVisitsMonth"]
df["Spend"] = df[mnt].sum(axis=1)
df["Frequency"] = df[num].sum(axis=1)
df["Recency_"] = df["Recency"] if "Recency" in df.columns else 0

feat_options = [c for c in ["Income", "Age", "Spend", "Frequency", "Recency_",
                            "NumWebVisitsMonth", "NumDealsPurchases"] if c in df.columns]
default_feats = [c for c in ["Income", "Spend", "Frequency"] if c in feat_options]
feats = st.sidebar.multiselect("Features for clustering", feat_options, default=default_feats)
if len(feats) < 2:
    st.warning("Select at least 2 features in the sidebar.")
    st.stop()
k = st.sidebar.slider("Number of segments (k)", 2, 8, 4)

X = StandardScaler().fit_transform(df[feats].fillna(df[feats].median()))
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
df["Segment"] = km.labels_.astype(str)
sil = silhouette_score(X, km.labels_)

c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Segments", k)
c3.metric("Silhouette score", f"{sil:.2f}")

coords = PCA(n_components=2, random_state=42).fit_transform(X)
df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
st.plotly_chart(px.scatter(df, x="PC1", y="PC2", color="Segment",
                title="Segment map (PCA view)"))

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

st.subheader("Segment profiles")
st.dataframe(profile)

a, b = st.columns(2)
a.plotly_chart(px.bar(profile.reset_index(), x="Segment", y="Spend",
               title="Average spend per segment"))
b.plotly_chart(px.pie(profile.reset_index(), names="Segment", values="Size",
               title="Segment size"))

if "Response" in df.columns:
    resp = df.groupby("Segment")["Response"].mean().mul(100).round(1).reset_index()
    st.plotly_chart(px.bar(resp, x="Segment", y="Response",
                   title="Last campaign response rate (%) by segment"))

with st.expander("How to choose k (elbow and silhouette)"):
    ks = list(range(2, 9))
    inertia, sils = [], []
    for i in ks:
        m = KMeans(n_clusters=i, n_init=10, random_state=42).fit(X)
        inertia.append(m.inertia_)
        sils.append(silhouette_score(X, m.labels_))
    st.plotly_chart(px.line(x=ks, y=inertia, markers=True,
                    labels={"x": "k", "y": "Inertia"}, title="Elbow curve"))
    st.plotly_chart(px.line(x=ks, y=sils, markers=True,
                    labels={"x": "k", "y": "Silhouette"}, title="Silhouette score by k"))

st.subheader("AI persona builder")
key = st.sidebar.text_input("Gemini API key", type="password")
seg = st.selectbox("Choose a segment", profile.index.tolist())

if st.button("Generate persona"):
    if not key:
        st.info("Add a Gemini API key in the sidebar first.")
    else:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            stats = profile.loc[seg].drop(["Name", "Recommendation"]).to_dict()
            prompt = f"""You are a senior marketing strategist. Segment averages: {stats}.
Dataset averages for comparison: {avg.round(1).to_dict()}.
Create: 1) a persona name and one-line description, 2) motivations and pain points,
3) best channels, 4) three campaign ideas, 5) one risk. Use only the numbers provided;
do not invent data. Keep it under 200 words, in bullets."""
                        import time
            text = None
            for model_name in ["gemini-flash-latest", "gemini-flash-lite-latest"]:
                for attempt in range(3):
                    try:
                        out = client.models.generate_content(model=model_name, contents=prompt)
                        text = out.text
                        break
                    except Exception:
                        time.sleep(2 * (attempt + 1))
                if text:
                    break
            if text:
                st.markdown(text)
            else:
                st.warning("Gemini is busy right now. Please try again in a few minutes.")
        except Exception as e:
            st.error(f"Gemini error: {e}")
