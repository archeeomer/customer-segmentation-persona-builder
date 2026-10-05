import pandas as pd, numpy as np, streamlit as st, plotly.express as px
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
df["Recency_"] = df["Recency"] if "Recency" in df else 0

feat_options = [c for c in ["Income", "Age", "Spend", "Frequency", "Recency_",
                            "NumWebVisitsMonth", "NumDealsPurchases"] if c in df.columns]
feats = st.sidebar.multiselect("Features for clustering", feat_options, default=feat_options)
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
                title="Segment map (PCA view)"), use_container_width=True)

cols = list(dict.fromkeys(feats + ["Spend"]))
profile = df.groupby("Segment")[cols].mean().round(1)
profile["Size"] = df["Segment"].value_counts()
profile["Share %"] = (profile["Size"] / len(df) * 100).round(1)
st.subheader("Segment profiles")
st.dataframe(profile)

a, b = st.columns(2)
a.plotly_chart(px.bar(profile.reset_index(), x="Segment", y="Spend",
               title="Average spend per segment"), use_container_width=True)
b.plotly_chart(px.pie(profile.reset_index(), names="Segment", values="Size",
               title="Segment size"), use_container_width=True)

if "Response" in df.columns:
    resp = df.groupby("Segment")["Response"].mean().mul(100).round(1).reset_index()
    st.plotly_chart(px.bar(resp, x="Segment", y="Response",
                   title="Last campaign response rate (%) by segment"), use_container_width=True)

st.session_state["profile"] = profile
