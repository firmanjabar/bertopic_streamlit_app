# app.py (patched for BERTopic: remove unsupported 'seed' kwarg; add UMAP/HDBSCAN seeds)
import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import numpy as np
import random

st.set_page_config(page_title="BERTopic Visual Demo", page_icon="🧵", layout="wide")
st.title("🧵 BERTopic — Topic Modeling Demo (Multilingual)")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Embedding model", value="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    min_topic_size = st.number_input("min_topic_size", min_value=2, max_value=200, value=10, step=1)
    top_n_words = st.number_input("Top-N words per topic", min_value=3, max_value=20, value=10, step=1)
    low_memory = st.checkbox("Low memory mode", value=False)
    language = st.selectbox("Language", ["multilingual","english","indonesian"], index=0)
    seed = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

st.subheader("1) Upload Data")
uploaded = st.file_uploader("Upload CSV (default sample used if empty)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/sample.csv")

if len(df) == 0:
    st.error("Empty dataset.")
    st.stop()

st.write("Preview:")
st.dataframe(df.head(10), use_container_width=True)

text_col = st.selectbox("Text column", options=list(df.columns), index=0)
date_col = st.selectbox("Date column (optional, YYYY-MM-DD)", options=["(None)"] + list(df.columns), index=0)

docs = df[text_col].astype(str).tolist()

dates = None
if date_col != "(None)":
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        st.warning("All date values are NaT; ignoring date column.")
        dates = None

st.markdown("---")
st.subheader("2) Run BERTopic")
if st.button("🚀 Fit Model"):
    # Set seeds for reproducibility
    np.random.seed(int(seed))
    random.seed(int(seed))

    with st.spinner("Building model..."):
        embedder = SentenceTransformer(model_name)
        vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words=None)

        # Configure UMAP/HDBSCAN with random_state/min_cluster_size
        umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=int(seed), low_memory=low_memory)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=int(min_topic_size), metric="euclidean",
                                        cluster_selection_method="eom", prediction_data=True)

        topic_model = BERTopic(
            embedding_model=embedder,
            vectorizer_model=vectorizer_model,
            language=language,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            low_memory=low_memory,
            verbose=True
        )

        if dates is not None:
            topics, probs = topic_model.fit_transform(docs, timestamps=dates.dt.to_pydatetime().tolist())
        else:
            topics, probs = topic_model.fit_transform(docs)

    st.success("Done.")
    info = topic_model.get_topic_info()
    st.markdown("### Topic Info")
    st.dataframe(info, use_container_width=True, height=350)

    st.markdown("### Visualize Topics")
    try:
        fig_topics = topic_model.visualize_topics(width=900, height=600)
        st.plotly_chart(fig_topics, use_container_width=True)
    except Exception as e:
        st.warning(f"visualize_topics failed: {e}")

    st.markdown("### Top Words per Topic")
    try:
        fig_bar = topic_model.visualize_barchart(top_n_topics=10, n_words=top_n_words, width=900, height=600)
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.warning(f"visualize_barchart failed: {e}")

    if dates is not None:
        st.markdown("### Topics Over Time")
        try:
            tot = topic_model.topics_over_time(docs, dates.dt.to_pydatetime().tolist(), nr_bins=10)
            fig_time = topic_model.visualize_topics_over_time(tot, width=900, height=600)
            st.plotly_chart(fig_time, use_container_width=True)
        except Exception as e:
            st.warning(f"topics_over_time failed: {e}")

    st.markdown("### Download topic_info.csv")
    st.download_button("⬇️ Download", data=info.to_csv(index=False).encode("utf-8"),
                       file_name="topic_info.csv", mime="text/csv")
else:
    st.info("Configure parameters in the sidebar, then click **Fit Model**.")
