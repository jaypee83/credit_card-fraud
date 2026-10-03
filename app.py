"""
Credit Card Fraud Detection – Streamlit App
Loads the models trained in the Colab notebook (artifacts/ folder) and serves:
  1. Single-transaction prediction (pick a test transaction or enter values)
  2. Batch prediction on an uploaded CSV (with downloadable results)
  3. Model performance dashboard (metrics, confusion matrices, ROC / PR curves)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = os.environ.get("ARTIFACT_DIR", "artifacts")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ART, "meta.json")) as f:
        meta = json.load(f)
    scaler = joblib.load(os.path.join(ART, "scaler.joblib"))
    models = {}
    for name, info in meta["models"].items():
        path = os.path.join(ART, info["file"])
        if info["kind"] == "sklearn":
            models[name] = joblib.load(path)
        else:  # keras
            from tensorflow import keras
            models[name] = keras.models.load_model(path, compile=False)
    return meta, scaler, models


@st.cache_data
def load_sample():
    p = os.path.join(ART, "test_sample.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def score(model_name, X_scaled, meta, models):
    """Return a fraud score in [0, 1]-ish for each row (AE returns reconstruction error)."""
    info = meta["models"][model_name]
    m = models[model_name]
    if info["kind"] == "sklearn":
        return m.predict_proba(X_scaled)[:, 1]
    if info["type"] == "autoencoder":
        recon = m.predict(X_scaled, verbose=0)
        return np.mean(np.square(X_scaled - recon), axis=1)
    return m.predict(X_scaled, verbose=0).ravel()


def prepare(df, meta, scaler):
    feats = meta["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(meta["medians"]))
    return scaler.transform(X.values).astype("float32")


# ----------------------------------------------------------------------------- UI
try:
    meta, scaler, models = load_artifacts()
except Exception as e:
    st.error(f"Could not load artifacts from '{ART}/'. Run the Colab notebook first.\n\n{e}")
    st.stop()

sample = load_sample()
model_names = list(meta["models"].keys())
best = meta.get("best_model", model_names[0])

st.sidebar.title("💳 Fraud Detector")
model_name = st.sidebar.selectbox("Model", model_names, index=model_names.index(best))
info = meta["models"][model_name]
is_ae = info["type"] == "autoencoder"
default_thr = float(info["threshold"])
if is_ae:
    thr = st.sidebar.number_input("Anomaly threshold (reconstruction MSE)",
                                  value=default_thr, format="%.5f")
else:
    thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, default_thr, 0.01)
st.sidebar.caption(f"Tuned threshold (max F1 on validation): **{default_thr:.4f}**")
st.sidebar.caption(f"Best model on test PR-AUC: **{best}**")

st.title("Credit Card Fraud Detection")
st.caption(f"Dataset: {meta['dataset']} · {meta['n_rows']:,} transactions · "
           f"fraud rate {meta['fraud_rate']*100:.3f}%")

tab1, tab2, tab3 = st.tabs(["🔍 Single transaction", "📂 Batch prediction", "📊 Model performance"])

# ---- Tab 1: single transaction
with tab1:
    feats = meta["features"]
    mode = st.radio("Input", ["Pick from test set", "Enter manually"], horizontal=True)
    if mode == "Pick from test set" and sample is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            which = st.selectbox("Show", ["Any", "Fraud only", "Genuine only"])
            pool = sample
            if which == "Fraud only":
                pool = sample[sample[meta["target"]] == 1]
            elif which == "Genuine only":
                pool = sample[sample[meta["target"]] == 0]
            if st.button("🎲 Random transaction") or "row_idx" not in st.session_state \
                    or st.session_state.get("pool") != which:
                st.session_state.row_idx = int(np.random.choice(pool.index))
                st.session_state.pool = which
        row = sample.loc[[st.session_state.row_idx]]
        with c2:
            st.dataframe(row, width="stretch")
        true_label = int(row[meta["target"]].iloc[0])
    else:
        vals = {}
        cols = st.columns(5)
        for i, f in enumerate(feats):
            vals[f] = cols[i % 5].number_input(f, value=float(meta["medians"][f]), format="%.4f")
        row = pd.DataFrame([vals])
        true_label = None

    Xs = prepare(row, meta, scaler)
    s = float(score(model_name, Xs, meta, models)[0])
    pred = int(s >= thr)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "🚨 FRAUD" if pred else "✅ Genuine")
    c2.metric("Anomaly score" if is_ae else "Fraud probability", f"{s:.4f}")
    if true_label is not None:
        c3.metric("Actual label", "Fraud" if true_label else "Genuine",
                  "correct" if true_label == pred else "wrong",
                  delta_color="normal" if true_label == pred else "inverse")

    if not is_ae:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=s * 100,
                                     number={"suffix": "%"},
                                     gauge={"axis": {"range": [0, 100]},
                                            "bar": {"color": "crimson" if pred else "seagreen"},
                                            "threshold": {"line": {"color": "black", "width": 3},
                                                          "value": thr * 100}}))
        fig.update_layout(height=260, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("All models on this transaction")
    rows = []
    for n in model_names:
        sc = float(score(n, Xs, meta, models)[0])
        t = float(meta["models"][n]["threshold"])
        rows.append({"Model": n, "Score": round(sc, 5), "Threshold": round(t, 5),
                     "Prediction": "Fraud" if sc >= t else "Genuine"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ---- Tab 2: batch
with tab2:
    st.write(f"Upload a CSV with these columns: `{', '.join(meta['features'])}` "
             f"(a `{meta['target']}` column is optional – if present, metrics are computed).")
    up = st.file_uploader("CSV file", type="csv")
    if up is None and sample is not None and st.checkbox("Use bundled test sample instead"):
        up_df = sample.copy()
    elif up is not None:
        up_df = pd.read_csv(up)
    else:
        up_df = None

    if up_df is not None:
        try:
            Xs = prepare(up_df, meta, scaler)
            sc = score(model_name, Xs, meta, models)
            out = up_df.copy()
            out["fraud_score"] = sc
            out["prediction"] = (sc >= thr).astype(int)
            c1, c2 = st.columns(2)
            c1.metric("Transactions", f"{len(out):,}")
            c2.metric("Flagged as fraud", f"{out['prediction'].sum():,}")
            if meta["target"] in out.columns:
                from sklearn.metrics import (confusion_matrix, f1_score,
                                             precision_score, recall_score)
                y = out[meta["target"]].astype(int)
                p = out["prediction"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{precision_score(y, p, zero_division=0):.3f}")
                m2.metric("Recall", f"{recall_score(y, p, zero_division=0):.3f}")
                m3.metric("F1", f"{f1_score(y, p, zero_division=0):.3f}")
                cm = confusion_matrix(y, p, labels=[0, 1])
                st.plotly_chart(px.imshow(cm, text_auto=True, x=["Genuine", "Fraud"],
                                          y=["Genuine", "Fraud"], labels=dict(x="Predicted", y="Actual"),
                                          color_continuous_scale="Blues", height=350),
                                width="stretch")
            st.dataframe(out.sort_values("fraud_score", ascending=False).head(200),
                         width="stretch")
            st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                               "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.error(str(e))

# ---- Tab 3: performance
with tab3:
    mdf = pd.DataFrame(meta["test_metrics"]).T.reset_index().rename(columns={"index": "Model"})
    st.subheader("Test-set metrics (threshold tuned on validation set)")
    st.dataframe(mdf.style.highlight_max(subset=[c for c in mdf.columns if c != "Model"],
                                         color="#c6efce"),
                 width="stretch", hide_index=True)
    st.plotly_chart(px.bar(mdf.melt(id_vars="Model", value_vars=["PR_AUC", "ROC_AUC", "F1", "Recall", "Precision"]),
                           x="variable", y="value", color="Model", barmode="group",
                           labels={"variable": "Metric", "value": "Score"}, height=400),
                    width="stretch")

    c1, c2 = st.columns(2)
    curves = meta.get("curves", {})
    if curves:
        roc = go.Figure()
        pr = go.Figure()
        for n, cv in curves.items():
            roc.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], name=n, mode="lines"))
            pr.add_trace(go.Scatter(x=cv["rec"], y=cv["prec"], name=n, mode="lines"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False))
        roc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR", height=420)
        pr.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision", height=420)
        c1.plotly_chart(roc, width="stretch")
        c2.plotly_chart(pr, width="stretch")

    st.subheader(f"Confusion matrix – {model_name}")
    cm = np.array(meta["confusion"][model_name])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Genuine", "Fraud"], y=["Genuine", "Fraud"],
                              labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues",
                              height=380), width="stretch")

    if "feature_importance" in meta:
        fi = pd.Series(meta["feature_importance"]).sort_values().tail(15)
        st.subheader("Top 15 features (XGBoost gain)")
        st.plotly_chart(px.bar(fi, orientation="h", height=450, labels={"value": "Importance", "index": ""}),
                        width="stretch")
