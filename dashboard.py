import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    adjusted_rand_score, silhouette_score,
    precision_score, recall_score, f1_score,
)

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_B       = "#1D9E75"
COLOR_M       = "#E24B4A"
COLOR_PRIMARY = "#378ADD"
COLOR_ACCENT  = "#7F77DD"
COLORS_MOD    = [COLOR_PRIMARY, COLOR_ACCENT, "#EF9F27", COLOR_M]
NOMBRES       = ["SVM", "Red Neuronal", "Árbol de Decisión", "K-Means"]

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important; }
div[data-testid="stMetric"] {
    background: #1e2130; border: 1px solid #2e3250;
    border-radius: 10px; padding: 1rem;
}
.sec-title {
    font-size: 1rem; font-weight: 600; color: #378ADD;
    border-left: 4px solid #378ADD;
    padding-left: 10px; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos():
    df = pd.read_csv("cancer.csv")
    df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")

    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])   # B=0, M=1

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, feature_names, df

X_train, X_test, y_train, y_test, feature_names, df = cargar_datos()

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def entrenar_svm():
    m = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    return {
        "model": m, "name": "SVM (RBF)",
        "accuracy": accuracy_score(y_test, yp),
        "report": classification_report(y_test, yp, target_names=["Benigno", "Maligno"]),
        "cm": confusion_matrix(y_test, yp),
        "y_pred": yp, "y_test": np.array(y_test),
    }

@st.cache_data
def entrenar_nn():
    m = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                      max_iter=300, random_state=42)
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    return {
        "model": m, "name": "Red Neuronal (MLP)",
        "accuracy": accuracy_score(y_test, yp),
        "report": classification_report(y_test, yp, target_names=["Benigno", "Maligno"]),
        "cm": confusion_matrix(y_test, yp),
        "y_pred": yp, "y_test": np.array(y_test),
        "loss_curve": m.loss_curve_,
    }

@st.cache_data
def entrenar_dt():
    m = DecisionTreeClassifier(max_depth=5, random_state=42)
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    return {
        "model": m, "name": "Árbol de Decisión",
        "accuracy": accuracy_score(y_test, yp),
        "report": classification_report(y_test, yp, target_names=["Benigno", "Maligno"]),
        "cm": confusion_matrix(y_test, yp),
        "y_pred": yp, "y_test": np.array(y_test),
        "importances": m.feature_importances_,
    }

@st.cache_data
def entrenar_kmeans():
    m = KMeans(n_clusters=2, random_state=42, n_init=10)
    m.fit(X_train)
    labels_train = m.labels_
    labels_test  = m.predict(X_test)

    mapping = {}
    for c in [0, 1]:
        mask = labels_train == c
        if mask.sum() > 0:
            mapping[c] = mode(np.array(y_train)[mask], keepdims=True).mode[0]

    yp = np.array([mapping[l] for l in labels_test])
    yt = np.array(y_test)

    return {
        "model": m, "name": "K-Means (k=2)",
        "accuracy": accuracy_score(yt, yp),
        "ari": adjusted_rand_score(yt, labels_test),
        "silhouette": silhouette_score(X_test, labels_test),
        "report": classification_report(yt, yp, target_names=["Benigno", "Maligno"]),
        "cm": confusion_matrix(yt, yp),
        "y_pred": yp, "y_test": yt,
    }

@st.cache_data
def entrenar_todos():
    return {
        "SVM": entrenar_svm(),
        "Red Neuronal": entrenar_nn(),
        "Árbol de Decisión": entrenar_dt(),
        "K-Means": entrenar_kmeans(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fig_dark(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#9ca3af")
    ax.xaxis.label.set_color("#9ca3af")
    ax.yaxis.label.set_color("#9ca3af")
    ax.title.set_color("#e5e7eb")
    for s in ax.spines.values():
        s.set_edgecolor("#2e3250")
    return fig, ax

def sec(title):
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)

def plot_confusion(cm):
    fig, ax = fig_dark(5, 4)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"],
                ax=ax, linewidths=0.5, linecolor="#0e1117",
                annot_kws={"color": "white", "size": 13})
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.tight_layout(); st.pyplot(fig); plt.close()

def plot_pca(y_pred):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    fig, ax = fig_dark(5, 4)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="RdYlGn",
               alpha=0.75, edgecolors="#0e1117", linewidths=0.4, s=55)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(handles=[mpatches.Patch(color=COLOR_B, label="Benigno"),
                       mpatches.Patch(color=COLOR_M, label="Maligno")],
              facecolor="#1a1d2e", labelcolor="#e5e7eb", edgecolor="#2e3250")
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Cancer ML")
    st.caption("Wisconsin Breast Cancer Dataset")
    st.divider()
    pagina = st.radio("", ["Resumen", "Por modelo", "Comparativa", "Exploración"],
                      label_visibility="collapsed")
    st.divider()
    st.markdown(f"Muestras: **569**")
    st.markdown(f"Train / Test: **{len(y_train)} / {len(y_test)}**")
    st.markdown(f"Features: **{len(feature_names)}**")
    st.markdown(f"Clases: **Benigno · Maligno**")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Resumen":
    st.title("Clasificación de Cáncer de Mama")
    st.caption("Comparativa de modelos de Machine Learning — Wisconsin Breast Cancer")
    st.divider()

    with st.spinner("Entrenando los 4 modelos..."):
        todos = entrenar_todos()

    c1, c2, c3, c4 = st.columns(4)
    iconos = ["⚡", "🧠", "🌳", "🔵"]
    for col, nombre, ico in zip([c1, c2, c3, c4], NOMBRES, iconos):
        col.metric(f"{ico} {nombre}", f"{todos[nombre]['accuracy']:.2%}")

    st.divider()
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        sec("Accuracy por modelo")
        fig, ax = fig_dark(7, 3.5)
        accs = [todos[n]["accuracy"] for n in NOMBRES]
        bars = ax.barh(NOMBRES, accs, color=COLORS_MOD, height=0.5)
        ax.set_xlim(0.78, 1.01)
        ax.set_xlabel("Accuracy")
        ax.axvline(0.95, color="#2e3250", linestyle="--", lw=1)
        for bar, val in zip(bars, accs):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2%}", va="center", color="#e5e7eb", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        sec("Distribución del dataset")
        fig2, ax2 = fig_dark(5, 3.5)
        counts = df["diagnosis"].value_counts()
        wedges, texts, autos = ax2.pie(
            counts, labels=["Benigno", "Maligno"],
            autopct="%1.1f%%", colors=[COLOR_B, COLOR_M],
            startangle=90, wedgeprops={"edgecolor": "#0e1117", "linewidth": 2},
        )
        for t in texts: t.set_color("#e5e7eb")
        for a in autos: a.set_color("white"); a.set_fontsize(11)
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    sec("Preprocesamiento aplicado")
    p1, p2, p3, p4 = st.columns(4)
    p1.info("**Valores nulos:** 0\nDataset limpio sin imputación.")
    p2.info("**Columnas eliminadas:**\n`id` y `Unnamed: 32`")
    p3.info("**Encoding:** LabelEncoder\nB → 0 · M → 1")
    p4.info("**Escalado:** StandardScaler\nmedia=0 · desv=1")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — POR MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Por modelo":
    st.title("Análisis por modelo")
    st.divider()

    modelo_sel = st.selectbox("Selecciona un modelo", NOMBRES)

    fns = {
        "SVM": entrenar_svm,
        "Red Neuronal": entrenar_nn,
        "Árbol de Decisión": entrenar_dt,
        "K-Means": entrenar_kmeans,
    }
    with st.spinner(f"Cargando {modelo_sel}..."):
        r = fns[modelo_sel]()

    sec(r["name"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{r['accuracy']:.2%}")
    if "ari" in r:
        m2.metric("Adjusted Rand Index", f"{r['ari']:.4f}")
        m3.metric("Silhouette Score", f"{r['silhouette']:.4f}")
    else:
        m2.metric("Train", len(y_train))
        m3.metric("Test", len(y_test))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        sec("Matriz de confusión")
        plot_confusion(r["cm"])
    with col2:
        sec("PCA — proyección 2D")
        plot_pca(r["y_pred"])

    st.divider()
    sec("Reporte de clasificación")
    st.code(r["report"], language=None)

    # Extras
    if modelo_sel == "Árbol de Decisión":
        st.divider()
        sec("Top 10 features más importantes")
        imp = r["importances"]
        idx = np.argsort(imp)[::-1][:10]
        fig3, ax3 = fig_dark(9, 4)
        bars = ax3.bar(range(10), imp[idx],
                       color=[COLOR_PRIMARY if i == 0 else "#2E6FB5" for i in range(10)])
        ax3.set_xticks(range(10))
        ax3.set_xticklabels([feature_names[i] for i in idx],
                            rotation=38, ha="right", fontsize=9, color="#9ca3af")
        ax3.set_ylabel("Importancia")
        for b in bars[:3]:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                     f"{b.get_height():.3f}", ha="center", va="bottom",
                     color="#e5e7eb", fontsize=9)
        fig3.tight_layout(); st.pyplot(fig3); plt.close()

    if modelo_sel == "Red Neuronal":
        st.divider()
        sec("Curva de pérdida (loss)")
        fig4, ax4 = fig_dark(9, 3)
        lc = r["loss_curve"]
        ax4.plot(lc, color=COLOR_PRIMARY, lw=2)
        ax4.fill_between(range(len(lc)), lc, alpha=0.15, color=COLOR_PRIMARY)
        ax4.set_xlabel("Iteración"); ax4.set_ylabel("Loss")
        fig4.tight_layout(); st.pyplot(fig4); plt.close()

    if modelo_sel == "K-Means":
        st.divider()
        sec("Método del codo")
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train).inertia_
                    for k in range(1, 9)]
        fig5, ax5 = fig_dark(7, 3)
        ax5.plot(range(1, 9), inertias, marker="o", color=COLOR_PRIMARY, lw=2,
                 markersize=7, markerfacecolor=COLOR_ACCENT)
        ax5.axvline(2, color=COLOR_M, linestyle="--", lw=1.2, alpha=0.7)
        ax5.set_xlabel("Número de clusters (k)"); ax5.set_ylabel("Inercia")
        fig5.tight_layout(); st.pyplot(fig5); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Comparativa":
    st.title("Comparativa de modelos")
    st.divider()

    with st.spinner("Entrenando todos los modelos..."):
        todos = entrenar_todos()

    sec("Métricas generales")
    filas = []
    for nombre in NOMBRES:
        r = todos[nombre]
        yt = r["y_test"]; yp = r["y_pred"]
        filas.append({
            "Modelo": nombre,
            "Tipo": "No supervisado" if nombre == "K-Means" else "Supervisado",
            "Accuracy":  f"{r['accuracy']:.2%}",
            "Precisión": f"{precision_score(yt, yp, zero_division=0):.2%}",
            "Recall":    f"{recall_score(yt, yp, zero_division=0):.2%}",
            "F1-Score":  f"{f1_score(yt, yp, zero_division=0):.2%}",
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        sec("Accuracy")
        fig, ax = fig_dark(6, 4)
        accs = [todos[n]["accuracy"] for n in NOMBRES]
        bars = ax.bar(NOMBRES, accs, color=COLORS_MOD, width=0.5)
        ax.set_ylim(0.78, 1.0); ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=15)
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{v:.2%}", ha="center", va="bottom", color="#e5e7eb", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("F1-Score — Benigno vs Maligno")
        f1_b, f1_m = [], []
        for nombre in NOMBRES:
            r = todos[nombre]
            rep = classification_report(r["y_test"], r["y_pred"],
                                        output_dict=True, zero_division=0)
            f1_b.append(rep["0"]["f1-score"])
            f1_m.append(rep["1"]["f1-score"])
        x = np.arange(len(NOMBRES)); w = 0.35
        fig2, ax2 = fig_dark(6, 4)
        ax2.bar(x - w/2, f1_b, w, label="Benigno", color=COLOR_B)
        ax2.bar(x + w/2, f1_m, w, label="Maligno", color=COLOR_M)
        ax2.set_ylim(0.7, 1.0); ax2.set_ylabel("F1-Score")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["SVM", "RN", "DT", "KM"], color="#9ca3af")
        ax2.legend(facecolor="#1a1d2e", labelcolor="#e5e7eb", edgecolor="#2e3250")
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    sec("Matrices de confusión — todos los modelos")
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 3.5))
    fig3.patch.set_facecolor("#0e1117")
    for ax, nombre in zip(axes3, NOMBRES):
        ax.set_facecolor("#1a1d2e")
        sns.heatmap(todos[nombre]["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["B", "M"], yticklabels=["B", "M"], ax=ax,
                    linewidths=0.5, linecolor="#0e1117",
                    annot_kws={"color": "white"})
        ax.set_title(nombre, color="#e5e7eb", fontsize=10, pad=8)
        ax.tick_params(colors="#9ca3af")
        ax.set_xlabel("Predicho", color="#9ca3af", fontsize=9)
        ax.set_ylabel("Real", color="#9ca3af", fontsize=9)
    fig3.tight_layout(); st.pyplot(fig3); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Exploración":
    st.title("Exploración del dataset")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Datos crudos"])

    with tab1:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            feat = st.selectbox("Feature", feature_names)
            st.dataframe(
                df.groupby("diagnosis")[feat].describe().round(3)
                  .rename(index={0: "Benigno", 1: "Maligno"}),
                use_container_width=True,
            )
        with col_b:
            fig, ax = fig_dark(7, 4)
            for cls, label, color in [(0, "Benigno", COLOR_B), (1, "Maligno", COLOR_M)]:
                ax.hist(df[df["diagnosis"] == cls][feat], bins=28,
                        alpha=0.7, label=label, color=color, edgecolor="#0e1117")
            ax.set_xlabel(feat); ax.set_ylabel("Frecuencia")
            ax.legend(facecolor="#1a1d2e", labelcolor="#e5e7eb", edgecolor="#2e3250")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        sec("Boxplot ")
        top_f = feature_names[:29]
        df_melt = df[top_f + ["diagnosis"]].melt(
            id_vars="diagnosis", var_name="feature", value_name="valor")
        df_melt["clase"] = df_melt["diagnosis"].map({0: "Benigno", 1: "Maligno"})
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        fig2.patch.set_facecolor("#0e1117"); ax2.set_facecolor("#1a1d2e")
        sns.boxplot(data=df_melt, x="feature", y="valor", hue="clase",
                    palette={"Benigno": COLOR_B, "Maligno": COLOR_M},
                    ax=ax2, linewidth=0.8, fliersize=2)
        ax2.tick_params(axis="x", rotation=30, colors="#9ca3af")
        ax2.tick_params(axis="y", colors="#9ca3af")
        ax2.set_xlabel(""); ax2.set_ylabel("Valor", color="#9ca3af")
        ax2.legend(facecolor="#1a1d2e", labelcolor="#e5e7eb", edgecolor="#2e3250")
        for s in ax2.spines.values(): s.set_edgecolor("#2e3250")
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    with tab2:
        n_feats = st.slider("Número de features", 5, 30, 15, 5)
        corr = df[feature_names[:n_feats]].corr()
        fig3, ax3 = plt.subplots(figsize=(12, 9))
        fig3.patch.set_facecolor("#0e1117"); ax3.set_facecolor("#1a1d2e")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=n_feats <= 10, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax3,
                    linewidths=0.3, linecolor="#0e1117",
                    annot_kws={"size": 8})
        ax3.tick_params(colors="#9ca3af", labelsize=8)
        fig3.tight_layout(); st.pyplot(fig3); plt.close()

    with tab3:
        df_view = df.copy()
        df_view["diagnosis"] = df_view["diagnosis"].map({0: "Benigno", 1: "Maligno"})
        st.dataframe(df_view, use_container_width=True, height=420)
        st.caption(f"{len(df_view)} filas · {len(df_view.columns)} columnas")
