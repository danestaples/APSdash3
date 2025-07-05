import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

import io

st.set_page_config(page_title="APS Dashboard", layout="wide")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("APS Dataset.csv")
    return df

df = load_data()

# ========== SIDEBAR ==========
st.sidebar.header("Filters")
class_options = st.sidebar.multiselect("Class", options=df["Class"].unique(), default=df["Class"].unique())
gender_options = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
travel_type = st.sidebar.multiselect("Type of Travel", options=df["Type of Travel"].unique(), default=df["Type of Travel"].unique())

filtered_df = df[
    (df["Class"].isin(class_options)) &
    (df["Gender"].isin(gender_options)) &
    (df["Type of Travel"].isin(travel_type))
]

uploaded_file = st.sidebar.file_uploader("Upload new data for prediction (CSV, no target column)", type=["csv"])
new_data = pd.read_csv(uploaded_file) if uploaded_file else None

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"]
)

# ========== TAB 1: DATA VISUALIZATION ==========
with tab1:
    st.header("1. Data Visualization")
    st.markdown("This tab presents a suite of interactive visualizations for macro and micro understanding of the APS dataset. All charts and tables include brief insights above each visualization.")

    service_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness"
    ]

    st.markdown("**1.1 Distribution of Passenger Satisfaction**: This pie chart shows the proportion of satisfied vs dissatisfied passengers.")
    fig = px.pie(filtered_df, names='satisfaction', title="Passenger Satisfaction Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.2 Satisfaction across Travel Class**: Compare satisfaction rates by travel class.")
    fig = px.histogram(filtered_df, x="Class", color="satisfaction", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.3 Age distribution by Satisfaction**: Analyze how age groups relate to satisfaction.")
    fig = px.box(filtered_df, x="satisfaction", y="Age", color="satisfaction")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.4 Flight Distance distribution by Satisfaction**: Longer flights may correlate with experience.")
    fig = px.histogram(filtered_df, x="Flight Distance", color="satisfaction", nbins=40)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.5 Satisfaction by Gender**: Review gender-based trends in satisfaction.")
    fig = px.histogram(filtered_df, x="Gender", color="satisfaction", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.6 Satisfaction by Customer Type**: Are loyal customers happier?")
    fig = px.histogram(filtered_df, x="Customer Type", color="satisfaction", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.7 Satisfaction by Type of Travel**: Differentiate between business and personal travel experiences.")
    fig = px.histogram(filtered_df, x="Type of Travel", color="satisfaction", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.8 Average Service Ratings (Heatmap)**: Visualize which service areas excel or need improvement.")
    avg_ratings = filtered_df.groupby("satisfaction")[service_cols].mean()
    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(avg_ratings, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig, use_container_width=True)

    st.markdown("**1.9 Delay Analysis by Satisfaction**: Compare departure/arrival delays among satisfied and dissatisfied passengers.")
    fig = px.box(filtered_df, x="satisfaction", y="Departure Delay in Minutes", color="satisfaction")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(filtered_df, x="satisfaction", y="Arrival Delay in Minutes", color="satisfaction")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1.10 Age vs Flight Distance by Satisfaction**: Uncover joint trends among key variables.")
    fig = px.scatter(filtered_df, x="Age", y="Flight Distance", color="satisfaction", size="Flight Distance", hover_data=["Class"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Download the filtered data:**")
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='filtered_APS_data.csv', mime='text/csv')

# ========== TAB 2: CLASSIFICATION ==========
with tab2:
    st.header("2. Classification: Predict Passenger Satisfaction")
    st.markdown("""
    Compare KNN, Decision Tree, Random Forest, and Gradient Boosted Trees for passenger satisfaction prediction.
    Explore metrics, confusion matrices, ROC curves, and batch prediction on new data.
    """)

    df_clf = df.copy()
    for col in ['id', 'Unnamed: 0']:
        if col in df_clf.columns:
            df_clf = df_clf.drop(col, axis=1)
    df_clf = df_clf.dropna(subset=["satisfaction"])
    df_clf["satisfaction"] = df_clf["satisfaction"].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    X = pd.get_dummies(df_clf.drop("satisfaction", axis=1), drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_clf["satisfaction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42)
    }
    results = {}
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        results[name] = {
            "Train Acc": acc_train, "Test Acc": acc_test,
            "Precision": precision, "Recall": recall, "F1": f1,
            "y_pred_test": y_pred_test, "model": model
        }
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc(fpr, tpr))

    st.markdown("**2.1 Model Metrics Table**: See accuracy, precision, recall, F1 for all algorithms.")
    metrics_df = pd.DataFrame({
        model: {
            "Train Acc": f"{results[model]['Train Acc']:.3f}",
            "Test Acc": f"{results[model]['Test Acc']:.3f}",
            "Precision": f"{results[model]['Precision']:.3f}",
            "Recall": f"{results[model]['Recall']:.3f}",
            "F1-score": f"{results[model]['F1']:.3f}"
        } for model in results
    }).T
    st.dataframe(metrics_df)

    st.markdown("**2.2 Confusion Matrix**: Select model to display the confusion matrix (labels: 0=Dissatisfied, 1=Satisfied).")
    conf_model = st.selectbox("Choose model for confusion matrix:", list(results.keys()))
    conf_matrix = confusion_matrix(y_test, results[conf_model]["y_pred_test"])
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Dissatisfied", "Satisfied"], yticklabels=["Dissatisfied", "Satisfied"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {conf_model}")
    st.pyplot(fig, use_container_width=True)

    st.markdown("**2.3 ROC Curve (All Models)**: Visual comparison of model performance (AUC in legend).")
    fig = go.Figure()
    colors = ["royalblue", "orangered", "green", "purple"]
    for i, (name, (fpr, tpr, aucv)) in enumerate(roc_curves.items()):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={aucv:.2f})", line=dict(color=colors[i])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
    fig.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**2.4 Upload New Data (No Target Column) for Prediction**")
    if new_data is not None:
        try:
            predict_X = pd.get_dummies(new_data, drop_first=True)
            for col in X.columns:
                if col not in predict_X.columns:
                    predict_X[col] = 0
            predict_X = predict_X[X.columns]
            predict_X = predict_X.replace([np.inf, -np.inf], np.nan).fillna(0)
            selected_pred_model = st.selectbox("Choose model for new predictions:", list(models.keys()), key="predmodel")
            model_to_use = results[selected_pred_model]["model"]
            preds = model_to_use.predict(predict_X)
            result_df = new_data.copy()
            result_df["Predicted Satisfaction"] = preds
            result_df["Predicted Satisfaction"] = result_df["Predicted Satisfaction"].map({1:"satisfied", 0:"neutral or dissatisfied"})
            st.write(result_df.head())
            csv = result_df.to_csv(index=False)
            st.download_button(label="Download Prediction Results", data=csv, file_name="APS_predictions.csv", mime='text/csv')
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ========== TAB 3: CLUSTERING ==========
with tab3:
    st.header("3. Clustering & Segmentation")
    st.markdown("Apply K-Means clustering to identify passenger personas and download labeled data.")

    cluster_cols = ["Age", "Flight Distance"] + service_cols
    cluster_data = filtered_df[cluster_cols].dropna()
    cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)
    k = st.slider("Number of Clusters", 2, 10, 4)
    # Elbow plot
    inertia = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(cluster_scaled)
        inertia.append(km.inertia_)
    st.markdown("**3.1 Elbow Chart**: Find the optimal number of clusters for KMeans.")
    fig = px.line(x=list(range(2, 11)), y=inertia, labels={'x':'Number of Clusters', 'y':'Inertia'})
    st.plotly_chart(fig, use_container_width=True)
    # KMeans and persona
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(cluster_scaled)
    persona_df = cluster_data.copy()
    persona_df["Cluster"] = clusters
    st.markdown("**3.2 Cluster Personas Table**: Mean feature values by cluster segment.")
    st.write(persona_df.groupby("Cluster").mean())
    # Download cluster labeled data
    persona_out = filtered_df.copy()
    persona_out = persona_out.loc[persona_df.index].copy()
    persona_out["Cluster"] = clusters
    csv = persona_out.to_csv(index=False)
    st.download_button(label="Download Data with Cluster Labels", data=csv, file_name='APS_with_clusters.csv', mime='text/csv')

# ========== TAB 4: ASSOCIATION RULE MINING ==========
with tab4:
    st.header("4. Association Rule Mining")
    st.markdown("Discover rules among service ratings. Adjust support/confidence and analyze clusters separately if desired.")
    cols_apriori = st.multiselect("Select columns for Apriori (at least 2):", options=service_cols, default=service_cols[:3])
    min_sup = st.slider("Min Support", 1, 20, 5)/100
    min_conf = st.slider("Min Confidence", 1, 100, 60)/100
    cluster_toggle = st.checkbox("Analyze by Cluster?", value=False)
    apriori_df = filtered_df.copy()
    if cluster_toggle:
        num_clusters = st.slider("Clusters for Rules", 2, 6, 3, key="apriori_nclust")
        cluster_data_tmp = apriori_df[cluster_cols].dropna()
        cluster_data_tmp = cluster_data_tmp.replace([np.inf, -np.inf], np.nan).fillna(0)
        kmeans_tmp = KMeans(n_clusters=num_clusters, random_state=42)
        apriori_scaled = scaler.fit_transform(cluster_data_tmp)
        cluster_labels = kmeans_tmp.fit_predict(apriori_scaled)
        apriori_df = apriori_df.iloc[cluster_data_tmp.index].copy()
        apriori_df["Cluster"] = cluster_labels
        sel_cluster = st.selectbox("Select Cluster to View Rules", options=range(num_clusters))
        apriori_df = apriori_df[apriori_df["Cluster"]==sel_cluster]
    apriori_bin = apriori_df[cols_apriori].apply(lambda x: x>=4).astype(int)
    frequent_items = apriori(apriori_bin, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_conf)
    st.markdown("**4.1 Top-10 Association Rules by Confidence**")
    if not rules.empty:
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
    else:
        st.info("No rules found with current settings.")

# ========== TAB 5: REGRESSION ==========
with tab5:
    st.header("5. Regression Insights")
    st.markdown("Linear, Ridge, Lasso, and Decision Tree regressors: Explore patterns and predictions for continuous outcomes like delays and service scores.")

    df_reg = df.copy()
    df_reg = df_reg.dropna(subset=["Departure Delay in Minutes"])
    reg_targets = [
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        "Inflight wifi service",
        "Seat comfort",
        "Food and drink"
    ]
    reg_feature = st.selectbox("Select Target for Regression:", reg_targets)
    feature_cols = ["Age", "Flight Distance"] + service_cols
    Xreg = df_reg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    yreg = df_reg[reg_feature]
    Xreg_train, Xreg_test, yreg_train, yreg_test = train_test_split(Xreg, yreg, random_state=42)

    regr_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    reg_results = {}
    for name, regr in regr_models.items():
        regr.fit(Xreg_train, yreg_train)
        y_pred = regr.predict(Xreg_test)
        score = regr.score(Xreg_test, yreg_test)
        reg_results[name] = (score, yreg_test, y_pred)
    st.markdown("**5.1 Regression Scores (R^2) on Test Data**")
    reg_scores = pd.DataFrame({k:[v[0]] for k,v in reg_results.items()}, index=["R^2 Score"]).T
    st.dataframe(reg_scores)

    st.markdown("**5.2 Regression Predictions vs Actuals**")
    fig = go.Figure()
    for name, (score, y_true, y_pred) in reg_results.items():
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name=name))
    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", title="Actual vs Predicted Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**5.3 Decision Tree Feature Importances**")
    importances = regr_models["Decision Tree"].feature_importances_
    feat_imp = pd.Series(importances, index=Xreg.columns).sort_values(ascending=False)[:10]
    fig = px.bar(feat_imp, orientation="h", title="Top 10 Feature Importances (Decision Tree)")
    st.plotly_chart(fig, use_container_width=True)
