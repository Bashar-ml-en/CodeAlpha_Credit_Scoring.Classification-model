import os
import streamlit as st     # type: ignore
import pandas as pd        # type: ignore
import numpy as np         # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression   # pyright: ignore[reportMissingModuleSource]
from sklearn.tree import DecisionTreeClassifier       # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestClassifier   # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt                       # pyright: ignore[reportMissingModuleSource]
import plotly.express as px                           # pyright: ignore[reportMissingImports, reportMissingModuleSource]


def main():
    st.title('Credit Score Dashboard 💳')
    st.sidebar.title('Credit Score Dashboard 💳')
    st.markdown('Are the Credit Scores classified as Good or Bad? 🪪')
    st.sidebar.markdown('Are the Credit Scores classified as Good or Bad? 🪪')

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    @st.cache_data(persist=True)
    def load_data():
        file_path = os.path.join(os.path.dirname(__file__), "german.csv")
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            st.stop()

        data = pd.read_csv(file_path)

        target_col = 'target'
        label = LabelEncoder()
        data[target_col] = label.fit_transform(data[target_col])

        cont_data = data.select_dtypes(include=['int', 'float']).drop(columns=[target_col], errors='ignore')
        cato_data = data.select_dtypes(include=['object'])

        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(cont_data), columns=cont_data.columns)

        if not cato_data.empty:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            data_encoded = pd.DataFrame(
                encoder.fit_transform(cato_data),
                columns=encoder.get_feature_names_out(cato_data.columns)
            )
            X_processed = pd.concat([data_scaled, data_encoded], axis=1)
        else:
            X_processed = data_scaled

        y = data[target_col]
        return X_processed, y

    @st.cache_data(persist=True)
    def split_data(X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def plot_metrics(metrics_list, model, X_test, y_test):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(plt)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(plt)

    # -----------------------------
    # Data load & split
    # -----------------------------
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('German Credit dataset (Classification) 📊')
        st.write(pd.concat([X, y], axis=1))

    class_labels = ['Good Credit', 'Bad Credit']
    st.sidebar.subheader('Choose Classifier')

    classifier = st.sidebar.selectbox(
        'Select Classifier',
        ('Logistic Regression', 'Decision Tree', 'Random Forest')
    )

    st.sidebar.subheader('Model Hyperparameters')

    if classifier == 'Logistic Regression':
        C = st.sidebar.number_input('C (Inverse of regularization strength)', 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider('Maximum number of iterations', 100, 500, key='max_iter_LR')
        model = LogisticRegression(C=C, max_iter=max_iter)

    elif classifier == 'Decision Tree':
        max_depth = st.sidebar.slider('Max Depth', 2, 15, key='max_depth_DT')
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, key='min_samples_split_DT')
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

    else:
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, key='n_estimators_RF')
        max_depth = st.sidebar.slider('Max Depth', 2, 15, key='max_depth_RF')
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    metrics = st.sidebar.multiselect(
        'Select Metrics to Plot',
        ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Model Comparison')
    )

    if st.sidebar.button('Classify', key='classify_button'):
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        label_map = {0: 'Bad Credit', 1: 'Good Credit'}
        pred_label = label_map.get(int(y_pred[0]), 'Unknown')
        st.write(f"### 🏷️ Predicted Class: **{pred_label}**")

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        st.subheader('Model Performance')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')

        plot_metrics(metrics, model, X_test, y_test)

        # -----------------------------
        # Model Comparison Section (Below metrics)
        # -----------------------------
        if 'Model Comparison' in metrics:
            st.markdown("---")
            st.subheader("📊 Model Comparison")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=300),
                "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=4),
                "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8)
            }

            results = []

            for name, clf in models.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = clf.score(X_test, y_test)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = 2 * (prec * rec) / (prec + rec)
                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-score": f1
                })

            results_df = pd.DataFrame(results)

            st.dataframe(results_df.style.format({
                "Accuracy": "{:.2f}",
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-score": "{:.2f}"
            }))

            st.write("### Interactive Performance Comparison")
            results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(
                results_melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                title="Model Performance Comparison (Interactive)"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(yaxis=dict(title="Score", range=[0, 1.1]), legend_title_text="Metrics")
            st.plotly_chart(fig, use_container_width=True)

            # Explanation below visualization
            st.markdown("""
            #### 💡 Importance of Model Performance Evaluation
            Evaluating models helps ensure the best one is chosen for credit scoring:
            - **Accuracy**: Measures overall correctness  
            - **Precision**: Reduces risky loans being approved  
            - **Recall**: Ensures good clients aren’t rejected  
            - **F1-score**: Balances both metrics  
            For **Malaysian financial banks**, these models strengthen **credit risk analysis**, improve **loan decision-making**, and support **compliance** with Bank Negara Malaysia’s credit policies.
            """)


if __name__ == "__main__":
    main()
