from pathlib import Path
from io import BytesIO


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from joblib import load
from lime.lime_tabular import LimeTabularExplainer

from fit_sklearn_models import fit_models


def calculate_test_score(model, X_test, y_test):
    model_test_score = model.score(X_test, y_test)
    round_percent = round(model_test_score * 100, 1)
    return f"{round_percent} %"


def local_interpret_clf_model(
    model,
    X_train: pd.DataFrame,
    query: pd.DataFrame,
    explain_predicted=True,
):
    pred_proba_fn = model.predict_proba
    explainer = LimeTabularExplainer(
        X_train.values,
        mode="classification",
        feature_names=X_train.columns,
        discretize_continuous=True,
        sample_around_instance=True,
        class_names=model.classes_,
    )
    probas = pred_proba_fn(query)
    if explain_predicted:
        label = np.argmax(probas)
    else:
        label = np.argmin(probas)
    exp = explainer.explain_instance(
        query.values.ravel(), pred_proba_fn, labels=[label]
    )
    fi = {exp.class_names[label]: exp.as_list(label=label)}
    # fi_human_readable = {
    #     exp.class_names[l]: [
    #         (exp.domain_mapper.discretized_feature_names[f], fi)
    #         for f, fi in f_fi
    #     ]
    #     for l, f_fi in exp.as_map().items()
    # }
    return fi


def plot_feature_importance_for_clf_model(
    model, X_train, query, explain_predicted=True, figsize=(10, 5)
):
    fi = local_interpret_clf_model(
        model, X_train, query, explain_predicted=explain_predicted
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fi_data = pd.DataFrame(*fi.values())
    positive_fi = fi_data.iloc[:, 1] > 0
    color = positive_fi.map({True: "blue", False: "red"})

    ax.barh(fi_data.iloc[:, 0], fi_data.iloc[:, 1], height=0.3, color=color)
    ax.set_xlim(-0.6, 0.6)
    ax.vlines(0, -0.25, 3.25, colors="k", linestyles="dashed", alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=14)
    ax.set_title(f"Class: {list(fi.keys())[0]}", fontsize=18)
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    st.image(buffer)


if __name__ == "__main__":
    # https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    # https://jamesmccaffrey.wordpress.com/2020/08/18/in-the-banknote-authentication-dataset-class-0-is-genuine-authentic/
    data_dir = Path(__file__).parents[1] / "demos/data"

    if st.button("Refit models on new data split"):
        fit_models()
        st.write("Refitted!")

    df_train = pd.read_csv(data_dir / "train_banknote_authentication.csv")
    df_test = pd.read_csv(data_dir / "test_banknote_authentication.csv")

    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    svc = load(data_dir / "fitted_svc.joblib")
    knn = load(data_dir / "fitted_knn.joblib")
    rfc = load(data_dir / "fitted_rfc.joblib")
    mlp = load(data_dir / "fitted_mlp.joblib")

    model_names = [
        "Support Vector Machine Classifier",
        "K-Nearest Neighbor Classifier",
        "Random Forest Classifier",
        "Fully-Connected Neural Network Classifier"
    ]
    models = [svc, knn, rfc, mlp]
    for name, model in zip(model_names, models):
        st.sidebar.subheader(name)
        st.sidebar.metric(
            "Test Accuracy", calculate_test_score(model, X_test, y_test)
        )

    st.dataframe(X_test)
    query_idx = st.number_input(
        "Select a row index:", min_value=0, max_value=len(X_test)
    )
    query = X_test.loc[[query_idx], :]
    st.write("Data row to be explained:")
    st.table(query)

    explain_predicted = st.checkbox(
        "Explain model's predicted class", value=True
    )
    for name, model in zip(model_names, models):
        st.subheader(name)
        plot_feature_importance_for_clf_model(
            model,
            X_train,
            query,
            explain_predicted=explain_predicted,
            figsize=(10, 3),
        )
