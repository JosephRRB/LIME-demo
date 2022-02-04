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
    round_percent = round(model_test_score * 100, 2)
    return f"{round_percent} %"


def local_interpret_clf_model(
    model,
    X_train: pd.DataFrame,
    query: pd.DataFrame,
    code_container,
    explain_predicted=True,
):
    with code_container.expander("See LIME explanation code details:"):
        with st.echo():
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
    feature_importances: dict, figsize=(10, 5)
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fi_data = pd.DataFrame(*feature_importances.values()).sort_values(
        by=1, key=lambda x: abs(x)
    )
    class_name = list(feature_importances.keys())[0]
    positive_fi = fi_data.iloc[:, 1] > 0
    color = positive_fi.map({True: "blue", False: "red"})

    ax.barh(fi_data.iloc[:, 0], fi_data.iloc[:, 1], height=0.3, color=color)
    ax.set_xlim(-0.6, 0.6)
    ax.vlines(0, -0.25, 3.25, colors="k", linestyles="dashed", alpha=0.3)
    ax.set_xlabel("Feature Importance", fontsize=14)
    ax.set_title(f"Local Explanation for Class: {class_name}", fontsize=18)
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    st.image(buffer)


def deploy_plots_for_sklearn_models():
    st.header("Scikit-learn Models")
    st.markdown(
        """
    Now, we can consider explaining the predictions of different models from 
    scikit-learn. Namely, they are: 
    - [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),
    - [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html),
    - [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
    - and [Fully-Connected Neural Networks](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
    
    We will be using the [banknote authentication dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
    for binary classification. The dataset contains four continuous numerical 
    features extracted from the image of the banknotes and its 
    wavelet-transformed version. The class labels are also balanced and their
    meanings are clarified [here](https://jamesmccaffrey.wordpress.com/2020/08/18/in-the-banknote-authentication-dataset-class-0-is-genuine-authentic/).
    
    The classification models have already been fitted on a certain split of the 
    data, but can be refitted by clicking the button below. This will create a 
    new train and test data split and the models will be fitted accordingly.
    The new split and new models will then be used in the remainder of the demo.
    After clicking the refit button, there will also be an option to take a look
    at the basic training code used.  
    """
    )
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
        "Fully-Connected Neural Network Classifier",
    ]
    models = [svc, knn, rfc, mlp]
    for name, model in zip(model_names, models):
        st.sidebar.subheader(name)
        st.sidebar.metric(
            "Test Accuracy", calculate_test_score(model, X_test, y_test)
        )

    st.markdown(
        """
    Along the sidebar, the accuracy of the four models on the test set are 
    displayed.
    """
    )
    st.header("Explaining an instance")
    st.markdown(
        """
    We can now pick an instance from the test dataset 
    """
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
    code_container = st.empty()
    for name, model in zip(model_names, models):
        st.subheader(name)
        fi = local_interpret_clf_model(
            model,
            X_train,
            query,
            code_container,
            explain_predicted=explain_predicted,
        )
        plot_feature_importance_for_clf_model(
            fi,
            figsize=(10, 3),
        )
