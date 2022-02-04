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
                discretizer='quartile',
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
    ax.set_xlim(-0.5, 0.5)
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
    wavelet-transformed version (WTI). The class labels are also balanced and 
    their meanings are clarified [here](https://jamesmccaffrey.wordpress.com/2020/08/18/in-the-banknote-authentication-dataset-class-0-is-genuine-authentic/).
    
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
    We can now pick an instance from the test dataset as shown in the table 
    below. Select a row from the table by selecting the row index in the 
    selection box below the table.
    """
    )
    st.dataframe(X_test)
    query_idx = st.number_input(
        "Select a row index:", min_value=0, max_value=len(X_test)
    )
    query = X_test.loc[[query_idx], :]
    st.write(
        """
    This selected row will then be the data instance whose model predictions 
    we wish to be explained:
    """
    )
    st.table(query)
    st.write(
        """
    Since our models are binary classification models, we have the option to
    see the explanations for the predicted class or the non-predicted class.
    By default, we show the explanations for the predicted class. The checkbox
    below can be unticked to see the non-predicted class. The predicted class 
    should refer to the class with the highest predicted probability. Note 
    however that the sklearn `SVC` implementation has some 
    [issues](https://scikit-learn.org/stable/modules/svm.html#scores-probabilities) 
    regarding its probabilities. 
    
    We can also see a code snippet on how to use the `LimeTabularExplainer()` 
    class. It mainly needs the training dataset, the data instance we wish to 
    explain, and the model prediction function (which in our case is the 
    `predict_proba()` method of our models). We also have the option of 
    discretizing our continuous numerical features for the explanation (which 
    we set to be based on the feature quartiles according to the training 
    dataset).
    """
    )
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
    st.markdown(
        """
    The plots above show how much impact the features have on the model 
    predictions for the selected data instance. Specifically in our setup, the 
    presence of the feature in the particular quartile translates into the
    increase or decrease of the model's prediction probabilities. The magnitude
    of their corresponding feature importances refer to how much the prediction
    probability increased or decreased. The features in the figures are sorted
    with respect to the feature importance magnitude or how impactful they were 
    on the model's prediction. That is, the most impactful feature is at the top
    while the least impactful is at the bottom. Additionally, since we are doing 
    a binary classification task, we can observe that the explanations for the 
    predicted class are opposite to those for the non-predicted class.
    
    We can see that even if the models were trained on the same dataset and 
    makes the same prediction for the same instance, the explanations estimated
    by LIME are not necessarily the same. The most readily seen is the 
    difference in ranking of the most impactful features. This suggests that 
    the models may be using the features differently. On a technical note, since 
    LIME has some randomness in its process, it would be beneficial to see the 
    average effect on a variety of trials and datasets.
    """
    )
