from functools import partial
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st
from lime.lime_tabular import LimeTabularExplainer


def fitted_model_prediction(
    array: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    x = array[:, 0]
    y = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return y


def local_approximation(
    array: np.ndarray, approx_at: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    x = array[:, 0]
    x0 = approx_at[:, 0]
    m = -x0 * np.exp(-(x0 ** 2) / (2 * sigma ** 2)) / (sigma ** 2)
    b = fitted_model_prediction(approx_at, sigma)

    y = m[0] * (x - x0[0]) + b[0]
    return y


def plot_model_and_local_preds(
    train_df: pd.DataFrame,
    instance_to_be_explained: pd.DataFrame,
    sigma: float = 1.0,
    figsize=(10, 5),
):
    x = train_df.values[:, 0]
    y_pred = fitted_model_prediction(train_df.values, sigma=sigma)

    local_data = train_df.values / 10 + instance_to_be_explained.values
    y_approx = local_approximation(
        local_data,
        approx_at=instance_to_be_explained.values,
        sigma=sigma,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.scatter(x=x, y=y_pred, marker=".", label="Model predictions")
    ax.scatter(
        x=local_data[:, 0],
        y=y_approx,
        marker=".",
        label="Local linear approximation",
    )
    ax.scatter(
        x=instance_to_be_explained.values[:, 0],
        y=fitted_model_prediction(
            instance_to_be_explained.values, sigma=sigma
        ),
        marker="*",
        s=200,
        label="Instance being explained",
    )
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.5, 0.5)
    ax.legend(loc="upper right")
    ax.set_ylabel("Model Predictions", fontsize=14)
    ax.set_xlabel("Feature 1", fontsize=14)
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    st.image(buffer)


def local_interpret(
    train_df: pd.DataFrame,
    instance_to_be_explained: pd.DataFrame,
    sigma: float = 1.0,
):
    pred_fn = partial(fitted_model_prediction, sigma=sigma)
    with st.expander("See LIME explanation code details:"):
        with st.echo():
            explainer = LimeTabularExplainer(
                train_df.values,
                mode="regression",
                feature_names=train_df.columns,
                discretize_continuous=False,
                sample_around_instance=True,
            )
            explanation = explainer.explain_instance(
                instance_to_be_explained.values.ravel(), pred_fn
            ).as_list()
    return explanation


def plot_feature_importances(
    feature_importances: list,
    figsize=(10, 3),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fi_data = pd.DataFrame(feature_importances).sort_values(by=0)
    positive_fi = fi_data.iloc[:, 1] > 0
    color = positive_fi.map({True: "blue", False: "red"})

    ax.barh(fi_data.iloc[:, 0], fi_data.iloc[:, 1], height=0.3, color=color)
    ax.set_xlim(-0.4, 0.4)
    ax.vlines(0, -0.25, 3.25, colors="k", linestyles="dashed", alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=14)
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    st.image(buffer)


@st.cache
def generate_data(data_min, data_max):
    train_df = pd.DataFrame(
        np.random.uniform(low=data_min, high=data_max, size=(5000, 4)),
        columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"],
    )
    return train_df


def deploy_plots_for_theoretical_model():
    st.header("Theoretical Model")
    st.markdown(
        """
    To get an intuition for how LIME works under the hood, let's first consider
    a theoretical model defined as a concrete mathematical function $M_{\sigma}$
    with parameter $\sigma$:
    """
    )
    st.latex(
        r"""
    M_{\sigma}(x) = \exp\left(-\frac{x^2}{2\sigma^2}\right)
    """
    )
    st.sidebar.markdown("Model parameter:")
    s = st.sidebar.slider("Sigma", 0.05, 0.5, 0.2)
    st.markdown(
        """
    In the sidebar, we can choose a value for $\sigma$. As for the data, we will
    use a randomly generated dataset with four numerical features and we show a 
    sample here: 
    """
    )
    data_min = -0.5
    data_max = 0.5
    train_df = generate_data(data_min, data_max)
    st.table(train_df.head())
    st.markdown(
        """
    Let's further define $M_{\sigma}(x)$ to take only the first feature and 
    ignore the rest. That is, $x$ only represents "Feature 1". Lastly, let's
    say that $M_{\sigma}$ has already been "trained" on this dataset.
    """
    )
    st.header("Explaining an instance")
    st.markdown(
        """
    Now, we pick an instance to be explained. That is, we want to know how the
    features of the said instance impact the model's predictions. In the 
    sidebar, we can choose the feature values of the instance we want to explain
    """
    )
    st.sidebar.markdown("Features of the instance to be explained:")
    f1 = st.sidebar.slider("Feature 1", data_min, data_max, 0.0)
    f2 = st.sidebar.slider("Feature 2", data_min, data_max, 0.0)
    f3 = st.sidebar.slider("Feature 3", data_min, data_max, 0.0)
    f4 = st.sidebar.slider("Feature 4", data_min, data_max, 0.0)
    instance_to_be_explained = pd.DataFrame(
        {
            "Feature 1": [f1],
            "Feature 2": [f2],
            "Feature 3": [f3],
            "Feature 4": [f4],
        }
    )

    st.markdown(
        """
    In the first figure below, we visualize the model predictions over the
    "training" dataset and include a marker to represent the model prediction 
    for the instance being explained. Around the instance, we also consider a
    local linear approximation of the model. 
    
    In the second figure, we show a barplot representing the feature importances
    of the instance as estimated by LIME. Blue bars correspond to positive
    impact, while red bars correspond to negative impact. The size of the bars
    refer to how much impact the feature has on the model prediction. In our 
    setup, a positive feature impact means that increasing the feature 
    translates to an increase in the model prediction, while a negative feature
    impact means the opposite.
    """
    )
    plot_model_and_local_preds(
        train_df, instance_to_be_explained, sigma=s, figsize=(10, 4)
    )
    fi = local_interpret(train_df, instance_to_be_explained, sigma=s)
    plot_feature_importances(fi, figsize=(10, 3))

    st.markdown(
        """
    We observe that the feature importance of the instance is related to the 
    slope of the linear approximation around the said instance. This is directly
    seen by the feature importance of "Feature 1". As expected, when "Feature 1"
    is negative, the slope of the local linear approximation is positive and the 
    estimated "Feature 1" importance is also positive. The opposite is true 
    when "Feature 1" is positive. As for the other features, we see that their
    feature importances are always very close to zero meaning that they have
    little to no impact on the model prediction. This makes sense because by
    construction, the model ignores the other features and only uses "Feature 1"
    
    The intuition for LIME is then about **finding a linear approximation of the
    model around the instance being explained** and extracting the coefficients 
    of the resulting linear model. As an overview, LIME does this by:
    - Generating a set of points that mimic the training set (called 
        perturbations) 
    - Assigning weights on each of the perturbations (higher weights are given 
        to those closer to the instance being explained)
    - Getting the model predictions on the instance and the perturbations
    - And lastly, training a weighted linear model based on the instance along 
        with the perturbations and the corresponding model predictions
    """
    )
