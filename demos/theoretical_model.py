import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from lime.lime_tabular import LimeTabularExplainer
from functools import partial


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
):
    x = train_df.values[:, 0]
    y_pred = fitted_model_prediction(train_df.values, sigma=sigma)

    local_data = train_df.values / 6 + instance_to_be_explained.values
    y_approx = local_approximation(
        local_data,
        approx_at=instance_to_be_explained.values,
        sigma=sigma,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.scatter(x=x, y=y_pred, marker=".")
    ax.scatter(x=local_data[:, 0], y=y_approx, marker=".")
    ax.scatter(
        x=instance_to_be_explained.values[:, 0],
        y=fitted_model_prediction(
            instance_to_be_explained.values, sigma=sigma
        ),
        marker="*",
        s=200,
    )
    ax.set_ylim(0, 1.2)
    return fig


def local_interpret(
    train_df: pd.DataFrame,
    instance_to_be_explained: pd.DataFrame,
    sigma: float = 1.0,
):
    pred_fn = partial(fitted_model_prediction, sigma=sigma)
    explainer = LimeTabularExplainer(
        train_df.values,
        mode="regression",
        feature_names=train_df.columns,
        discretize_continuous=False,
        sample_around_instance=True,
    )
    explanation = explainer.explain_instance(
        instance_to_be_explained.values.ravel(), pred_fn
    )
    return explanation.as_list()


def plot_feature_importances(
    train_df: pd.DataFrame,
    instance_to_be_explained: pd.DataFrame,
    sigma: float = 1.0,
):
    fi = local_interpret(train_df, instance_to_be_explained, sigma=sigma)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    fi_data = pd.DataFrame(fi)
    positive_fi = fi_data.iloc[:, 1] > 0
    color = positive_fi.map({True: "blue", False: "red"})

    ax.barh(fi_data.iloc[:, 0], fi_data.iloc[:, 1], height=0.3, color=color)
    ax.set_xlim(-0.5, 0.5)
    ax.invert_yaxis()
    return fig

@st.cache
def generate_data(data_min, data_max):
    train_df = pd.DataFrame(
        np.random.uniform(low=data_min, high=data_max, size=(5000, 2)),
        columns=["Feature 1", "Feature 2"],
    )
    return train_df

def deploy_plots():
    data_min = -0.5
    data_max = 0.5

    st.title("Theoretical Model")
    st.sidebar.markdown("Model parameter:")
    s = st.sidebar.slider("Sigma", 0.05, 0.5, 0.2)

    st.sidebar.markdown("Get feature importance for:")
    f1 = st.sidebar.slider("Feature 1", data_min, data_max, 0.0)
    f2 = st.sidebar.slider("Feature 2", data_min, data_max, 0.0)

    train_df = generate_data(data_min, data_max)

    instance_to_be_explained = pd.DataFrame(
        {
            "Feature 1": [f1],
            "Feature 2": [f2],
        }
    )

    fig = plot_model_and_local_preds(
        train_df, instance_to_be_explained, sigma=s
    )
    st.pyplot(fig)

    bar = plot_feature_importances(train_df, instance_to_be_explained, sigma=s)
    st.pyplot(bar)
