import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from lime.lime_tabular import LimeTabularExplainer


def fitted_model_prediction(array: np.ndarray) -> np.ndarray:
    x = array[:, 0]
    y = np.exp(-(x ** 2) / 2)
    return y


def local_approximation(
    array: np.ndarray, approx_at: np.ndarray
) -> np.ndarray:
    x = array[:, 0]
    x0 = approx_at[:, 0]
    m = -x0 * np.exp(-(x0 ** 2) / 2)
    b = fitted_model_prediction(approx_at)

    y = m[0] * (x - x0[0]) + b[0]
    return y


def plot_model_and_local_preds(train_df, instance_to_be_explained):
    x = train_df.values[:, 0]
    y_pred = fitted_model_prediction(train_df.values)

    local_data = train_df.values / 6 + instance_to_be_explained.values
    y_approx = local_approximation(
        local_data, approx_at=instance_to_be_explained.values
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.scatter(x=x, y=y_pred, marker=".")
    ax.scatter(x=local_data[:, 0], y=y_approx, marker=".")
    ax.scatter(
        x=instance_to_be_explained.values[:, 0],
        y=fitted_model_prediction(instance_to_be_explained.values),
        marker="*",
        s=200,
    )
    ax.set_ylim(0, 1.2)
    return fig


def local_interpret(train_df, instance_to_be_explained):
    explainer = LimeTabularExplainer(
        train_df.values,
        mode="regression",
        feature_names=train_df.columns,
        discretize_continuous=False,
        sample_around_instance=True,
    )
    explanation = explainer.explain_instance(
        instance_to_be_explained.values.ravel(), fitted_model_prediction
    )
    return explanation


# --------------------------------- APP ----------------------------------------
st.title("Theoretical Model")
st.sidebar.markdown("Get feature importance for:")
f1 = st.sidebar.slider("Feature 1", -2.5, 2.5, 0.0)
f2 = st.sidebar.slider("Feature 2", -2.5, 2.5, 0.0)

train_df = pd.DataFrame(
    np.random.uniform(low=-2.5, high=2.5, size=(10000, 2)),
    columns=["Feature 1", "Feature 2"],
)

instance_to_be_explained = pd.DataFrame(
    {
        "Feature 1": [f1],
        "Feature 2": [f2],
    }
)

# print(explanation.as_list())

fig = plot_model_and_local_preds(train_df, instance_to_be_explained)
st.pyplot(fig)
