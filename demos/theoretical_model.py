import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    train_df = pd.DataFrame(
        np.random.uniform(low=-2.5, high=2.5, size=(10000, 2)),
        columns=["Feature 1", "Feature 2"],
    )

    explainer = LimeTabularExplainer(
        train_df.values,
        mode="regression",
        feature_names=train_df.columns,
        discretize_continuous=False,
        sample_around_instance=True,
    )

    instance_to_be_explained = pd.DataFrame(
        {
            "Feature 1": [-0.5],
            "Feature 2": [0.2],
        }
    )

    explanation = explainer.explain_instance(
        instance_to_be_explained.values.ravel(), fitted_model_prediction
    )
    print(explanation.as_list())

    x = train_df.values[:, 0]
    y_pred = fitted_model_prediction(train_df.values)

    local_data = train_df.values / 5 + instance_to_be_explained.values
    y_approx = local_approximation(
        local_data, approx_at=instance_to_be_explained.values
    )
    plt.scatter(x=x, y=y_pred, marker=".")
    plt.scatter(x=local_data[:, 0], y=y_approx, marker=".")
    plt.scatter(
        x=instance_to_be_explained.values[:, 0],
        y=fitted_model_prediction(instance_to_be_explained.values),
        marker="*",
        s=200,
    )
    plt.show()
