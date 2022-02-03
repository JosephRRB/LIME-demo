from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import streamlit as st


def fit_models():
    data_dir = Path(__file__).parents[1] / "demos/data"
    with st.expander("See training code details:"):
        with st.echo():
            df = pd.read_csv(
                data_dir / "data_banknote_authentication.txt", header=None
            )
            df.columns = [
                "variance of WTI",
                "skewness of WTI",
                "curtosis of WTI",
                "entropy of image",
                "class",
            ]
            df["class"] = df["class"].map({0: "genuine", 1: "forgery"})

            df_train, df_test = train_test_split(
                df,
                test_size=0.25,
            )
            X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]

            # Note: SVC has some issues with probability
            svc = SVC(probability=True).fit(X_train, y_train)
            knn = KNeighborsClassifier().fit(X_train, y_train)
            rfc = RandomForestClassifier().fit(X_train, y_train)
            mlp = MLPClassifier().fit(X_train, y_train)

    df_train.to_csv(
        data_dir / "train_banknote_authentication.csv",
        index=False,
    )
    df_test.to_csv(
        data_dir / "test_banknote_authentication.csv",
        index=False,
    )
    dump(svc, data_dir / "fitted_svc.joblib")
    dump(knn, data_dir / "fitted_knn.joblib")
    dump(rfc, data_dir / "fitted_rfc.joblib")
    dump(mlp, data_dir / "fitted_mlp.joblib")


# if __name__ == "__main__":
#     fit_models()
