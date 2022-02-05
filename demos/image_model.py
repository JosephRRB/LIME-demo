# tensorflow-2.4.0
# scikit-image-0.17.2
# tqdm-4.62.3
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import mark_boundaries

import streamlit as st
# import tensorflow as tf
from lime.lime_image import LimeImageExplainer
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input
)
from tensorflow.keras.preprocessing import image


def choose_from_default_images():
    data_dir = Path(__file__).parents[1] / "demos/data"
    image_filenames = ["dog_test.jpg", "cat_test.jpeg"]
    image_names = ["Image 1", "Image 2"]
    loaded_images = {
        name: image.load_img(data_dir / file, target_size=(224, 224))
        for name, file in zip(image_names, image_filenames)
    }
    cols = st.columns(2)
    for col, (name, loaded_image) in zip(cols, loaded_images.items()):
        col.image(loaded_image)
        col.subheader(name)
    selected_name = st.sidebar.radio("Choose an image:", image_names)
    selected_image = loaded_images[selected_name]
    return selected_image


def select_image():
    selected_image = None
    use_camera = st.checkbox("Use camera to take a picture?", value=False)
    if use_camera:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            selected_image = Image.open(img_file_buffer).resize((224, 224))
    else:
        selected_image = choose_from_default_images()
    return selected_image


def preprocess_image(img):
    image_array = image.img_to_array(img)
    preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    return preprocessed


@st.cache
def get_pretrained_model():
    model = MobileNetV2(weights="imagenet", include_top=True)
    return model


def get_model_pred_probas(model, image_input):
    preds = model.predict(image_input)
    decoded_preds = decode_predictions(preds, top=5)[0]
    pred_probas = pd.DataFrame(
        [(name, proba) for _, name, proba in decoded_preds],
        columns=["Class Names", "Probabilities"],
    ).set_index("Class Names")
    return pred_probas


def plot_top_probable_classes(pred_probas, figsize=(10, 5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pred_probas.plot.bar(ax=ax, rot=45)
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    st.image(buffer)

@st.cache
def explain_instance(model, image_input):
    exp = LimeImageExplainer().explain_instance(
        image_input[0].astype("double"), model.predict, top_labels=5
    )
    return exp

def explain_predicted_class(exp, class_idx):
    class_idx_in_model = exp.top_labels[class_idx]
    temp_image, mask = exp.get_image_and_mask(
        class_idx_in_model, positive_only=True, num_features=5, hide_rest=False
    )
    explained_image = mark_boundaries(temp_image / 2 + 0.5, mask)
    return explained_image


selected_image = select_image()
if selected_image is not None:
    preprocessed_image = preprocess_image(selected_image)
    model = get_pretrained_model()
    pred_probas = get_model_pred_probas(model, preprocessed_image)

    plot_top_probable_classes(pred_probas, figsize=(10, 5))
    # st.pyplot(pred_probas.plot.bar())
    # st.dataframe(pred_probas)

    exp = explain_instance(model, preprocessed_image)
    class_to_explain = st.selectbox(
        "Choose class to explain:", pred_probas.index, index=0
    )
    class_idx = pred_probas.index.get_loc(class_to_explain)
    explained_image = explain_predicted_class(exp, class_idx)
    st.image(explained_image)
