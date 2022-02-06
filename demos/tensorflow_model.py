from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import mark_boundaries

import streamlit as st
from lime.lime_image import LimeImageExplainer
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image


@st.cache
def load_default_images():
    data_dir = Path(__file__).parents[1] / "demos/data"
    image_names = [
        "Nathan Philipps Square",
        "Mary, Queen of the World Cathedral",
        "Mount Royal Park",
        "The Joust",
        "Maple the dog",
        "Venus the cat",
    ]
    loaded_images = {
        name: image.load_img(
            data_dir / (name + ".jpg"),
        )
        for name in image_names
    }
    return loaded_images


def choose_from_default_images():
    st.markdown(
        """
    By default, we can choose from a selection of images to be classified
    and explained for illustrative purposes. We can choose an image using the
    sidebar.
    """
    )
    loaded_images = load_default_images()
    cols = st.columns(2)
    for i, (name, loaded_image) in enumerate(loaded_images.items()):
        cols[i % 2].image(loaded_image, caption=name)
    selected_name = st.sidebar.radio(
        "Choose an image to be classified and explained:", loaded_images.keys()
    )
    selected_image = loaded_images[selected_name]
    return selected_image


def select_image():
    selected_image = None
    user_input = st.checkbox("Provide a picture?", value=False)
    if user_input:
        user_input_choices = ["Upload a picture", "Use the camera"]
        choice = st.selectbox(
            "Select an input method", user_input_choices, index=0
        )
        if choice == user_input_choices[0]:
            img_file_buffer = st.file_uploader(
                "Upload an image here", type=["jpg", "jpeg", "png"]
            )
        else:
            img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            selected_image = Image.open(img_file_buffer)
    else:
        selected_image = choose_from_default_images()
    return selected_image


def preprocess_image(img):
    image_array = image.img_to_array(img.resize((224, 224)))
    preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    return preprocessed


@st.cache
def get_pretrained_model():
    model = MobileNetV2(weights="imagenet", include_top=True)
    return model


@st.cache
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
    ax.bar(pred_probas.index, pred_probas.values.ravel(), width=0.5)
    ax.set_xlabel("Top 5 Predicted Classes", fontsize=14)
    ax.set_ylabel("Class Probabilities", fontsize=14)
    ax.set_title("Predictions of MobileNetV2", fontsize=18)
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
        class_idx_in_model, positive_only=True, num_features=10, hide_rest=True
    )
    explained_image = mark_boundaries(temp_image / 2 + 0.5, mask)
    return explained_image


def deploy_plots_for_tensorflow_model():
    st.header("Tensorflow Model")
    st.markdown(
        """
    For fun, let us now try to explain the predictions of a deep neural network
    for image classification. Specifically, we will use the tensorflow
    implementation of [`MobileNetV2()`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2)
    pretrained on the 1000-class image dataset [`ImageNet`](https://www.image-net.org/). 
    The architecture is described in this [paper](https://arxiv.org/abs/1801.04381)
    and we chose it for its [small size and fast inference time](https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    We will now use `LimeImageExplainer()` for explaining the image 
    classification predictions. It still uses the basic principles as the 
    tabular explainer but now the perturbations of the image to be explained are
    represented by collections of present or absent "*superpixels*". There are
    more details of how to apply LIME for images are available [here](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb)
    
    To begin, we now ask the user whether they want to provide an image to be
    classified by `MobileNetV2` and whose predictions are going to be explained 
    by `LimeImageExplainer`.
    """
    )
    selected_image = select_image()
    if selected_image is not None:
        st.markdown(
            """
        Having selected an image, we can now let `MobileNetV2` classify the 
        image and return the top 5 most probable classes (out of 1000 classes) 
        along with the model's probabilities associated to those classes. The 
        bar plot is ordered such that the most probable class is on the left
        while the 5th probable class is on the right.
        """
        )
        preprocessed_image = preprocess_image(selected_image)
        model = get_pretrained_model()
        pred_probas = get_model_pred_probas(model, preprocessed_image)
        plot_top_probable_classes(pred_probas, figsize=(10, 3))

        st.markdown(
            """
        We can now choose which class we want to explain. By default, we explain
        the most probable class of our selected image.
        """
        )
        class_to_explain = st.selectbox(
            "Choose a class to explain:", pred_probas.index, index=0
        )
        exp = explain_instance(model, preprocessed_image)
        class_idx = pred_probas.index.get_loc(class_to_explain)
        explained_image = explain_predicted_class(exp, class_idx)

        st.markdown(
            f"""
        Below, we show our chosen image (resized for the model) on the left and 
        the explanation image by LIME on the right. That is, `MobileNetV2` 
        classified our chosen image as `{class_to_explain}` because of the top
        *superpixels* present in the image explanation.
        """
        )
        col1, col2 = st.columns(2)
        col1.image(
            preprocessed_image[0] / 2 + 0.5,
            caption="Chosen image",
            use_column_width=True,
        )
        col2.image(
            explained_image,
            caption="Explanation for model prediction",
            use_column_width=True,
        )
