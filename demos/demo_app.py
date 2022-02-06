import streamlit as st

from sklearn_models import deploy_plots_for_sklearn_models
from theoretical_model import deploy_plots_for_theoretical_model
from tensorflow_model import deploy_plots_for_tensorflow_model

st.title("Explaining Model Predictions using LIME")
st.markdown(
    """
Machine learning is an important tool nowadays for predictive analytics. 
Fast-paced research is ongoing to make predictive pipelines more accurate for 
many different tasks. This high accuracy however gives us a trade-off. 
State-of-the-art models usually need to be very complex in order to be highly
accurate. This complexity means that we humans wouldn't know why the model
made a certain prediction. It might as well be a black-box that takes in data
and outputs random numbers. In high risk situations where lives can be saved or 
lost depending on the model predictions, we would be right not to trust a 
black-box whose conclusions we don't know how were made. To trust a model, we
then have a need to understand how its predictions were made.
    
["Why should I trust you?"](https://arxiv.org/abs/1602.04938) is the question
asked by the authors who proposed the model explanation technique called 
**LIME** (*Local Interpretable Model-agnostic Explanations*). Their work is also 
publicly available in this [GitHub repository](https://github.com/marcotcr/lime) 
which we will use to demonstrate how LIME works and how the code is used for 
different models and applications. For those interested, the source code for 
this set of demos can be found in my [GitHub repository](https://github.com/JosephRRB/LIME-demo).

We will use LIME to explain a model's behaviour on a single instance. The 
explanations here refer to the feature's impact on the model behaviour. 
These explanations are specialized only in a local neighborhood of the instance 
being explained.
"""
)
demo_options = ["Theoretical Model", "Sklearn Models", "Tensorflow Model"]
demo = st.selectbox(
    "Select demo to run:", demo_options, index=0
)
if demo == demo_options[0]:
    deploy_plots_for_theoretical_model()
elif demo == demo_options[1]:
    deploy_plots_for_sklearn_models()
else:
    deploy_plots_for_tensorflow_model()
