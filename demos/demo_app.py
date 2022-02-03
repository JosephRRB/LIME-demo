import streamlit as st

from sklearn_models import deploy_plots_for_sklearn_models
from theoretical_model import deploy_plots_for_theoretical_model

st.title("Demos for Local Interpretability")
st.markdown(
    """
We will use LIME to explain a model's behaviour on a single instance. The 
explanations here refer to the feature's impact on the model behaviour. 
These explanations are specialized only in a local neighborhood of the instance 
being explained.
"""
)
demo_options = ["Theoretical Model", "Sklearn Models"]
demo = st.selectbox(
    "Select demo to run:", demo_options, index=0
)
if demo == demo_options[0]:
    deploy_plots_for_theoretical_model()
else:
    deploy_plots_for_sklearn_models()
