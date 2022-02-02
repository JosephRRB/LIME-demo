import streamlit as st
from theoretical_model import deploy_plots

st.title("Demos for Local Interpretability")
st.markdown(
    """
We will use LIME to explain a model's behaviour on a single instance. The 
explanations here refer to the feature's impact on the model behaviour. 
These explanations are specialized only in a local neighborhood of the instance 
being explained.
"""
)

deploy_plots()
