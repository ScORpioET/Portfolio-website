import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to my portfolio! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is a place to showcase my Machine Learning and Data Science projects.
    **👈 Select the sidebar** to see some projects.
"""
)

st.markdown("![Alt Text](https://64.media.tumblr.com/1b2e59310a622150071a7b2513f530fe/9d06023dd253e49e-eb/s640x960/94190608daa3f1fff3477f9b70745a2e25bfa124.gifv)")