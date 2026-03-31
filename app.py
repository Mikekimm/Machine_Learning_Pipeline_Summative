import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Intel Scene Classifier",
    page_icon="🔬",
    layout="wide"
)

# Hide Streamlit's default header/footer for a clean look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container { padding: 0 !important; }
    </style>
""", unsafe_allow_html=True)

with open("ui/dashboard.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=950, scrolling=True)