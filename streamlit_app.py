import streamlit as st
from rag import ask_bot

st.set_page_config(
    page_title="KalaStree GI Assistant",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 KalaStree GI Assistant")

question = st.text_input(
    "Ask a question about GI products"
)

if st.button("Ask"):

    with st.spinner("Searching..."):

        result = ask_bot(question)

        st.subheader("Answer")

        st.write(result["answer"])

        st.subheader("Sources")

        for source in result["sources"]:
            st.write(f"• {source}")