import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from api_client import APIClientError, ask_question, get_health


load_dotenv(Path(__file__).resolve().parent / ".env")
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")

st.set_page_config(page_title="Hands-On ML Book Assistant", layout="wide")
st.title("Hands-On Machine Learning Book Assistant")
st.caption("Ask questions about the indexed book and inspect the cited passages used for each answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Backend connection")
    if API_BASE_URL:
        st.code(API_BASE_URL)
        if st.button("Check API health", use_container_width=True):
            try:
                health = get_health(API_BASE_URL)
                if health.get("status") == "ok":
                    st.success("API, vector store, and Ollama are ready.")
                else:
                    st.warning(health.get("startup_error", "The backend is degraded."))
                st.json(health)
            except APIClientError as exc:
                st.error(str(exc))
    else:
        st.error("API_BASE_URL is not configured. Copy frontend/.env.example to frontend/.env.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
            if message.get("passages"):
                with st.expander("Retrieved passage evidence"):
                    for passage in message["passages"]:
                        st.markdown(
                            f"**{passage['source']} | page {passage['page']} | "
                            f"distance {passage['distance']:.4f}**"
                        )
                        st.caption(passage["text"])

question = st.chat_input("Ask about a machine learning concept")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not API_BASE_URL:
            message = "The backend URL is not configured. Set API_BASE_URL and restart Streamlit."
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})
        else:
            with st.spinner("Searching the book and preparing a grounded answer..."):
                try:
                    response = ask_question(API_BASE_URL, question)
                    st.markdown(response["answer"])
                    if response["sources"]:
                        with st.expander("Sources used"):
                            for source in response["sources"]:
                                st.markdown(f"- {source}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                        "passages": response["passages"],
                    })
                except APIClientError as exc:
                    message = f"I couldn't get an answer: {exc}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})
