import streamlit as st
import requests

API_BASE = "https://zmffo3m7oe.execute-api.us-east-1.amazonaws.com/prod"

st.set_page_config(page_title="DocLens", page_icon="🔍", layout="centered")
st.title("🔍 DocLens")
st.caption("Upload a PDF to summarize it or ask questions about it.")

tab1, tab2, tab3 = st.tabs(["Ingest", "Query", "Summarize"])

with tab1:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Ingest", key="ingest_btn"):
        with st.spinner("Ingesting..."):
            response = requests.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Ingested in {data['duration']}")
            st.metric("Chunks stored", data["chunk_count"])
            st.metric("Pages", data["page_count"])
        elif response.status_code == 409:
            st.warning(response.json()["detail"])
        else:
            st.error(response.json().get("detail", "Something went wrong."))

with tab2:
    st.header("Ask a question")
    filename_q = st.text_input("Filename (exactly as uploaded)", key="filename_q")
    question = st.text_input("Question")

    if st.button("Ask", key="ask_btn"):
        if not filename_q or not question:
            st.warning("Please enter both a filename and a question.")
        else:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_BASE}/query",
                    json={"filename": filename_q, "question": question},
                )
            if response.status_code == 200:
                data = response.json()
                st.success(f"Answered in {data['duration']}")
                st.write(data["answer"])
                st.caption(f"Based on {data['chunks_used']} retrieved chunks")
            else:
                st.error(response.json().get("detail", "Something went wrong."))

with tab3:
    st.header("Summarize a document")
    filename_s = st.text_input("Filename (exactly as uploaded)", key="filename_s")

    if st.button("Summarize", key="summarize_btn"):
        if not filename_s:
            st.warning("Please enter a filename.")
        else:
            with st.spinner("Summarizing... this may take up to a minute for long documents."):
                response = requests.post(
                    f"{API_BASE}/summarize",
                    json={"filename": filename_s},
                )
            if response.status_code == 200:
                data = response.json()
                st.success(f"Summarized {data['chunk_count']} chunks in {data['duration']}")
                st.write(data["summary"])
            else:
                st.error(response.json().get("detail", "Something went wrong."))

with st.sidebar:
    st.header("Ingested documents")
    if st.button("Refresh", key="refresh_btn"):
        response = requests.get(f"{API_BASE}/documents")
        if response.status_code == 200:
            data = response.json()
            if data["documents"]:
                for doc in data["documents"]:
                    st.write(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)")
                st.caption(f"{data['total_documents']} document(s), {data['total_chunks']} total chunks")
            else:
                st.write("No documents ingested yet.")