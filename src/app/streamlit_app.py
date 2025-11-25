# src/app/streamlit_app.py
import streamlit as st
import os
from orchestration.orchestrator import orchestrate

st.set_page_config(page_title="Audit Intelligence", layout="wide")

st.title("🧠 Audit Intelligence System")

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
st.sidebar.header("📄 Upload File")

debug_mode = st.sidebar.toggle("Debug Mode", value=False)

uploaded = st.sidebar.file_uploader(
    "Upload PDF / CSV / XLSX",
    type=["pdf", "csv", "xlsx"]
)

doc_path = None

if uploaded:
    doc_path = os.path.join("datasets/raw", uploaded.name)
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    with open(doc_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.sidebar.success("Uploaded: " + uploaded.name)

    st.info("📄 Document saved. Ready for questions!")

# ------------------------------------------
# USER QUERY
# ------------------------------------------
st.header("💬 Ask a Question About the Document")

query = st.text_input("Type your question and press ENTER:", placeholder="Example: 'Show all email IDs'")

if query:
    with st.status("🎯 Processing your request...", expanded=False) as s:

        result = orchestrate({"user_query": query, "doc_id": doc_path})

        s.update(label="✅ Completed!", state="complete")

    st.subheader("🔍 Answer")

    final = result.get("final_answer")

    if isinstance(final, list):
        st.write("\n".join([str(x) for x in final]))
    else:
        st.write(final)

    if debug_mode:
        st.divider()
        st.subheader("🛠 Debug Information")
        st.json(result)
