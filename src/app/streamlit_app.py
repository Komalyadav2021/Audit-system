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
    "Upload PDF / CSV / XLSX (optional)",
    type=["pdf", "csv", "xlsx"]
)

doc_path = None

if uploaded:
    doc_path = os.path.join("datasets/raw", uploaded.name)
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    with open(doc_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.sidebar.success("Uploaded: " + uploaded.name)
    st.info("📄 Document saved. You can now ask document-related questions!")

else:
    st.sidebar.info("No document uploaded — system will answer using fine-tuned knowledge.")

# ------------------------------------------
# USER QUERY
# ------------------------------------------
st.header("💬 Ask a Question")

query = st.text_input(
    "Type your question and press ENTER:",
    placeholder="Example: 'What is audit risk?' or 'Extract all email IDs from the PDF'"
)

if query:
    with st.status("🎯 Processing your request...", expanded=False) as s:

        # The orchestrator now supports BOTH:
        # 1. Question over PDF/CSV/XLSX
        # 2. Pure fine-tuned model Q&A with no doc
        result = orchestrate({
            "user_query": query,
            "doc_id": doc_path,   # will be None if no file uploaded
            "allow_no_doc": True, # new flag for your fine-tuner system
        })

        s.update(label="✅ Completed!", state="complete")

    st.subheader("🔍 Answer")

    final = result.get("final_answer")

    if isinstance(final, list):
        st.write("\n".join([str(x) for x in final]))
    else:
        st.write(final)

    # Show debug info
    if debug_mode:
        st.divider()
        st.subheader("🛠 Debug Information")
        st.json(result)
