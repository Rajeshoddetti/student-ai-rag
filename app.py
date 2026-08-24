import time
from collections import Counter

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from groq import RateLimitError

VECTORSTORE_DIR = "vectorstore"
COOLDOWN_SECONDS = 8
RETRIEVE_K = 6

FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

MARKS_RULES = {
    "2 Marks": "Give a crisp definition followed by 2-3 bullet points. No extra explanation, no examples. Under 40 words.",
    "5 Marks": "Explain with a short intro line and 4-6 bullet points covering the core sub-concepts. Roughly 100-130 words.",
    "10 Marks": "Write a detailed exam answer: brief intro, then organized sections/headings covering definition, working, types, examples, and note where a diagram belongs as '[Diagram: description]'. 250-350 words.",
}

st.set_page_config(page_title="Student AI", page_icon="🎓", layout="wide")
st.title("🎓 Student AI")
st.caption("JNTUH syllabus-based answers from your own PDFs only — no outside knowledge, no guessing.")


@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)


try:
    db = load_vector_db()
except Exception:
    st.error(
        f"Couldn't find '{VECTORSTORE_DIR}/'. Run `python ingest.py` locally first, "
        "then commit the vectorstore folder before deploying."
    )
    st.stop()


def get_llm():
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            return ChatGroq(
                api_key=st.secrets["GROQ_API_KEY"],
                model_name=model_name,
                temperature=0.15,
                max_tokens=700,
            )
        except Exception as e:
            last_error = e
            continue
    st.error(f"All Groq models are currently unavailable: {last_error}")
    st.stop()


llm = get_llm()

col1, col2 = st.columns(2)
with col1:
    subject_filter = st.selectbox(
        "Subject (optional — narrows the search)",
        ["Any"] + sorted({d.metadata.get("subject", "unknown") for d in db.docstore._dict.values()}),
    )
with col2:
    answer_type = st.selectbox("Select answer type", ["2 Marks", "5 Marks", "10 Marks"])

question = st.text_input("Ask any question (Define / Explain / Advantages / Applications etc.)")

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

if question:
    if time.time() - st.session_state.last_request_time < COOLDOWN_SECONDS:
        st.warning("⏳ Please wait a few seconds before asking another question.")
        st.stop()
    st.session_state.last_request_time = time.time()

    with st.spinner("Searching your syllabus PDFs..."):
        search_filter = None if subject_filter == "Any" else {"subject": subject_filter}
        docs = db.similarity_search(question, k=RETRIEVE_K, filter=search_filter)

        if not docs:
            st.warning("No relevant content found in your PDFs for this question/subject.")
            st.stop()

        detected_subject = Counter(d.metadata.get("subject", "unknown") for d in docs).most_common(1)[0][0]
        context = "\n\n".join(d.page_content for d in docs)

        system_prompt = f"""You are Student-AI, an exam-prep assistant for JNTUH B.Tech students.

ANSWER FORMAT RULE:
{MARKS_RULES[answer_type]}

STRICT RULES:
- Use ONLY the information in CONTEXT below. Never add outside facts or assume anything.
- If the context doesn't actually cover the question, say: "Not explicitly covered in the uploaded syllabus material."
- Write in clear, point-wise, exam-oriented English.
- Do not repeat sentences or pad the answer."""

        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
        except RateLimitError:
            st.warning("⚠️ Free usage limit reached. Please wait a minute and try again.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    st.subheader("📝 Exam-Ready Answer")
    st.success(f"Detected subject: {detected_subject}")
    st.info(f"Answer type: {answer_type}")
    st.write(response.content)
    st.caption("📘 Generated strictly from your syllabus PDFs (retrieval-grounded, model fallback enabled).")
