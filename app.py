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

# FAISS returns L2 distance (lower = more similar). If the best match's
# distance is above this, we treat the PDFs as "not covering this question"
# and fall back to general knowledge instead of forcing a weak match.
# Tune this if you notice good matches being rejected, or bad matches
# being accepted - all-MiniLM-L6-v2 typically sits in the 0.3-1.3 range.
RELEVANCE_DISTANCE_THRESHOLD = 1.1

FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

MARKS_RULES = {
    "2 Marks": "Give a crisp definition followed by 2-3 bullet points. No extra explanation, no examples. Under 40 words.",
    "5 Marks": "Explain with a short intro line and 4-6 bullet points covering the core sub-concepts. Roughly 100-130 words.",
    "10 Marks": "Write a detailed exam answer: brief intro, then organized sections/headings covering definition, working, types, examples, and note where a diagram belongs as '[Diagram: description]'. 250-350 words.",
}

st.set_page_config(page_title="Student AI", page_icon="🎓", layout="wide")
st.title("🎓 Student AI")
st.caption("JNTUH exam-ready answers - grounded in your syllabus PDFs when available, clearly flagged when not.")


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


def build_grounded_prompt(context, answer_type):
    return f"""You are Student-AI, an exam-prep assistant for JNTUH B.Tech students.

ANSWER FORMAT RULE:
{MARKS_RULES[answer_type]}

STRICT RULES:
- Use ONLY the information in CONTEXT below. Never add outside facts or assume anything.
- Write in clear, point-wise, exam-oriented English.
- Do not repeat sentences or pad the answer.

CONTEXT:
{context}"""


def build_fallback_prompt(answer_type, subject_hint):
    subject_line = f" The question is likely from the subject area: {subject_hint}." if subject_hint != "Any" else ""
    return f"""You are Student-AI, an exam-prep assistant for JNTUH B.Tech students.
No matching notes were found in the uploaded syllabus PDFs for this question, so answer using your
own general subject knowledge instead.{subject_line}

ANSWER FORMAT RULE:
{MARKS_RULES[answer_type]}

RULES:
- Write the answer in standard JNTUH exam style - clear, point-wise, exam-oriented English, as if it
  were going into an official syllabus note.
- Do not claim this came from any specific textbook or the student's own notes.
- If the question is too vague or not a real academic topic, say so plainly instead of making
  something up."""


if question:
    if time.time() - st.session_state.last_request_time < COOLDOWN_SECONDS:
        st.warning("⏳ Please wait a few seconds before asking another question.")
        st.stop()
    st.session_state.last_request_time = time.time()

    with st.spinner("Searching your syllabus PDFs..."):
        search_filter = None if subject_filter == "Any" else {"subject": subject_filter}
        scored_docs = db.similarity_search_with_score(question, k=RETRIEVE_K, filter=search_filter)

    use_grounded = bool(scored_docs) and scored_docs[0][1] <= RELEVANCE_DISTANCE_THRESHOLD

    with st.spinner("Writing your answer..."):
        if use_grounded:
            docs = [d for d, _score in scored_docs]
            detected_subject = Counter(d.metadata.get("subject", "unknown") for d in docs).most_common(1)[0][0]
            context = "\n\n".join(d.page_content for d in docs)
            system_prompt = build_grounded_prompt(context, answer_type)
        else:
            system_prompt = build_fallback_prompt(answer_type, subject_filter)

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"QUESTION:\n{question}"),
            ])
        except RateLimitError:
            st.warning("⚠️ Free usage limit reached. Please wait a minute and try again.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    st.subheader("📝 Exam-Ready Answer")
    if use_grounded:
        st.success(f"📘 From your syllabus PDFs — detected subject: {detected_subject}")
    else:
        st.warning("🌐 General knowledge answer — not found in your uploaded PDFs. Cross-check before relying on this for exams.")
    st.info(f"Answer type: {answer_type}")
    st.write(response.content)
