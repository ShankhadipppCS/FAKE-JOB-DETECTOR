import streamlit as st
from src.predict import FakeJobDetector

st.set_page_config(
    page_title="Fake Job Detector",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return FakeJobDetector()

detector = load_model()

# Title
st.markdown("<h1 style='text-align: center;'>🕵️ AI Fraud Job Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect suspicious job postings using AI</p>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

# LEFT SIDE - INPUT
with col1:
    st.subheader("📄 Job Description")
    text = st.text_area("Paste job description", height=300)

# RIGHT SIDE - OUTPUT
with col2:
    st.subheader("📊 Analysis Result")

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter job description")
        else:
            result = detector.predict(text)

            label = result["label"]
            confidence = result["confidence"]

            # Big status banner
            if label == "Fake":
                st.error(f"🚨 FAKE JOB DETECTED")
            else:
                st.success("✅ LEGIT JOB")

            # Risk meter
            st.write("### Risk Score")
            st.progress(int(confidence * 100))

            st.write(f"Confidence: {confidence * 100:.1f}%")

            # Reasons
            st.write("### ⚠️ Warning Signals")
            if result["reasons"]:
                for r in result["reasons"]:
                    st.markdown(f"- {r}")
            else:
                st.write("No major red flags detected")

            # Highlight suspicious words
            st.write("### 🔍 Suspicious Keywords")
            keywords = ["no experience", "earn", "work from home", "quick money", "no interview"]
            found = [k for k in keywords if k in text.lower()]

            if found:
                st.write(", ".join(found))
            else:
                st.write("No suspicious keywords found")