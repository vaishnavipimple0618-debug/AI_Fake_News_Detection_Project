import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Fake News Detection", page_icon="📰", layout="wide")

# Small illustrative training dataset for demonstration only.
# Do not treat this starter model as a reliable real-world fact checker.
TRAINING_DATA = [
    ("The city council approved a new public transport plan after a meeting on Tuesday.", "REAL"),
    ("The health department published its annual vaccination schedule on its official website.", "REAL"),
    ("Researchers published their findings in a peer reviewed scientific journal.", "REAL"),
    ("The school announced that classes will resume on Monday after the holiday.", "REAL"),
    ("The central bank released its quarterly report on inflation and economic growth.", "REAL"),
    ("The weather office issued a heavy rainfall warning for several districts.", "REAL"),
    ("The university opened applications for its scholarship programme this week.", "REAL"),
    ("The railway department announced revised train timings effective next month.", "REAL"),
    ("The company said its annual financial results are available in the investor report.", "REAL"),
    ("The local administration said road repairs will begin after the monsoon.", "REAL"),
    ("A government notice lists the eligibility criteria and application deadline.", "REAL"),
    ("The hospital opened a new outpatient department, according to its press release.", "REAL"),
    ("Scientists report that the study needs further testing before conclusions are drawn.", "REAL"),
    ("The election commission published polling dates and voter information.", "REAL"),
    ("The public library will extend its opening hours during the examination period.", "REAL"),
    ("SHOCKING! Secret doctors reveal drinking this juice cures every disease overnight!", "FAKE"),
    ("You won't believe it! This miracle trick guarantees instant wealth for everyone!", "FAKE"),
    ("BREAKING: All banks will close forever tomorrow, share this message immediately!", "FAKE"),
    ("Scientists are hiding the truth: one simple food makes you immortal!", "FAKE"),
    ("Forward this now or your phone will be permanently blocked by the government!", "FAKE"),
    ("A secret formula guarantees every student a 100 percent score without studying!", "FAKE"),
    ("Celebrity secretly becomes president of the moon, sources say, no proof needed.", "FAKE"),
    ("This unbelievable cure works instantly and doctors do not want you to know!", "FAKE"),
    ("Share urgently: every citizen will receive free money today, click unknown link.", "FAKE"),
    ("The media is hiding this miracle discovery that solves all problems overnight!", "FAKE"),
    ("One strange trick guarantees a job and a huge salary in just one day!", "FAKE"),
    ("The shocking truth is being suppressed; forward this message to ten people.", "FAKE"),
    ("A secret message claims schools are closed nationwide tomorrow without any official notice.", "FAKE"),
    ("This product cures all illnesses and has absolutely no side effects, guaranteed!", "FAKE"),
    ("An anonymous post claims a major event happened but provides no evidence or source.", "FAKE"),
]
@st.cache_resource
def train_model():
    texts, labels = zip(*TRAINING_DATA)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(texts, labels)
    return model

model = train_model()

st.markdown("""
<style>
.stApp { background: #f4f7fb; }
.block-container { max-width: 1120px; padding-top: 1.5rem; }
.hero { padding: 28px 32px; border-radius: 18px; color: white; background: linear-gradient(115deg,#172554,#2563eb 65%,#7c3aed); margin-bottom: 20px; }
.hero h1 { color:white; margin:0; font-size:2.2rem; }
.hero p { color:#e5edff; margin:8px 0 0 0; }
.note { background:#fff7ed; border-left:4px solid #f59e0b; padding:12px 15px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📰 AI-Based Fake News Detection System</h1><p>Analyze text patterns and review news responsibly.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="note"><b>Educational prototype:</b> this model uses a tiny sample dataset and cannot verify facts. Predictions may be wrong. Always check trusted sources.</div>', unsafe_allow_html=True)
st.write("")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Enter news text")
    sample = st.selectbox("Try a sample", ["Choose a sample...", "Official-style report", "Sensational claim"])
    default_text = ""
    if sample == "Official-style report":
        default_text = "The weather office issued a heavy rainfall warning for several districts."
    elif sample == "Sensational claim":
        default_text = "SHOCKING! Secret doctors reveal drinking this juice cures every disease overnight!"
    news_text = st.text_area("Paste a headline or article", value=default_text, height=180, placeholder="Paste news text here...")
    analyze = st.button("🔍 Analyze news", type="primary", use_container_width=True)
with right:
    st.subheader("How it works")
    st.markdown("**1. Text processing**  \nTF-IDF converts text into numerical features.")
    st.markdown("**2. Classification**  \nLogistic Regression predicts a sample category.")
    st.markdown("**3. Review**  \nCheck reliable sources before sharing any story.")
    st.info("The label is a model prediction, not proof that a story is true or false.")

if analyze:
    if len(news_text.strip()) < 15:
        st.warning("Please enter a little more text (at least 15 characters).")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100
        st.divider()
        st.subheader("Analysis result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted category", prediction)
        c2.metric("Model confidence", f"{confidence:.1f}%")
        c3.metric("Characters analyzed", f"{len(news_text):,}")
        if prediction == "FAKE":
            st.error("⚠️ Pattern resembles examples labelled FAKE in the demo dataset.")
        else:
            st.success("ℹ️ Pattern resembles examples labelled REAL in the demo dataset.")
        st.progress(confidence / 100)
        st.caption("Confidence describes this model's classification, not the probability that the claim is factually true.")
        st.markdown("**Recommended verification steps**")
        st.write("- Find the original source and publication date.")
        st.write("- Check whether multiple reputable outlets report the same facts.")
        st.write("- Look for official statements or primary documents.")
        st.write("- Do not share the story based only on this prediction.")

with st.expander("About the project"):
    st.write("This student project demonstrates text classification with TF-IDF and Logistic Regression. The bundled examples are synthetic and intentionally small for an easy local demo. For a serious system, train and evaluate on a reputable, labelled dataset and report precision, recall, F1-score, and limitations.")
    st.write("Technology: Python, Streamlit, Pandas, scikit-learn.")
