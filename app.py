
import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.joblib")

model = load_model()

st.title("📰 Fake News Detection")
st.write(
    "Enter a news headline or article to see "
    "whether the model predicts it as likely fake or likely real."
)

news_text = st.text_area(
    "Paste news text here:",
    height=200,
    placeholder="Paste a headline or news article..."
)

if st.button("Analyze News"):
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        prediction = model.predict([news_text])[0]

        if prediction == 0:
            st.error("Prediction: Likely Fake")
        else:
            st.success("Prediction: Likely Real")

        st.caption(
            "This is a machine-learning prediction, not a fact-check. "
            "Verify important claims using reliable sources."
        )


