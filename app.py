import pickle
import string

import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ==========================================
# NLTK SETUP
# ==========================================

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download("punkt")
    try:
        nltk.download("punkt_tab")
    except Exception:
        pass


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# TEXT PREPROCESSING
# ==========================================

ps = PorterStemmer()


def transform_text(text):

    text = text.lower()

    words = nltk.word_tokenize(text)

    words = [
        word for word in words
        if word.isalnum()
    ]

    words = [
        word for word in words
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]

    words = [
        ps.stem(word)
        for word in words
    ]

    return " ".join(words)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .result {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-top: 25px;
    }

    .spam {
        background: #3b1717;
        color: #ff6b6b;
        border: 1px solid #ff6b6b;
    }

    .ham {
        background: #16351f;
        color: #5ee58a;
        border: 1px solid #5ee58a;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="main-title">📩 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Check whether your message is Spam or Ham</div>',
    unsafe_allow_html=True
)


# ==========================================
# MESSAGE INPUT
# ==========================================

message = st.text_area(
    "Enter your message",
    placeholder="Type or paste your SMS here...",
    height=180
)


# ==========================================
# CHECK BUTTON
# ==========================================

if st.button(
    "🔍 Check Message",
    use_container_width=True
):

    if not message.strip():

        st.warning("Please enter a message.")

    else:

        # Preprocess message
        transformed_message = transform_text(message)

        # Convert text into numerical features
        message_vector = vectorizer.transform(
            [transformed_message]
        )

        # Prediction
        prediction = model.predict(
            message_vector
        )[0]

        print("Prediction:", prediction)
        print("Model classes:", model.classes_)

        
        # ==================================
        # RESULT
        # ==================================

        if prediction == 1:

            st.markdown(
                """
                <div class="result spam">
                    🚨 SPAM MESSAGE
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result ham">
                    ✅ HAM — NOT SPAM
                </div>
                """,
                unsafe_allow_html=True
            )