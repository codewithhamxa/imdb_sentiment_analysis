import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK stopwords
import nltk
nltk.download('stopwords')

# Load the trained model
loaded_model = joblib.load('imdb_moviesreviews.joblib')

# Load the vectorizer used during training
vectorizer = joblib.load('your_vectorizer.joblib')

# Assuming you have defined stop_words somewhere in your code
stop_words = set(stopwords.words('english'))

# Streamlit app
st.title('Movie Review Sentiment Prediction')

# User input for a new movie review
new_review = st.text_area('Enter your movie review here:')

# Function for preprocessing new reviews


def data_processing(text, stop_words, vectorizer):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<br />', ' ', text)

    # Remove URLs
    text = re.sub(r'https?\S+|www\S+|http\S+', '', text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    text_tokens = word_tokenize(text)

    # Remove stop words
    filtered_text = [w for w in text_tokens if not w in stop_words]

    # Vectorize the new review using the same vectorizer as during training
    new_review_vectorized = vectorizer.transform([" ".join(filtered_text)])

    return new_review_vectorized


# Predict sentiment on button click
if st.button('Predict Sentiment'):
    if new_review:
        # Preprocess the new review
        processed_review = data_processing(new_review, stop_words, vectorizer)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(processed_review)

        # Display the predicted sentiment
        st.success(f'Predicted Sentiment: {prediction[0]}')
    else:
        st.warning('Please enter a movie review.')
