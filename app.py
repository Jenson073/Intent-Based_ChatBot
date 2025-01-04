import os
import json
import random
import nltk
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Download necessary NLTK datasets
nltk.download('punkt')

# Function to load data from the intents file
def load_data(file_path):
    with open(file_path, 'r') as file:
        intents = json.load(file)
    patterns, tags, responses = [], [], {}
    for intent in intents:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
        responses[intent['tag']] = intent['responses']
    return patterns, tags, responses

# Function to preprocess data
def preprocess_data(patterns, tags):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    label_encoder = LabelEncoder()
    x = vectorizer.fit_transform(patterns)
    y = label_encoder.fit_transform(tags)
    return x, y, vectorizer, label_encoder

# Chatbot prediction function
def chatbot(input_text, clf, vectorizer, label_encoder, responses):
    input_vector = vectorizer.transform([input_text])
    predicted_tag_index = clf.predict(input_vector)[0]
    predicted_tag = label_encoder.inverse_transform([predicted_tag_index])[0]
    return random.choice(responses.get(predicted_tag, ["I'm not sure how to respond to that."]))

# Streamlit Chat Interface
def display_chat():
    st.set_page_config(page_title="Enhanced Chatbot", layout="wide")
    st.sidebar.title("Chatbot Navigation")
    menu = ["Home", "Chat History", "Model Evaluation", "About"]
    choice = st.sidebar.radio("Menu", menu)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Load data and train model
    patterns, tags, responses = load_data('intents.json')
    x, y, vectorizer, label_encoder = preprocess_data(patterns, tags)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # CSS for custom styling
    st.markdown("""
        <style>
        body {
            background-color: #f7f8fc;
        }
        .user-message {
            background-color: #dcf8c6;
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 60%;
        }
        .bot-message {
            background-color: #f1f0f0;
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 60%;
        }
        .bot-message-container {
            display: flex;
            justify-content: flex-start;
        }
        .user-message-container {
            display: flex;
            justify-content: flex-end;
        }
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            background: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    if choice == "Home":
        st.title("Chat with the Enhanced Chatbot")
        st.subheader("Interact with the chatbot and get real-time responses!")

        chat_container = st.container()
        input_container = st.container()

        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for chat in st.session_state['chat_history']:
                if chat["user"]:
                    st.markdown(
                        f'<div class="user-message-container"><div class="user-message"><b>You:</b> {chat["user"]}</div></div>',
                        unsafe_allow_html=True)
                if chat["chatbot"]:
                    st.markdown(
                        f'<div class="bot-message-container"><div class="bot-message"><b>Bot:</b> {chat["chatbot"]}</div></div>',
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with input_container:
            user_input = st.text_area("Ask something:", key="user_input", placeholder="Type your message here...", height=100)
            if st.button("Send") or user_input:
                if user_input:
                    response = chatbot(user_input, clf, vectorizer, label_encoder, responses)
                    st.session_state['chat_history'].append({
                        "user": user_input,
                        "chatbot": response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.session_state.user_input = ""  # Clear input field after sending

        if st.button("Clear Chat History"):
            st.session_state['chat_history'] = []

    elif choice == "Chat History":
        st.title("Chat History")
        if st.session_state['chat_history']:
            for chat in st.session_state['chat_history']:
                st.write(f"**You:** {chat['user']}")
                st.write(f"**Bot:** {chat['chatbot']}")
                st.write(f"**Timestamp:** {chat['timestamp']}")
                st.markdown("---")
        else:
            st.write("No chat history available.")

    elif choice == "Model Evaluation":
        st.title("Model Evaluation")
        model_accuracy = accuracy_score(y_test, clf.predict(X_test))
        classification_rep = classification_report(y_test, clf.predict(X_test))

        st.write(f"Model Accuracy: {model_accuracy * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_rep)

    elif choice == "About":
        st.title("About the Project")
        st.write("""
            This is a chatbot application built using NLP and machine learning techniques.
            The interface has been enhanced with a clean and interactive design.
        """)

if __name__ == '__main__':
    display_chat()
