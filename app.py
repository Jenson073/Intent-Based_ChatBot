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
import time

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
    st.title("Enhanced Intent-Based Chatbot")

    # Sidebar menu
    menu = ["Home", "Chat History", "Model Evaluation", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Theme toggle
    theme = st.sidebar.radio("Choose Theme", options=["Light", "Dark"])
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Load data and train model
    patterns, tags, responses = load_data('/content/drive/My Drive/intents.json')
    x, y, vectorizer, label_encoder = preprocess_data(patterns, tags)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    if choice == "Home":
        st.write("Welcome! I'm here to assist you. You can ask me anything or choose an option from the menu.")
        input_container = st.empty()

        with input_container.container():
            user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")
            send_button = st.button("Send")

        if user_input or send_button:
            if user_input:
                with st.spinner("Chatbot is typing..."):
                    time.sleep(1)  # Simulate typing delay
                response = chatbot(user_input, clf, vectorizer, label_encoder, responses)
                st.session_state['chat_history'].append({
                    "user": user_input,
                    "chatbot": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.write(f"**You:** {user_input}")
                st.write(f"**Chatbot:** {response}")

    elif choice == "Chat History":
        st.subheader("Chat History")
        if st.session_state['chat_history']:
            for chat in st.session_state['chat_history'][-5:]:
                st.write(f"**You:** {chat['user']}")
                st.write(f"**Chatbot:** {chat['chatbot']}")
                st.write(f"**Timestamp:** {chat['timestamp']}")
                st.markdown("---")
        else:
            st.write("No chat history available.")

    elif choice == "Model Evaluation":
        st.subheader("Model Evaluation")
        model_accuracy = accuracy_score(y_test, clf.predict(X_test))
        classification_rep = classification_report(y_test, clf.predict(X_test))
        st.write(f"Model Accuracy: {model_accuracy * 100:.2f}%")
        st.write("### Classification Report")
        st.text(classification_rep)

    elif choice == "About":
        st.write("This project is a chatbot built using NLP and Streamlit.")

if __name__ == '__main__':
    display_chat()
