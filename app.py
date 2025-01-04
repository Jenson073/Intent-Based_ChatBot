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
    st.title("🤖 Enhanced Intent-Based Chatbot")
    st.markdown("Ask me anything! I'll do my best to provide meaningful answers.")

    # Sidebar menu
    st.sidebar.title("Menu")
    menu = ["Home", "Chat History", "Model Evaluation", "About"]
    choice = st.sidebar.radio("Navigate", menu)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Load data and train model
    patterns, tags, responses = load_data('intents.json')
    x, y, vectorizer, label_encoder = preprocess_data(patterns, tags)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    if choice == "Home":
        # Chat interface
        st.markdown("### Start Chatting:")
        input_container = st.empty()
        chat_container = st.container()

        with input_container:
            user_input = st.text_input("Type your message here...", key="user_input")
            send_button = st.button("Send")
            clear_button = st.button("Clear Chat")

        if clear_button:
            st.session_state['chat_history'] = []

        if user_input or send_button:
            if user_input:
                response = chatbot(user_input, clf, vectorizer, label_encoder, responses)
                st.session_state['chat_history'].append({"user": user_input, "chatbot": response})
                user_input = ""  # Clear input field

        # Display chat history
        with chat_container:
            for chat in st.session_state['chat_history']:
                st.markdown(f"**🧑 You:** {chat['user']}")
                st.markdown(f"**🤖 Bot:** {chat['chatbot']}")
                st.markdown("---")

    elif choice == "Chat History":
        st.subheader("Chat History")
        if st.session_state['chat_history']:
            for chat in st.session_state['chat_history']:
                st.write(f"**You:** {chat['user']}")
                st.write(f"**Chatbot:** {chat['chatbot']}")
                st.markdown("---")
        else:
            st.write("No chat history available.")

    elif choice == "Model Evaluation":
        st.subheader("Model Evaluation")
        y_pred = clf.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        st.write(f"**Model Accuracy:** {model_accuracy * 100:.2f}%")
        st.markdown("### Classification Report")
        st.text(classification_rep)

    elif choice == "About":
        st.write("""
        This project is an intent-based chatbot built using Natural Language Processing (NLP).
        It is designed to provide accurate responses based on user input and includes features like
        model evaluation, chat history, and an enhanced user interface.
        """)

if __name__ == '__main__':
    display_chat()
