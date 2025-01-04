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
    st.set_page_config(
        page_title="Intent-Based Chatbot",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 Intent-Based Chatbot")

    # Sidebar menu
    menu = ["Home", "Chat History", "Model Evaluation", "About"]
    choice = st.sidebar.selectbox("📜 Navigation Menu", menu)

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
        st.subheader("💬 Welcome to the Intent-Based Chatbot")
        st.write("""
        Start chatting with the bot by typing your query in the text box below. 
        Use the sidebar to navigate through chat history, model evaluation, or learn more about the chatbot.
        """)
        input_container = st.empty()

        with input_container.container():
            user_input = st.text_input("💬 Your Message:", key="user_input", placeholder="Ask me anything...")
            send_button = st.button("🚀 Send")

        if user_input or send_button:
            if user_input:
                with st.spinner("🤖 Generating a response..."):
                    time.sleep(1)  # Simulate typing delay
                response = chatbot(user_input, clf, vectorizer, label_encoder, responses)
                st.session_state['chat_history'].append({
                    "user": user_input,
                    "chatbot": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.write(f"**💬 You:** {user_input}")
                st.write(f"**🤖 Chatbot:** {response}")

    elif choice == "Chat History":
        st.subheader("📜 Conversation History")
        st.write("Below is a record of your recent conversations with the chatbot.")
        if st.session_state['chat_history']:
            for chat in st.session_state['chat_history'][-5:]:
                st.write(f"**💬 You:** {chat['user']}")
                st.write(f"**🤖 Chatbot:** {chat['chatbot']}")
                st.write(f"**⏱ Timestamp:** {chat['timestamp']}")
                st.markdown("---")
            if st.button("🗑️ Clear History"):
                st.session_state['chat_history'] = []
                st.success("Chat history cleared!")
        else:
            st.info("📂 No chat history available.")

    elif choice == "Model Evaluation":
        st.subheader("📊 Model Performance Evaluation")
        st.write("Evaluate the performance of the Intent-Based Chatbot.")
        model_accuracy = accuracy_score(y_test, clf.predict(X_test))
        classification_rep = classification_report(y_test, clf.predict(X_test))
        st.write(f"📈 **Model Accuracy:** {model_accuracy * 100:.2f}%")
        st.write("### 🛠 Classification Report")
        st.text(classification_rep)

    elif choice == "About":
        st.subheader("ℹ️ About the Intent-Based Chatbot")
        st.markdown("""
        This chatbot is designed to classify user intents and respond with appropriate answers.
        ### Features:
        - **TF-IDF (Term Frequency-Inverse Document Frequency):** Extracts meaningful features from text.
        - **Random Forest Classifier:** Predicts the user intent with high accuracy.
        - **Streamlit:** Creates an interactive web application for deployment.

        Developed as a demonstration of **NLP** and **Machine Learning** in building conversational AI.
        """)

if __name__ == '__main__':
    display_chat()
