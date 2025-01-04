# Intent-Based Chatbot with Streamlit

This repository contains two intent-based chatbot built using Natural Language Processing (NLP) techniques, trained using a machine learning model, and one using Localtunnel to deploy the Streamlit app to public and other directly uses Streamlit account to create a Streamlit app.

## Setup Instructions

To view and run the chatbot using Localtunnel, follow the instructions in the provided Notebook (`.ipynb`). The notebook contains step-by-step instructions for setting up the environment, preparing the data, and running the chatbot using Streamlit.

To run the chatbot using Streamlit, go to the Streamlit website and create a account and copy the app.py, requirements.txt, intents.json files to a github repository and then head to your account give create app option and connect with your github repository then fill the required fields once done Streamlit will fetch the details and give a public url for your app thus deploying your Streamlit app.

## Code Explanation

This project involves several key components that work together to create a functioning chatbot. Below is an overview of each part of the code:

### **1. Data Loading and Structure**
The data used by the chatbot is stored in a JSON file. This file contains predefined intents, patterns (user input examples), and responses. The `load_data()` function loads and parses this data, separating it into three main components:
- **Patterns**: The input phrases that a user may provide (e.g., "Hello", "How are you?").
- **Tags**: These are the corresponding intent labels associated with each input phrase (e.g., "greeting", "goodbye").
- **Responses**: The chatbot’s responses that correspond to each tag.

These components are essential to train the model and provide responses based on user input.

### **2. Data Preprocessing**
Once the data is loaded, the `preprocess_data()` function transforms the text data into a format that can be used by the machine learning model:
- **TF-IDF Vectorization**: The function uses the `TfidfVectorizer` to convert the textual patterns into numerical data. This process captures the importance of each word in the input patterns, allowing the model to differentiate between important and common words.
- **Label Encoding**: The intent tags (e.g., "greeting", "goodbye") are encoded into numerical values using the `LabelEncoder`. This allows the machine learning model to process and classify the intents as numbers.

### **3. Machine Learning Model Training**
The chatbot uses a **Random Forest Classifier** to classify the intent of user inputs. The `chatbot()` function processes the user input by:
- Converting the input text into a vector using the same TF-IDF vectorizer used during training.
- Passing the vector through the trained Random Forest Classifier to predict the intent of the input.
- Retrieving a response from a predefined set of responses associated with the predicted intent. If no matching intent is found, the chatbot provides a default response.

### **4. Streamlit Interface**
The Streamlit framework is used to create the web interface for the chatbot. The `display_chat()` function sets up a simple chat interface with the following features:
- **Text Input**: The user can type their message, and the chatbot responds in real-time.
- **Chat History**: The chatbot maintains a history of the conversation, which can be reviewed by the user.
- **Model Evaluation**: The app also includes a section to evaluate the performance of the machine learning model using accuracy metrics and a classification report.

### **5. Deployment with LocalTunnel and Streamlit**
Once the chatbot is set up and running locally, the app can be accessed via a publicly accessible URL using **LocalTunnel**. This allows users to interact with the chatbot through a web interface, making it easy to deploy and share with others.

---

## Troubleshooting

If you encounter issues while setting up or running the chatbot, try the following:

**Bad Gateway Error (while accessing Streamlit app via LocalTunnel)**:
   - Solution: If you see a **Bad Gateway** error, it might be due to LocalTunnel's temporary issues. Try stopping the current tunnel and running the command again. Alternatively, reload the tunnel URL or restart the Streamlit app.

**While accessing Streamlit app via Streamlit Account**:
   - Ensure that the requirements.txt and dataset file are correctly given since these are directly fetched by Streamlit from github for the running of app.py.
---

## Requirements

Before starting, make sure you have the following:
- A **Google Drive** account for storing files.
- **Google Colab** or any Python environment that supports Jupyter notebooks.
- **Python 3.6+**: Make sure Python 3.6 or a higher version is installed.
- **Streamlit**: For running the interactive app.
- **NLTK**: For natural language processing (used to download tokenization data).
- **scikit-learn**: For training the machine learning model and performing evaluations.
- **localtunnel**: For exposing the Streamlit app over a public URL.
- **GitHub Repository**: With the required files for being fetched by the Streamlit.
