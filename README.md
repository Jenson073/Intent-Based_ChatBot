# Intent-Based Chatbot with Streamlit

This repository contains two intent-based chatbots built using Natural Language Processing (NLP) techniques. One chatbot is deployed using Localtunnel, and the other is deployed directly through a Streamlit account.

## Setup Instructions

To view and run the chatbot deployed using Localtunnel, follow the instructions in the provided Jupyter Notebook (`.ipynb`). The notebook contains step-by-step instructions for setting up the environment, preparing the data, and running the chatbot using Streamlit.

To view and interact with the chatbot deployed with Streamlit account, click the link below:
[https://intent-basedchatbot-jenson.streamlit.app](https://intent-basedchatbot-jenson.streamlit.app)

To run the chatbot deployed using Streamlit account:
1. Create an account on the [Streamlit website](https://streamlit.io).
2. Copy the `app.py`, `requirements.txt`, and `intents.json` files to a GitHub repository.
3. Head to your [Streamlit account](https://streamlit.io), and click on the **Create App** option.
4. Connect your GitHub repository and fill out the required fields.
5. Once done, Streamlit will fetch the details from GitHub and give a public URL for your app.

## Code Explanation

This project involves several key components that work together to create a functioning chatbot. Below is an overview of each part of the code:

### **1. Data Loading and Structure**
The data used by the chatbot is stored in a JSON file (`intents.json`). This file contains predefined intents, patterns (user input examples), and responses. The `load_data()` function loads and parses this data, separating it into:
- **Patterns**: User input phrases (e.g., "Hello", "How are you?").
- **Tags**: Corresponding intent labels (e.g., "greeting", "goodbye").
- **Responses**: The chatbot’s responses corresponding to each tag.

### **2. Data Preprocessing**
The `preprocess_data()` function transforms the text data into a format that can be used by the machine learning model:
- **TF-IDF Vectorization**: Converts the input patterns into numerical data.
- **Label Encoding**: Converts intent tags (e.g., "greeting", "goodbye") into numerical values.

### **3. Machine Learning Model Training**
The chatbot uses a **Random Forest Classifier** to classify the intent of user inputs:
- Converts user input into a vector using the trained `TfidfVectorizer`.
- Predicts the intent of the input using the Random Forest Classifier.
- Retrieves a response from a predefined set of responses associated with the predicted intent.

### **4. Streamlit Interface**
The Streamlit framework is used to create the web interface for the chatbot:
- **Text Input**: Users can type their messages.
- **Chat History**: Displays the conversation between the user and chatbot.
- **Model Evaluation**: A section to evaluate the model’s performance with accuracy metrics and classification report.

### **5. Deployment with LocalTunnel and Streamlit**
Once the chatbot is set up locally, the app can be exposed via a public URL using **LocalTunnel**. This allows users to interact with the chatbot through the web interface.

---

## Troubleshooting

If you encounter issues while setting up or running the chatbot, try the following:

### **Bad Gateway Error (while accessing Streamlit app via LocalTunnel)**:
- **Solution**: If you encounter a **Bad Gateway** error, it may be due to temporary issues with LocalTunnel. Try stopping the current tunnel and running the command again, or reload the tunnel URL and restart the Streamlit app.

### **While accessing Streamlit app via Streamlit Account**:
- Ensure that the `requirements.txt` and dataset file are correctly linked to the GitHub repository, as Streamlit fetches these files for running `app.py`.

---

## Requirements

Before starting, ensure you have the following:
- **Google Drive** account for storing files.
- **Google Colab** or any Python environment supporting Jupyter notebooks.
- **Python 3.6+**: Ensure Python 3.6 or a higher version is installed.
- **Streamlit**: For running the interactive app.
- **NLTK**: For natural language processing (used for downloading tokenization data).
- **scikit-learn**: For training the machine learning model and performing evaluations.
- **localtunnel**: For exposing the Streamlit app over a public URL.
- **GitHub Repository**: With the required files for Streamlit to fetch and deploy the app.

---

## Steps to Deploy Streamlit App

1. **Prepare Files**: Ensure you have the following files:
    - `app.py`: Main Streamlit app for the chatbot.
    - `intents.json`: JSON file with the chatbot data.
    - `requirements.txt`: A file listing all required Python dependencies.
  
2. **GitHub Repository**: Push the above files to a GitHub repository.

3. **Create a Streamlit App**:
    - Visit your [Streamlit account](https://streamlit.io).
    - Click on **Create App**.
    - Select the GitHub repository containing the above files.
    - Streamlit will automatically fetch the files and generate a public URL for your chatbot.

---

Enjoy using your interactive chatbot with Streamlit!
