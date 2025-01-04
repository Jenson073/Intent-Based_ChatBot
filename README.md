# 🤖 Intent-Based Chatbot with Streamlit

This repository contains two intent-based chatbots built using Natural Language Processing (NLP) techniques. One chatbot is deployed using **Localtunnel**, and the other is deployed directly through a **Streamlit account**.  

---

## 🌐 View the Deployed Chatbot

To interact with the chatbot deployed using Streamlit, click the link below:  
[🔗 Chatbot URL](https://intent-basedchatbot-jenson.streamlit.app)  

---

## 🛠️ Running and Setup Instructions

### 🧪 For Localtunnel Deployment
To view and run the chatbot deployed using Localtunnel, follow the instructions in the provided Jupyter Notebook (`.ipynb`). The notebook contains step-by-step instructions for:  
1️⃣ Setting up the environment  
2️⃣ Preparing the data and code 
3️⃣ Running the chatbot using Localtunnel on Streamlit 

### 🌟 For Streamlit Deployment
To run the chatbot with a Streamlit account:
1. ✍️ Create an account on the [Streamlit website](https://streamlit.io).  
2. 📁 Copy the `app.py`, `requirements.txt`, and `intents.json` files to a GitHub repository.  
3. 🔧 Go to your [Streamlit account](https://streamlit.io), and click on **Create App**.  
4. 🔗 Connect your GitHub repository and fill out the required fields.  
5. 🚀 Streamlit will fetch the details from GitHub and generate a public URL for your app.

---

## 🧩 Code Explanation

### **1️⃣ Data Loading and Structure**
- **Patterns**: User input phrases (e.g., "Hello", "How are you?").  
- **Tags**: Intent labels (e.g., "greeting", "goodbye").  
- **Responses**: Predefined chatbot responses for each tag.  

### **2️⃣ Data Preprocessing**
- 📝 **TF-IDF Vectorization**: Converts patterns into numerical data.  
- 🔢 **Label Encoding**: Converts intent tags into numerical values.  

### **3️⃣ Machine Learning Model Training**
- 🤖 Uses a **Random Forest Classifier** for intent classification.  
- 📈 Predicts user intent and retrieves a relevant response.  

### **4️⃣ Streamlit Interface**
- 💬 **Text Input**: Users type their messages.  
- 🗂️ **Chat History**: Displays conversation logs.  
- 📊 **Model Evaluation**: Includes accuracy metrics and a classification report.

### **5️⃣ Deployment**
- 🌍 Localtunnel for public URL generation.  
- 🚀 Streamlit for seamless web app deployment.  

---

## 🛠️ Troubleshooting

### ⚠️ **Bad Gateway Error (Localtunnel)**  
- **Solution**: Restart Localtunnel or reload the URL.  

### ⚠️ **Issues with Streamlit Deployment**  
- Ensure `requirements.txt` and `intents.json` are properly linked in the GitHub repository.  

---

## 📋 Requirements

- **Google Drive** (account for storing files)
- **Google Colab** (Any Python environment supporting Jupyter notebooks)
- 🖥️ **Python 3.6+**  
- 🌐 **Streamlit**  (For running the interactive app)
- 🧠 **NLTK** (for natural language processing)  
- 📊 **scikit-learn** (for ML model training)  
- 🚀 **localtunnel** (for public URL generation)  
- 💾 A GitHub repository with the required files.  

---

## 🚀 Steps to Deploy Streamlit App

1. **Prepare Files**:  
    - `app.py`: Main Streamlit app.  
    - `intents.json`: Chatbot data file.  
    - `requirements.txt`: Python dependencies.  

2. **Push Files to GitHub**:  
    - Upload the above files to a GitHub repository.  

3. **Create a Streamlit App**:  
    - Visit [Streamlit](https://streamlit.io).  
    - Click **Create App**.  
    - Select your GitHub repository.  
    - 🎉 Streamlit will generate a public URL for your chatbot.  

---

### 🌟 Enjoy chatting with the bot! 🤖💬
