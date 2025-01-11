import streamlit as st
import requests
from decouple import config
import textract

# Load environment variables from .env file
API_KEY = config('DEEPSEEK_API_KEY', default='')

# Define the API endpoints
DEEPSEEK_V3_URL = config("DEEPSEEK_V3_URL", default="https://api.deepseek.com/api/v1/chat")
DEEPSEEK_V3_32K_URL = config("DEEPSEEK_V3_32K_URL", default="https://api.deepseek.com/api/v1/chat-32k")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'default_prompt' not in st.session_state:
    st.session_state['default_prompt'] = ''

# Function to call the DeepSeek API
def call_deepseek_api(api_url, messages, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "messages": messages
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")

# Sidebar for settings
st.sidebar.title("Settings")

# Option to set or clear default prompt
default_prompt = st.sidebar.text_area("Set Default Prompt", value=st.session_state['default_prompt'])
if st.sidebar.button("Save Default Prompt"):
    st.session_state['default_prompt'] = default_prompt
if st.sidebar.button("Clear Default Prompt"):
    st.session_state['default_prompt'] = ''

# Option to skip default prompt
skip_prompt = st.sidebar.checkbox("Skip Default Prompt")

# File upload section
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
document_context = ""
if uploaded_file:
    document_context = textract.process(uploaded_file.read()).decode('utf-8')

# Construct system message
system_message = ""
if not skip_prompt and st.session_state['default_prompt']:
    system_message += f"You are a file reader. Answer based only on the following context:\n\n{st.session_state['default_prompt']}\n\n"
if document_context:
    system_message += f"{document_context}\n\n"
system_message += "If you can't answer, say 'I did not find anything.'"

# User input
user_input = st.text_area("Enter your prompt", "")

# Model selection
model_choice = st.selectbox("Choose a model", ["DeepSeek-V3-32k"])

# Submit button
if st.button("Submit"):
    if not user_input:
        st.warning("Please enter a prompt.")
    elif not API_KEY:
        st.warning("Please set your DEEPSEEK_API_KEY in the .env file.")
    else:
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_input})
        
        # Select API URL based on model choice
        if model_choice == "DeepSeek-V3":
            api_url = DEEPSEEK_V3_URL
        else:
            api_url = DEEPSEEK_V3_32K_URL
        
        # Call the API
        response = call_deepseek_api(api_url, messages, API_KEY)
        
        # Display response
        if response:
            if "choices" in response and len(response["choices"]) > 0:
                reply = response["choices"][0]["message"]["content"]
                st.text_area("Model Response", reply, height=400)
                # Append to chat history
                st.session_state['chat_history'].append(f"User: {user_input}")
                st.session_state['chat_history'].append(f"Model: {reply}")
            else:
                st.warning("No response received from the API.")

# Chat History Display in Sidebar
st.sidebar.title("Chat History")
if 'chat_history' in st.session_state:
    for message in reversed(st.session_state['chat_history']):
        st.sidebar.write(message)

# Clear History Button
if st.sidebar.button("Clear History"):
    st.session_state['chat_history'] = []
    st.session_state['document_context'] = ''