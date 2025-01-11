import io

import chardet
import fitz
import streamlit as st
from decouple import config
from openai import OpenAI

API_KEY = config("DEEPSEEK_API_KEY", default="")

# Default context (first priority)
DEFAULT_CONTEXT = "You are a helpful assistant. Provide clear and concise answers. If you are writing a code make sure to summarize and provide a concise code, optimize the code output to the smartest and the shortest way with better readability and functionality"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "file_context" not in st.session_state:
    st.session_state["file_context"] = ""
if "default_prompt" not in st.session_state:
    st.session_state["default_prompt"] = ""


def call_deepseek_api(messages, streaming=True):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, stream=streaming
        )
        return response
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return None


def read_pdf(file):
    try:
        pdf_bytes = file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def read_file(file):
    try:
        if file.type == "application/pdf":
            return read_pdf(file)
        else:
            raw_data = file.read()
            encoding_info = chardet.detect(raw_data)
            encoding = encoding_info["encoding"]
            return raw_data.decode(encoding, errors="replace")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# Streamlit UI
st.title("Document Chat Interface")

# Sidebar settings for default prompt
st.sidebar.title("Settings")
default_prompt = st.sidebar.text_area(
    "Set Default Prompt", value=st.session_state["default_prompt"]
)
if st.sidebar.button("Save Default Prompt"):
    st.session_state["default_prompt"] = default_prompt
if st.sidebar.button("Clear Default Prompt"):
    st.session_state["default_prompt"] = ""

# File upload section
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    file_content = read_file(uploaded_file)
    st.session_state["file_context"] = file_content
    if file_content:
        st.success("File successfully loaded!")
        if st.checkbox("Show file content"):
            st.text_area("File Content", file_content, height=200)

user_input = st.text_area("Enter your prompt")

# Model selection
model_choice = st.selectbox("Choose a model", ["deepseek-chat"])

if st.button("Submit"):
    if not user_input:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Processing your request. This may take a few seconds..."):
            # Prepare messages with default context as first priority
            system_content = DEFAULT_CONTEXT
            if st.session_state["default_prompt"]:
                system_content += (
                    f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"
                )
            if st.session_state["file_context"]:
                file_context = st.session_state["file_context"]
                if len(file_context) > 5000:  # Truncate large files
                    file_context = file_context[:5000] + "... (truncated)"
                system_content += f"\n\nFile Context:\n{file_context}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input},
            ]

            response = call_deepseek_api(messages=messages)  # streamin default to True

            if response:
                reply_container = st.empty()
                full_response = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        reply_container.markdown(
                            full_response
                        )  # Update the response incrementally

                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                st.warning("No response received from the API.")

st.sidebar.title("Chat History")
for i, message in enumerate(st.session_state["chat_history"]):
    role = message["role"]
    content = message["content"]
    st.sidebar.write(f"**{role.capitalize()}:**")
    st.sidebar.write(content)
    if i < len(st.session_state["chat_history"]) - 1:
        st.sidebar.write("---")
