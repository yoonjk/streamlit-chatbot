import streamlit as st
import requests, json
import tempfile, os
from chroma import create_llm, create_embedding 
st.title("IBM watsonx.ai Prompt Lab!")

# Function for call serving of LLM 
@st.cache_resource(show_spinner=False)
def load_data(file_path, credentials, project_id):

    recursive_index = create_embedding(file_path, credentials, project_id)
    
    return recursive_index
                
def watsonx_ai_api(prompts, chain):

    response_text = chain.invoke(prompts)

    return response_text

with st.sidebar:
    if ('API_KEY' in st.secrets) and ('API_URL' in st.secrets) and ('PROJECT_ID' in st.secrets):
        st.success('Credentials already provided!', icon='✅')
        watsonx_api_key = st.secrets['API_KEY']
        watsonx_api_url = st.secrets['API_URL']
        watsonx_project_id = st.secrets['PROJECT_ID']
    else:
        watsonx_api_key = st.text_input('Enter API Key:')
        watsonx_api_url = st.text_input('Enter API Url:')
        watsonx_project_id = st.text_input('Enter Project_Id:')
        
        
    if not (watsonx_api_key and watsonx_api_url and watsonx_project_id):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
        
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process")
        
        if st.button("Process"):
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, pdf_docs.name)
            with open(path, "wb") as f:
                f.write(pdf_docs.getvalue())
            # get recursive index
            credentials = {
                "apikey": watsonx_api_key,
                "url" : watsonx_api_url
            }
            

            recursive_index = load_data(path, credentials, watsonx_project_id)
            print("recursive_index:", recursive_index)
            
            st.session_state.conversation = recursive_index
            st.session_state.credentials = credentials 
            
            print("st.session_sate:", st.session_state.conversation)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = watsonx_ai_api(prompt, st.session_state.conversation) 
            st.write(response) 
    st.session_state.messages.append({"role": "assistant", "content": response})