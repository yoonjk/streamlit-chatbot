## install streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit python_dotenv

## export package libraries
pip freeze > requirements.txt

pip install -U ibm-watson-machine-learning
pip install langchain
pip install pypdf
pip install chromadb
pip install unstructured
pip install sentence_transformers
pip install langchain_community