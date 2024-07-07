
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
# langchain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

def create_llm(credentials, project_id):
  generate_params = {
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.DECODING_METHOD: "greedy",
    GenParams.REPETITION_PENALTY: 1
  }

  model = Model(
      model_id=ModelTypes.LLAMA_2_70B_CHAT, 
      credentials=credentials,
      params=generate_params,
      project_id=project_id
  )

  # LangChainで使うllm
  llm = WatsonxLLM(model=model)
  
  return llm

def create_embedding(file_path, creds, project_id):
  loader = PyPDFLoader(file_path)

  print("===========================")
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0);
  
  index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter = text_splitter 
    ).from_loaders([loader])

  # Initialize watsonx google/flan-ul2 model
  params = {
      GenParams.DECODING_METHOD: "sample",
      GenParams.TEMPERATURE: 0.2,
      GenParams.TOP_P: 1,
      GenParams.TOP_K: 100,
      GenParams.MIN_NEW_TOKENS: 50,
      GenParams.MAX_NEW_TOKENS: 300
  }
  model = Model(
      model_id="meta-llama/llama-2-70b-chat",
      params=params,
      credentials=creds,
      project_id=project_id
  ).to_langchain()
  
  
  retriever = index.vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={'score_threshold': 0.5}
  )
  
  # Init RAG chain
  chain = RetrievalQA.from_chain_type(llm=model, 
                                      chain_type="stuff", 
                                      retriever=index.vectorstore.as_retriever(), 
                                      return_source_documents=True,
                                      input_key="question")

  return chain 