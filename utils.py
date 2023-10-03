# from sentence_transformers import SentenceTransformer
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pinecone
import openai
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings

openai.api_key = st.secrets['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'])
# model = SentenceTransformer('all-MiniLM-L6-v2')

llm = OpenAI(temperature=0, openai_api_key=st.secrets['OPENAI_API_KEY'])
chain = load_qa_chain(llm, chain_type="stuff")

pinecone.init(api_key=st.secrets['PINECONE_API_KEY'], environment=st.secrets['PINECONE_API_ENV'])
index = pinecone.Index('amotions-data-index')

loader = PyPDFDirectoryLoader("./data")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
docsearch = Chroma.from_documents(texts, embeddings)
def find_match(input_text):
    
    docs = docsearch.similarity_search(input_text)
    response = chain.run(input_documents=docs, question=input_text)

    return response

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string