import streamlit as st 
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

# Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT1"] = "RAG_QAbot"

# Initialize model
llm = ChatOllama(model="llama3.1")

# Prompt template 
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response possible based on the question.
    If the information is not in the provided context, just say I don't know.
    <Context>
    {context}
    </Context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        print("Creating embeddings and vector store.")
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        st.session_state.loader = PyPDFDirectoryLoader("pdf")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        
        if not st.session_state.docs:
            st.warning("No documents found in the specified directory.")
            return
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Create the Chroma vector store
        st.session_state.vectors = Chroma.from_documents(
            documents=st.session_state.final_documents,
            embedding=st.session_state.embeddings,
            persist_directory=os.path.join(os.getcwd(), "chroma")  # Ensure the correct persist directory
        )
        
        # Persist the vector store
        st.session_state.vectors.persist()  
        print("Vector store created and persisted with", len(st.session_state.final_documents), "documents.")
    else:
        print("Using existing vector store.")

user_prompt = st.text_input("Ask a question from the documents")

if st.button("Document embedding"):
    create_vector_embeddings()
    st.write("Vector db is ready")

import time

if user_prompt:
    # Check the state of the vectors before processing the input
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.warning("Vector database is not initialized. Please click 'Document embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}   
        )
        
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write(f"Response time: {time.process_time() - start}")
        
        # Debug: Print the entire response
        #print("Response:", response)

        
        if 'answer' in response:
            st.write(response['answer'])
        else:
            st.warning("No answer found for your question.")
        
        # With a Streamlit expander 
        with st.expander("Document Similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("-------------------")
