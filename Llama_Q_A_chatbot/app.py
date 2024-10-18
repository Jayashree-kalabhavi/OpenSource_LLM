import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Streamlit_Chatbot"

# Prompt template
System_prompt = "You are a useful assistant. Respond to user queries."
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", System_prompt),
        ("user", "question:{question}"),
    ]
)

from transformers import GPT2Tokenizer  # Assuming you have a tokenizer compatible with the model

# Function to count tokens
def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Replace with relevant tokenizer
    tokens = tokenizer.encode(text)
    return len(tokens)

# Modify your generate_response function
def generate_response(question, llm, temperature, max_tokens):
    # Debugging: print parameters to verify correctness
    st.write(f"Using model: {llm}, Temperature: {temperature}, Max Tokens: {max_tokens}")
    
    try:
        llm_model = ChatOllama(model=llm,
                            temperature=temperature, 
                            num_predict=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm_model | output_parser
        answer = chain.invoke({'question': question})
        
        # Count tokens in the response
        token_count = count_tokens(answer)
        st.write(f"Token count of the response: {token_count}")
        
        return answer
    except Exception as e:
        # Debugging: Print error if something goes wrong
        st.write(f"Error generating response: {str(e)}")
        return "Sorry, something went wrong."


# Streamlit configuration
st.title("Q&A Chatbot with Llama3.1")

# Sidebar for settings
st.sidebar.title("Settings")
# Dropdown to select models
llm = st.sidebar.selectbox("Select an open-source model", ["llama3.1", "llama3.1:70b", "llama3.1:8b-instruct-fp16"])

# Slider to select temperature and max_tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("How can I help you today?")
user_input = st.text_input("You:")

# If user input is provided, generate and display response
if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please ask a question.")
