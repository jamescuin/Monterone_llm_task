import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tiktoken
from langchain import PromptTemplate
import random
from langchain.callbacks import get_openai_callback

### Utility Functions ###

def preview_formatted_prompt(question, retriever, prompt_template):
    pages = retriever.invoke(question)
    context = "\n\n".join([page.page_content for page in pages])
    formatted_prompt = prompt_template.format(context=context, question=question)
    return formatted_prompt

def count_tokens(text, model_name):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    return len(tokens)

def ask_llm(question: str, llm_chain, retriever, verbose: bool = True) -> tuple:
    with get_openai_callback() as cb:
        pages = retriever.invoke(question)
        context = "\n\n".join([page.page_content for page in pages])
        question_and_answer = llm_chain.invoke(question)
        if verbose: 
            st.write(f"Total Tokens: {cb.total_tokens}")
            st.write(f"Prompt Tokens: {cb.prompt_tokens}")
            st.write(f"Completion Tokens: {cb.completion_tokens}")
            st.write(f"Total Cost (USD): ${cb.total_cost}")
    return context, question_and_answer

#################################################################################

# Load environment variables from `.env` file securely
load_dotenv()

# Set OpenAI API key for API access
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Create loader to read and split `.pdf` documents
loader = PyMuPDFLoader('How ASML took over the chipmaking chessboard (MIT 010424).pdf')

# Load and split `.pdf` document by pages
pages = loader.load_and_split()

# Utilise OpenAI vector embeddings
embeddings = OpenAIEmbeddings()

# Store embeddings, for each page, using ChromaDB
stored_embeddings = Chroma.from_documents(pages, embeddings, collection_name='asml_report')

retriever = stored_embeddings.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define (OpenAI) model hyperparameters to use (See pricing above!)
LLM_HYPERPARAMETERS = {
    'MODEL_NAME': 'gpt-4-turbo',
    'CHAT_MODEL': True,
    'TEMPERATURE': 0,
    'VERBOSE': True,
    'MAX_OUTPUT_TOKENS': -1 # Set to -1 for no limit
}

# Instantiate (Chat) OpenAI LLM
if LLM_HYPERPARAMETERS['CHAT_MODEL']:
    llm = ChatOpenAI(
        model=LLM_HYPERPARAMETERS['MODEL_NAME'], 
        temperature=LLM_HYPERPARAMETERS['TEMPERATURE'], 
        verbose=LLM_HYPERPARAMETERS['VERBOSE']
    )
else:
    llm = OpenAI(
        model=LLM_HYPERPARAMETERS['MODEL_NAME'], 
        temperature=LLM_HYPERPARAMETERS['TEMPERATURE'], 
        verbose=LLM_HYPERPARAMETERS['VERBOSE']
    )

# Create a custom prompt template
template = """You are a world-leading financial analyst. Please provide a comprehensive and illuminating answer to the following question based on the given context. DO NOT BE LAZY!

Context:
{context}

Q: {question}

A:"""

zero_shot_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create a RetrievalQA chain
qa_chain_zero_shot = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": zero_shot_prompt_template},
    verbose=True
)

# Streamlit app
st.title('Zero-shot Prompting (Closed Source)')

question = st.text_input('Enter your question:')
if st.button('Get Answer'):
    context, result = ask_llm(question, qa_chain_zero_shot, retriever, verbose=False)
    num_tokens = count_tokens(result['result'], LLM_HYPERPARAMETERS['MODEL_NAME'])
    st.write("### Answer")
    st.write(result['result'])

# Streamlit expander for Document Similarity Search
with st.expander('Document Similarity Search'):
    search_prompt = st.text_input('Enter search prompt for similarity search:')
    if st.button('Search'):
        search_results = stored_embeddings.similarity_search_with_score(search_prompt)
        if search_results:
            st.write("### Most Similar Document")
            st.write(search_results[0][0].page_content)
        else:
            st.write("No similar documents found.")
