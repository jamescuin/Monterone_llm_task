import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import random
import tiktoken

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

### Utility Functions ###

def preview_formatted_prompt(question, retriever, prompt_template):
    pages = retriever.invoke(question)
    context = "\n\n".join([page.page_content for page in pages])
    formatted_prompt = prompt_template.format(context=context, question=question)
    return formatted_prompt

def count_tokens(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def ask_llm(question: str, llm_chain, retriever, verbose: bool = True) -> tuple:
    pages = retriever.invoke(question)
    context = "\n\n".join([page.page_content for page in pages])
    question_and_answer = llm_chain.invoke(question)
    return context, question_and_answer

#################################################################################

# Create loader to read and split `.pdf` documents
loader = PyMuPDFLoader('How ASML took over the chipmaking chessboard (MIT 010424).pdf')

# Load and split `.pdf` document by pages
pages = loader.load_and_split()

# Utilise Hugging Face vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings, for each page, using ChromaDB
stored_embeddings = Chroma.from_documents(pages, embeddings, collection_name='asml_report')

retriever = stored_embeddings.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define model hyperparameters to use
LLM_HYPERPARAMETERS = {
    'MODEL_NAME': 'google/flan-t5-large',
    # 'MODEL_NAME': 'meta-llama/Llama-2-7b-chat-hf'
}

# Wrap the model with HuggingFaceHub to use LangChain
llm = HuggingFaceHub(
    repo_id=LLM_HYPERPARAMETERS['MODEL_NAME'],
    huggingfacehub_api_token="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"
    )

# Create custom prompt template
template = """You are a world-leading financial analyst. Please provide a comprehensive and illuminating answer to the following question based on the given context. DO NOT BE LAZY!

Context:
{context}

Q: {question}

A:"""

zero_shot_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create RetrievalQA chain
qa_chain_zero_shot = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": zero_shot_prompt_template},
    verbose=True
)

# Streamlit app
st.title('Zero-shot Prompting (Open Source)')

question = st.text_input('Enter your question:')
if st.button('Get Answer'):
    context, result = ask_llm(question, qa_chain_zero_shot, retriever, verbose=False)
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