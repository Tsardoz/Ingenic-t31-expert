import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import glob
#Huggingface is junk
#from langchain.llms import HuggingFaceHub
#from langchain.embeddings import HuggingFaceInstructEmbeddings

def get_documents_from_folder(folder_path):
    file_types = ['*.pdf', '*.txt', '*.py', '*.c', '*.h']  # Add or remove file types as needed
    docs = []
    for file_type in file_types:
        docs.extend(glob.glob(os.path.join(folder_path, file_type)))
    return docs

def get_text(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith('.pdf'):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            # read text from a .c, .txt, .py or .h file etc
            text_content = doc.read().decode('utf-8')
            text += text_content
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI(model_name="gpt-4-0125-preview")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with an Ingenic T31 expert :books:")
    user_question = st.text_input("Ask a question about the T31:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        # Automatically get documents from the 'data' folder
        folder_path = '../data'
        docs_paths = get_documents_from_folder(folder_path)
        # Convert file paths to file-like objects for further processing
        docs = [open(doc_path, 'rb') for doc_path in docs_paths]
        # Place the "Process" button in the sidebar
        if st.button("Process", key="process_button"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_text(docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.success("Files processed")
        # Print the names of the documents in the sidebar
        for doc_path in docs_paths:
            st.write(os.path.basename(doc_path))  # This prints the file name


if __name__ == '__main__':
    main()