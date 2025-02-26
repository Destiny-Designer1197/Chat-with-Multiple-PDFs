import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks

    
def get_vectorstore(text_chunks):
    # embs = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embs = OpenAIEmbeddings()
    embs = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
   
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embs)
    return vector_store

def get_conv_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conv_chain



  


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    
    if "conv" not in st.session_state:
        st.session_state.conv = None

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get raw text from all the pdfs
                raw_text =  get_pdf_text(pdf_docs)
               
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store after storing them as embs
                vector_store = get_vectorstore(text_chunks)

                st.session_state.conv = get_conv_chain(vector_store)


if __name__ == '__main__':
    main()