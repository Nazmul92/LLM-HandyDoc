import os
import streamlit as st
import pickle
import time
import tempfile
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from docx import Document  # For .docx files
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# App Title and Description
st.title("HandyDoc")
st.subheader("LLM-based Knowledge Extraction Tool")

# Sidebar for File Upload
st.sidebar.title("Upload Files")
st.sidebar.markdown("Upload files (.pdf, .txt, .doc) to extract knowledge and get your answer.")
uploaded_files = st.sidebar.file_uploader("Choose Files", type=["pdf", "txt", "doc", "docx"], accept_multiple_files=True)
process_files_clicked = st.sidebar.button("Process Files")

# Main Content
main_placeholder = st.empty()
file_path = "faiss_store_hf.pkl"
llm = ChatGroq(temperature=0.9, max_tokens=500)

# Function to load documents based on file type
def load_document(file_path, file_extension):
    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_extension == "txt":
        loader = TextLoader(file_path)
        return loader.load()
    elif file_extension in ["doc", "docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    else:
        raise ValueError("Unsupported file type")

# Process Files
if process_files_clicked and uploaded_files:
    with st.spinner("Processing files..."):
        # Load data from uploaded files
        main_placeholder.text("Data Loading...")
        docs = []
        
        for uploaded_file in uploaded_files:
            # Get file extension
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name  # Get the temporary file path
            
            # Load the file based on its type
            try:
                docs.extend(load_document(tmp_file_path, file_extension))
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
            finally:
                os.remove(tmp_file_path)  # Clean up the temporary file

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter Starting...")
        docs = text_splitter.split_documents(docs)

        # Create embeddings and save to FAISS index
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        vectorstore_hf = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_hf, f)

        st.success("Files processed successfully!")

# Query Section
query = st.text_input("Ask a Question")

if query:
    if os.path.exists(file_path):
        with st.spinner("Searching answers..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                
                # Display Answer
                st.markdown("### Answer")
                st.info(result["answer"])

                # Display Sources
                sources = result.get("sources", "")
                if sources:
                    st.markdown("### Sources")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(f"- {source}")
    else:
        st.warning("Please upload and process files first.")