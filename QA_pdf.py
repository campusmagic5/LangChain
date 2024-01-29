import os
from pathlib import Path
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader,PyPDFLoader


# Streamlit app
def main():
    st.title("Upload PDF and ask your question")

    # Create a directory named 'uploads' if it doesn't exist
    upload_directory = "./uploads/"
    Path(upload_directory).mkdir(parents=True, exist_ok=True)
    
    # Upload multiple PDF files
    st.header("Upload PDF Files")
    uploaded_files = st.file_uploader("Choose multiple PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader")

    # Display uploaded PDF files and save them in the 'uploads' directory
    if uploaded_files:
        st.success("Files successfully uploaded!")
        for file in uploaded_files:
            file_path = os.path.join(upload_directory, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.write(f"Uploaded: {file.name}")

    # Text input and submit button
    st.header("Text Input and Output")
    user_input = st.text_input("Enter text:")
    if st.button("Submit"):
        loader = DirectoryLoader('./uploads/', glob="./*.pdf", loader_cls=PyPDFLoader)

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)
        persist_directory = 'vectordb'

        ## We are using OpenAI embeddings
        embedding = OpenAIEmbeddings(openai_api_key="")

        vectorstore = Chroma.from_documents(documents=texts, 
                                         embedding=embedding,
                                         persist_directory=persist_directory)
        retriever = vectorstore.as_retriever()
        # Chain
        chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=""), 
                                          chain_type="stuff", 
                                          retriever=retriever, 
                                          return_source_documents=True)
       
        llm_response = chain(user_input)
        st.write(llm_response['result'])
        st.write(llm_response['source_documents'][0])

    

# Run the Streamlit app
if __name__ == "__main__":
    main()
