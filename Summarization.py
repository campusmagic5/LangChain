import streamlit as st
import os
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

# Function to upload PDF files to the 'uploads' directory
def upload_files(directory):
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(directory, file.name), 'wb') as f:
                f.write(file.read())
        st.success("Files uploaded successfully!")

# Function to handle different button clicks
def handle_button_click(selected_option):
    loader = PyPDFLoader("folder/file.pdf")
    docs = loader.load()
    
    llm = ChatOpenAI(openai_api_key=openai_api_key,temperature=0, model_name="gpt-3.5-turbo-1106")
    if selected_option == 'stuff':
        chain = load_summarize_chain(llm, chain_type="stuff")
        result_text = chain.run(docs)
        return result_text
    elif selected_option == 'map_reduce':
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        result_text = chain.run(docs)
        return result_text
    elif selected_option == 'refine':
        chain = load_summarize_chain(llm, chain_type="refine")
        result_text = chain.run(docs)
        return result_text
    else:
        return "Select an option from the dropdown."

# Main Streamlit app
def main():
    st.title("PDF Dashboard")

    # Create 'uploads' directory if not exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Feature 1: Upload PDF files
    upload_files('uploads')

    # Feature 2: Dropdown with three buttons
    selected_option = st.selectbox("Select an option:", ['stuff', 'map_reduce', 'refine'])

    # Feature 3: Submit button to display text based on the selected option
    if st.button("Submit"):
        
        result_text = handle_button_click(selected_option)
        st.write(result_text)

if __name__ == "__main__":
    main()
