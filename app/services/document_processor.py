import os
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(file_path: str) -> List[Document]:
    # Load raw pages/text
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    
    docs = loader.load()
    # Split into 1,000‑char chunks with 200‑char overlap 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)