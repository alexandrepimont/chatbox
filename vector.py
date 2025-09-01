from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
def load_documents(data_folder):
    document_loader = PyPDFDirectoryLoader(data_folder)
    return document_loader.load()

print(f"Loading documents from {DATA_PATH}...")
print(load_documents(DATA_PATH)[1])


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200,
                                                   length_function=len,
                                                   is_separator_regex=False)
    return text_splitter.split_documents(documents)

documents = load_documents(DATA_PATH)
chunks = split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    db.add_documents(chunks)
    db.persist

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3.2")
    return embeddings

last_page_id = None
current_chunk_index = 0
for chunk in chunks:
    if current_page_id == last_page_id:
        current_chunk_index += 1

    else:
        current_chunk_index = 0

if __name__ == "__main__":
    
    from IPython import embed
    embed()