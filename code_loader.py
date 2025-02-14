import os
import subprocess

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

def get_files():
    command = ["git", "ls-files"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    return result.stdout.strip().split("\n")

def build_docs(files):
    documents = []

    for file in files:
        print(file)

        if file.endswith((".avif", ".ico", ".jpg", ".png", ".svg", ".woff")):
            continue

        with open(file, "r", encoding="utf-8") as f:
            documents.append({"text": f.read(), "source": file})

    return documents

def chunk_docs(docs):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for doc in docs:
        splits = text_splitter.split_text(doc["text"])
        chunks.extend([{"text": s, "source": doc["source"]} for s in splits])

    return chunks

def embed_chunks(chunks, openai_api_key):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_db = FAISS.from_texts([chunk["text"] for chunk in chunks], embeddings_model)
    vector_db.save_local("faiss_code_index")

def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    files = get_files()
    docs = build_docs(files)
    chunks = chunk_docs(docs)
    embed_chunks(chunks, openai_api_key)

if __name__ == "__main__":
    main()
