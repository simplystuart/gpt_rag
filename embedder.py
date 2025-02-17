import os
import subprocess
import tiktoken

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_files():
    # Get files in the git repository
    command = ["git", "ls-files"]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True)

    return result.stdout.strip().split("\n")


def build_docs(files):
    documents = []

    for file in files:
        # Read text files; skip others
        try:
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"text": f.read(), "source": file})
                print(f"✅ Added {file}")
        except Exception as e:
            print(f"❌ Skipped {file}: {e}")

    return documents


def chunk_docs(docs):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)

    for doc in docs:
        splits = text_splitter.split_text(doc["text"])
        chunks.extend([{"text": s, "source": doc["source"]} for s in splits])

    return chunks


def estimate_usage(token_cost, chunks):
    # Tokenizer for OpenAI embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    total_tokens = sum(
        len(tokenizer.encode(chunk["text"])) for chunk in chunks
    )

    estimated_cost = (total_tokens / 1000000) * token_cost

    print(f"\n🔢 Estimated Token Usage: {total_tokens} tokens")
    print(f"💰 Estimated Cost: ${estimated_cost:.6f}")


def embed_chunks(key, model, chunks):
    # Store chunks from OpenAI embeddings in FAISS
    embeddings_model = OpenAIEmbeddings(model=model, openai_api_key=key)
    vector_db = FAISS.from_texts([chunk["text"]
                                 for chunk in chunks], embeddings_model)
    vector_db.save_local("faiss_code_index")


def main():
    # Load config
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL_ID")
        token_cost = float(os.getenv("OPENAI_EMBEDDINGS_MODEL_COST"))
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Prepare for embedding via OpenAI
    files = get_files()
    docs = build_docs(files)
    chunks = chunk_docs(docs)
    estimate_usage(token_cost, chunks)

    # Ask user to proceed with embedding
    proceed = input("\nProceed with embedding? (y/n): ")

    if proceed.lower() != "y":
        print("⛔︎ Embedding skipped")
    else:
        embed_chunks(api_key, model, chunks)
        print("🎉 Embedding finished")


if __name__ == "__main__":
    main()
