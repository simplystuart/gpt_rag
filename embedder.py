import openai
import os
import subprocess
import sys
import tenacity
import tiktoken
import time

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_files():
    # Get files in the git repository
    command = ["git", "ls-files"]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True)

    return result.stdout.strip().split("\n")


def map_file_metadata(file):
    return {
        "source": file,
        "filename": os.path.basename(file),
        "file_extension": os.path.splitext(file)[1],
        "file_path": os.path.dirname(os.path.abspath(file)),
        "last_modified": os.path.getmtime(file),
    }


def build_docs(files):
    documents = []
    for file in files:
        # Read text files; skip others
        try:
            with open(file, "r", encoding="utf-8") as f:
                metadata = map_file_metadata(file)
                documents.append({"text": f.read(), "metadata": metadata})
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
        # Preserve all metadata for each chunk
        chunks.extend([{"text": s, "metadata": doc["metadata"]}
                      for s in splits])
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

    # Calculate time estimation based on rate limits
    # text-embedding-3-large limits: 1,000,000 TPM, 3,000 RPM
    minutes_required_tokens = total_tokens / 1000000
    requests_count = len(chunks)  # Worst case: one request per chunk
    minutes_required_requests = requests_count / 3000

    estimated_minutes = max(minutes_required_tokens, minutes_required_requests)
    print(f"⏱️ Estimated processing time: {
          estimated_minutes:.2f} minutes (based on rate limits)")

    return total_tokens


class RateLimitedOpenAIEmbeddings(Embeddings):
    def __init__(self, base_embeddings, max_rpm=2900, max_tpm=950000):
        """
        Wrapper for OpenAIEmbeddings that manages rate limits

        Args:
            base_embeddings: The OpenAIEmbeddings instance
            max_rpm: Maximum requests per minute (keep below the 3000 limit)
            max_tpm: Maximum tokens per minute (keep below the 1,000,000 limit)
        """
        self.embeddings = base_embeddings
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.request_timestamps = []
        self.token_counts = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _wait_for_rate_limit(self, token_count):
        """Wait if necessary to stay within rate limits"""
        current_time = time.time()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if current_time - ts < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]

        # Check if we would exceed request rate limit
        if len(self.request_timestamps) >= self.max_rpm:
            # Make sure there's at least one timestamp before accessing it
            if self.request_timestamps:
                sleep_time = 60 - \
                    (current_time - self.request_timestamps[0]) + 0.1
                if sleep_time > 0:
                    print(f"⏳ Rate limit approaching. Waiting {
                          sleep_time:.2f}s for request limit...")
                    time.sleep(sleep_time)

        # Check if we would exceed token rate limit
        current_tokens = sum(self.token_counts) if self.token_counts else 0
        if current_tokens + token_count > self.max_tpm:
            # Make sure there's at least one timestamp before accessing it
            if self.request_timestamps:
                sleep_time = 60 - \
                    (current_time - self.request_timestamps[0]) + 0.1
                if sleep_time > 0:
                    print(f"⏳ Rate limit approaching. Waiting {
                          sleep_time:.2f}s for token limit...")
                    time.sleep(sleep_time)

        # Update tracking after potential waiting
        self.request_timestamps.append(time.time())
        self.token_counts.append(token_count)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.APIError
        )),
        reraise=True
    )
    def embed_documents(self, texts):
        """Rate-limited and retry-enabled embed_documents"""
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
        self._wait_for_rate_limit(total_tokens)

        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                # Handle token limit error differently
                # Don't retry this specific error
                raise ValueError(f"Token limit exceeded with {
                                 total_tokens} tokens. Try smaller batches.")
            raise  # Re-raise for other errors to trigger retry

    def embed_query(self, text):
        """Rate-limited and retry-enabled embed_query"""
        token_count = len(self.tokenizer.encode(text))
        self._wait_for_rate_limit(token_count)
        return self.embeddings.embed_query(text)


def embed_chunks(key, model, chunks, codebase):
    # Create base embeddings model
    base_embeddings = OpenAIEmbeddings(model=model, openai_api_key=key)

    # Wrap with rate limiting
    embeddings_model = RateLimitedOpenAIEmbeddings(base_embeddings)

    # Maximum tokens per request (leaving some buffer)
    MAX_TOKENS_PER_BATCH = 500000

    # Tokenizer for counting tokens
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Prepare batches
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk["text"]))

        # If adding this chunk would exceed the limit, start a new batch
        if current_tokens + chunk_tokens > MAX_TOKENS_PER_BATCH:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens

    # Add the last batch if it has items
    if current_batch:
        batches.append(current_batch)

    print(f"Split into {len(batches)} batches for processing")

    # Process each batch and merge results
    vector_db = None

    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        texts = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        try:
            # If this is the first batch, create a new FAISS index
            if i == 0:
                vector_db = FAISS.from_texts(
                    texts, embeddings_model, metadatas=metadatas)
            else:
                # For subsequent batches, add to the existing index
                vector_db.add_texts(texts, metadatas=metadatas)
            print(f"✅ Successfully processed batch {i+1}")
        except Exception as e:
            print(f"❌ Error processing batch {i+1}: {e}")
            # If we've created a vector_db, save what we have so far
            if vector_db is not None:
                vector_db.save_local(f"./data/{codebase}_partial")
                print(f"💾 Saved partial results to ./data/{codebase}_partial")
            raise

    # Save the final vector store
    vector_db.save_local(f"./data/{codebase}")
    print(f"💾 Saved complete vector database to ./data/{codebase}")


def main():
    # Load config
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL_ID",
                          "text-embedding-3-large")
        token_cost = float(
            os.getenv("OPENAI_EMBEDDINGS_MODEL_COST", "0.00013"))
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Check for codebase path
    if (len(sys.argv) < 2):
        print("Usage: python embedder.py <relative-path-to-codebase>")
        return
    else:
        path = sys.argv[1]

    # Prepare for embedding via OpenAI
    dir = os.getcwd()
    os.chdir(path)
    codebase = os.path.basename(os.getcwd())
    files = get_files()
    docs = build_docs(files)
    chunks = chunk_docs(docs)
    total_tokens = estimate_usage(token_cost, chunks)
    os.chdir(dir)

    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    # Ask user to proceed with embedding
    proceed = input(f"\nProceed with embedding {
                    len(chunks)} chunks ({total_tokens} tokens)? (y/n): ")
    if proceed.lower() != "y":
        print("⛔︎ Embedding skipped")
    else:
        try:
            embed_chunks(api_key, model, chunks, codebase)
            print("🎉 Embedding finished successfully")
        except Exception as e:
            print(f"❌ Error during embedding process: {e}")
            print("Check for partial results in the data directory.")


if __name__ == "__main__":
    main()
