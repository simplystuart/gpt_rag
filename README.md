# GPT RAG

Chat with openAI using code as context.

## Setup

1. Run the following:

```bash
git clone https://github.com/simplystuart/gpt_rag.git
cd gpt_rag
pip install -r requirements.txt
cp .env.example .env
```

2. Add `OPENAI_API_KEY` to `.env`

[!NOTE]
The embedder script will estimate usage and prompt for approval before using APIs

## Usage

### Step 1: Embed Repo(s)

```bash
python embedder.py [RELATIVE PATH TO REPO]
```

### Step 2: Run Server

```bash
python server.py
```

[!IMPORTANT]
To use ChatGPT with locally running server use `ngrok`.

### Step 3: Create Custom GPT

Create an action to your `ngrok` endpoint with the schema in `action_schema.json` (replace `YOUR_NGROK_ENDPOINT`).
### Step 4

Chat with your custom GPT
