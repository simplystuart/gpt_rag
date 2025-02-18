import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def ask_codebase(api_key, model, prompt, top_k=5):
    # Load codebase embeddings from FAISS
    vector_db = FAISS.load_local(
        "faiss_code_index",
        OpenAIEmbeddings(openai_api_key=api_key, model=model),
        allow_dangerous_deserialization=True
    )

    results = vector_db.similarity_search(prompt, top_k=top_k)

    return "".join([r.page_content for r in results])


def ask_ai(api_key, embeddings_model, chat_model, prompt):
    client = OpenAI(api_key=api_key)
    context = ask_codebase(api_key, embeddings_model, prompt)

    # Create chat messages with context
    content = '''\
        Here is relevant code from our codebase:
        {context}
        Answer this question based on the code:
        {prompt}
    '''.format(context=context, prompt=prompt)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": content}
    ]

    # Get response from chat model
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


def main():
    # Load config
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        chat_model = os.getenv("OPENAI_CHAT_MODEL_ID")
        embeddings_model = os.getenv("OPENAI_EMBEDDINGS_MODEL_ID")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Start chat
    print("\n")
    print("👋 Hi there! I am a helpful coding assistant!")
    print("You can ask me questions about our codebase.")
    print("When you are all done, type 'bye!' to exit.")

    # loop until user exits
    while True:
        # Get user input
        prompt = input("\nAsk me a question:\n")
        print("\n")

        if prompt.lower() == "bye!":
            # Exit if user types "exit"
            print("Goodbye!")
            break
        else:
            # Print response
            print(ask_ai(api_key, embeddings_model, chat_model, prompt))


if __name__ == "__main__":
    main()
