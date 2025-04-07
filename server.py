import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
embeddings_model = os.getenv("OPENAI_EMBEDDINGS_MODEL_ID")


def ask_codebase(codebase, prompt, top_k=5, filter=None):
    results = []

    vector_db = FAISS.load_local(
        f'./data/{codebase}',
        OpenAIEmbeddings(openai_api_key=api_key, model=embeddings_model),
        allow_dangerous_deserialization=True
    )

    documents = vector_db.similarity_search(prompt, top_k=top_k, filter=filter)

    for doc in documents:
        results.append({
            'page_content': doc.page_content,
            'metadata': doc.metadata,
        })

    return results


@app.route('/', methods=['GET'])
def handle_index():
    return jsonify({'status': 'success'})


@app.route("/codebases", methods=['GET'])
def handle_codebases():
    try:
        codebases = [f for f in os.listdir('./data') if not f.startswith('.')]
    except OSError as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(codebases)


@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400

    codebase = data['codebase']
    query = data['query']
    top_k = data.get('top_k', 12)
    filter = data.get('filter', None)

    results = ask_codebase(codebase, data['query'], top_k=top_k, filter=filter)

    result = {
        'codebase': codebase,
        'message': f'Found {len(results)} results',
        'query': query,
        'results': results,
        'status': 'success',
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
