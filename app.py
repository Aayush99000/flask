from flask import Flask, request, jsonify
import requests
import os
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize global variables
posts = []
endpoints = []
text_chunks = []
index = None
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
load_dotenv()
# Initialize conversation memory
memory = ConversationBufferMemory()
llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id='mistralai/Mistral-7B-v0.1',
    model_kwargs={"temperature": 0.5, "max_length": 512}
)
conversation = ConversationChain(llm=llm, memory=memory)

api_endpoints = ["https://blog.ted.com/wp-json/wp/v2/posts",
    "https://techcrunch.com/wp-json/wp/v2/posts"]

# Fetch WordPress posts (replace with your actual WordPress API URL)
def fetch_latest_posts(endpoints):
    texts = []
    for url in endpoints:
        try:
            # Make a GET request to fetch posts from the endpoint
            response = requests.get(url)
            response.raise_for_status()  # Raises an error for bad responses 
            
            # Parse the JSON response from the endpoint
            posts = response.json()
            for post in posts:
                # Parse the post content using BeautifulSoup to remove HTML tags
                soup = BeautifulSoup(post['content']['rendered'], 'html.parser')
                text = soup.get_text()
                texts.append({'title': post['title']['rendered'], 'content': text})
        
        except requests.exceptions.RequestException as err:
            # Log request errors but continue with other endpoints
            print(f"Request error for {url}: {err}")
    return texts

# Split text into chunks for processing (used by the Streamlit frontend)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Create a vector store (FAISS index) from text chunks
def get_vectorstore(text_chunks):
    embeddings = [embedding_model.encode(chunk) for chunk in text_chunks]  #Generate embeddings for each text chunk
    embeddings = np.array(embeddings).astype("float32")
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, text_chunks

# Retrieve relevant documents using FAISS search
def retrieve_documents(query, index, documents, top_k=5):
    query_embedding = embedding_model.encode([query])
    # Perform similarity search in the FAISS index
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)

    # Retrieve the top matching document chunks
    return [documents[i] for i in indices[0]]

# API endpoint to fetch posts from WordPress
@app.route('/fetch-posts', methods=['GET'])
def fetch_posts():
    # Fetch posts from all registered endpoints
    posts = fetch_latest_posts(api_endpoints)
    if not posts:
        return jsonify({"error": "Failed to fetch posts"}), 500
    return jsonify(posts), 200

# API endpoint to list available API endpoints
@app.route('/endpoints', methods=['GET'])
def list_endpoints():
    return jsonify(endpoints), 200

# API endpoint to add a new endpoint
@app.route('/endpoints', methods=['POST'])
def add_endpoint():
    # Get the WordPress site URL from the request
    wordpress_site = request.json.get('endpoint')
    
    if wordpress_site:
        # Ensure the URL ends with a slash ("/") for proper concatenation
        if not wordpress_site.endswith('/'):
            wordpress_site += '/'
        
        # Append the default API path
        full_api_endpoint = wordpress_site + "wp-json/wp/v2/posts"
        
        # Add the complete API endpoint to the list
        endpoints.append(full_api_endpoint)
        
        return jsonify({"message": "Endpoint added successfully", "endpoint": full_api_endpoint}), 201
    
    return jsonify({"message": "WordPress site URL is required"}), 400

# API endpoint to remove an endpoint by index
@app.route('/endpoints/<int:index>', methods=['DELETE'])
def remove_endpoint(index):
    if 0 <= index < len(endpoints):
        endpoints.pop(index)
        return jsonify({"message": "Endpoint removed successfully"}), 200
    return jsonify({"message": "Invalid endpoint index"}), 404

# API endpoint to process posts and generate vector store
# @app.route('/process-posts', methods=['POST'])
# def process_posts():
#     global text_chunks, index
#     if not posts:
#         return jsonify({"message": "No posts available. Please fetch posts first."}), 400

#     all_content = " ".join(post.get('content', '') for post in posts)

#     text_chunks = get_text_chunks(all_content)
#     index, text_chunks = get_vectorstore(text_chunks)

#     return jsonify({"message": "Posts processed and vector store created successfully"}), 200

# # API endpoint to query the chatbot
# @app.route('/query', methods=['POST'])
# def query():
#     # Query the FAISS vector store with a given query text.
#     query = request.json.get('query')
#     if not query:
#         return jsonify({"message": "Query is required"}), 400

#     if not text_chunks or not index:
#         return jsonify({"message": "No vector store available. Please process posts first."}), 400

#     # Retrieve relevant documents from the vector store
#     retrieved_docs = retrieve_documents(query, index, text_chunks)
#     context = " ".join(retrieved_docs)

#     # Add the context to the conversation chain
#     memory.add_user_message(query)
#     memory.add_ai_message(context)

#     response = conversation.run(query)

#     return jsonify({"answer": response}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)