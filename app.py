import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small",
)

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

#Loading documents from the directory
def load_documents_from_directory(directory_path):
    print("==== Loading documtents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
                ) as file:
                    documents.append({"id": filename, "text": file.read()})

    return documents

# Funmction to split documents into chunks
def split_text(text, chunk_size=1000,chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./news_article"
documents = load_documents_from_directory(directory_path)

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
         chunked_documents.append({"id": f"{doc['id']}_chunks{i+1}", "text": chunk})

# Function to get OpenAI embedding
def get_openai_embedding(text):
     response = client.embeddings.create(input=text, model="text-embedding-3-small")
     embedding = response.data[0].embedding
     print("==== Getting OpenAI embedding ====")
     return embedding

for doc in chunked_documents:
     print("==== Adding chunk to collection ====")
     doc["embedding"] = get_openai_embedding(doc["text"])

for doc in chunked_documents:
     collection.upsert(
          ids = [doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
     )

#Function to query the documents
def query_documents(question, n_results=2):
    # Get OpenAI embedding for the question
    results = collection.query(
        query_texts=question,
        n_results=n_results)
    
    # Extract relevant chunks from the results
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks

# Function to generate a response using OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},  
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message
    return answer

question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)