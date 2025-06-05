 # News Article Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about news articles. It uses OpenAI's embeddings and GPT-3.5-turbo model along with ChromaDB for efficient document storage and retrieval.

## Features

- Document loading and processing from text files
- Text chunking with configurable chunk size and overlap
- Vector embeddings using OpenAI's text-embedding-3-small model
- Persistent storage using ChromaDB
- Semantic search for relevant document chunks
- Question answering using GPT-3.5-turbo

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages (install via pip):
  ```
  python-dotenv
  openai
  chromadb
  ```

## Setup

1. Clone the repository
2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Create a `news_article` directory and place your text files (`.txt`) containing news articles in it
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── app.py              # Main application file
├── .env               # Environment variables
├── news_article/      # Directory containing news article text files
└── chroma_persistent_storage/  # ChromaDB persistent storage
```

## Usage

1. Place your news article text files in the `news_article` directory
2. Run the application:
   ```
   python app.py
   ```
3. The system will:
   - Load and process the documents
   - Split them into chunks
   - Generate embeddings
   - Store them in ChromaDB
   - Answer questions based on the content

## How It Works

1. **Document Processing**: The system loads text files from the `news_article` directory and splits them into manageable chunks.

2. **Embedding Generation**: Each text chunk is converted into a vector embedding using OpenAI's text-embedding-3-small model.

3. **Storage**: The embeddings and text chunks are stored in ChromaDB for efficient retrieval.

4. **Question Answering**: When a question is asked:
   - The question is converted to an embedding
   - Similar chunks are retrieved from the database
   - GPT-3.5-turbo generates a concise answer based on the retrieved context

## Configuration

You can modify the following parameters in `app.py`:
- `chunk_size`: Size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 20 characters)
- `n_results`: Number of relevant chunks to retrieve (default: 2)

## Note

Make sure to keep your OpenAI API key secure and never commit it to version control. The `.env` file is included in `.gitignore` by default.