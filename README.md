# Youtube Transcript RAG

A Streamlit-based application that allows user to **interact with a YouTube video through its transcript** using RAG based pipeline.
Enter the URL or video ID of YouTube video and ask the chatbot query

## Features

- Automatic retrieval of transcript using YoutubeTrascriptAPI
- Multilingual trascript support 
- Translation of non-English trascripts to english using Groq model
- Storing both original language transcript and translated trasncript
- Chunking of transcript into smaller chunks for better understanding
- HuggingFace Embeddings based embedding of chunks
- FAISS based vector store 
- Similarity based retrieval
- Summary based questions using map-reduce approach and whole-video questions
- Caches to store the videos to avoid re-embedding of already processed videos
- Streamlit web interface

## Structure of the Project

```text
Youtube-Transcript-RAG/
│
├── app.py              # Streamlit application
├── model.py            # Entire RAG pipeline - transcript fetch
├── requirements.txt    # Dependencies
├── .env.example        # Example of environment variables
├── .gitignore          
└── README.md           # Project documentation
```

## Working

```text
Get YouTube URL / Video ID
        │
        ▼
Extraction of Video ID
        │
        ▼
Fetch YouTube Transcript via YoutubeTranscriptAPI
        │
        ├── English ───────────────┐
        │                           │
        └── Non-English             │
                │                   │
                ▼                   │
        Translate using Groq        │
                │                   │
                └─────────┬─────────┘
                          ▼
                       Chunking
                          │
                          ▼
                      Embeddings
                          │
                          ▼
                    FAISS Vector Store
                          │
                          ▼
                       Retriever
                          │
                          ▼
                        Query
                          │
                          ▼
                  Similarity Retrieval
                          │
                          ▼
                       Groq LLM
                          │
                          ▼
                       Answer
```

## RAG Pipeline

1. Regex is used to extract the video ID.
2. Transcript retrieval is done using `youtube-transcript-api`.
3. If an English transcript is available, it is used directly.
4. If the transcript is in another language, it is translated into English using Groq.
5. The translated transcript is divided into chunks.
6. Embeddings are generated using:

```text
hotchpotch/bekko-embedding-v1-a25m
```

7. The embeddings are stored in a FAISS vector store.
8. For normal questions, the most relevant transcript chunks are retrieved.
9. The retrieved transcript snippet are passed to the Groq LLM.
10. The LLM generates an answer using the transcript context.

The application also has a separate whole-video path for questions that require information from the complete transcript.

## Models

### LLM

```text
openai/gpt-oss-120b
```

Accessed through Groq using `ChatGroq`.

### Embedding Model

```text
hotchpotch/bekko-embedding-v1-a25m
```
Accessed via Hugging Face

## Environment Variables

Create a `.env` file locally based on `.env.example`.

The application loads environment variables using `python-dotenv` and can also read secrets from Streamlit when deployed.

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment and activate it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file, then add your API keys to `.env`.

### 5. Run the application

```powershell
streamlit run app.py
```

## For Streamlit deployment

Add API keys to Streamlit Secrets instead of uploading your .env file

In the Streamlit application's settings, add your secrets
```toml
GROQ_API_KEY = "your_groq_api_key"
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token"
```
The application checks Streamlit Secrets when the environment variables are not already available.

## Caching

Processed videos are cached locally.

The application stores: FAISS vector indexes, Transcript information, Original transcript, Translated transcript, Chunk information

It is excluded from Git using `.gitignore`.

## Limitations

- Depends completely on YouTube transcript availability.
- Videos with disabled captions cannot be processed.
- If no transcript is available, the application cannot answer questions about the video.
- Answers are based on the transcript, do not represent information that appears only visually in the video.
- Non-English transcripts require translation before retrieval.
- API rate limits can affect processing and question answering.
- Processing a long video for whole-video questions can require multiple LLM calls.

## Technologies Used

- Python
- Streamlit
- LangChain
- Groq
- FAISS
- Hugging Face
- YouTube Transcript API
- Python-dotenv

## License
This project is intended for learning and demonstration purposes.
