# Youtube Transcript RAG

A Streamlit-based application that allows user to **interact with a YouTube video through its transcript** using RAG based pipeline.
Enter the URL or video ID of YouTube video and ask the chatbot query.

## Features

- Automatic retrieval of transcript using YoutubeTrascriptAPI
- Multilingual trascript support 
- Translation of non-English trascripts to english using Groq model
- Storing both original language transcript and translated trasncript
- Chunking of transcript into smaller chunks for better understanding and retrieval
- HuggingFaceEmbeddings based embedding of chunks
- FAISS based vector store 
- Similarity based retrieval
- Question based map-reduce method for entire video so that answer target only what was asked instead of providing complete summary
- Webshare Proxy support when YouTube blocks the API requests
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
Extraction of Video ID
        │
    Cache Check
        │
        ├── Cache hit ──────────────────────────────┐
        │                                           │
        └── Cache miss                              │
                │                                   │
   Fetch transcript via YouTubeTranscriptAPI        │
   (optionally through a Webshare proxy)            │
                │                                   │
                ├── English available ─────┐        │
                │                          │        │
                └── Non-English            │        │
                       │                   │        │
              Manual caption               │        │
              over auto-generated          │        │
                       │                   │        │
              Translate using Groq         │        │
                       │                   │        │
                       └─────────┬─────────┘        │
                              Chunking              │
                                 │                  |
                             Embeddings             │
                                 │                  │
                        FAISS Vector Store          │
                                 │                  │
                          Save to cache             │
                                 │                  │
                                 └────────────┬─────┘
                                            Query
                                              │
                              WHOLE video required or not?
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        │                                           │
                       No                                          Yes
                        │                                           │
              Top-K similarity retrieval               Map-reduce over the ENTIRE
                  (original language is                  transcript, guided by the
               given to the model as context)            actual question at every step
                        │                                             │
                        └─────────────────────┬───────────────────────┘
                                          Groq LLM
                                              │
                                            Answer
```

## RAG Pipeline

1. Regex is used to extract the video ID.
2. Cache is checked first, if available video is retrieved directly.
3. If not found, transcript is fetched using `youtube-transcript-api` (if requests blocked done using Webshare Proxy).
4. If an English transcript is available, it is used directly.
5. If the transcript is in another language, it is translated into English using Groq.
6. The translated transcript is divided into chunks.
7. Embeddings are generated using:
```text
hotchpotch/bekko-embedding-v1-a25m
```

8. The embeddings are stored in a FAISS vector store.
9. For the video that need context of whole video, map-reduce method runs for the entire transcript while keeping the question in mind instead of generating generic summary everytime. For normal questions, the most relevant transcript chunks are retrieved.
10. The LLM generates an answer using the updated context.

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
Accessed via `Hugging Face`.

## Environment Variables

Create a `.env` file locally based on `.env.example`.

The application loads environment variables using `python-dotenv` and can also read secrets from Streamlit setting if deployed.

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

In the Streamlit application's settings, add your secrets
```toml
GROQ_API_KEY = "your_groq_api_key"
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token"
WEBSHARE_PROXY_USERNAME = "your_webshare_proxy_username"
WEBSHARE_PROXY_PASSWORD = "your_webshare_proxy_password"
```
The application checks Streamlit Secrets when the environment variables are not already available.

## Blocked IP requests by YouTube
Used Webshare proxy to tackle the problem. Reuqests are routed through these proxies.
Youtube may still block some requests, this problem is not relted to the code.

## Caching
Processed videos are cached locally based on the video-ID and is shared across the users.

The following are stored: FAISS vector indexes, Original transcript, Translated transcript, Chunk information

It is excluded from Git using `.gitignore`.

## Limitations

- Depends completely on YouTube transcript availability.
- Videos with disabled captions cannot be processed.
- If no transcript is available, the application cannot answer questions about the video.
- Answers are based on the transcript, do not provide visual information.
- Non-English transcripts require translation before retrieval.
- API rate limits can affect processing and question answering.
- Hosting on cloud application, youtube IP requests may get blocked.
- Processing a long video for whole-video questions can require multiple LLM calls.
- Cache resets on redeploy or when the app wake up from sleep.

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
