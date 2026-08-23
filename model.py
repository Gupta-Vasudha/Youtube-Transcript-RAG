import os
import re
import time
import logging
import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import WebshareProxyConfig

logger = logging.getLogger(__name__)

LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "hotchpotch/bekko-embedding-v1-a25m"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVER_K = 4
MAX_CONTEXT_CHARS = 9000
MAP_REDUCE_BATCH_CHARS = 6000

SUMMARY_TRIGGER_WORDS = {"summarize", "summarise", "summary", "overview", "recap"}
WHOLE_VIDEO_WORDS = { "entire", "whole", "full", "complete", "overall","all"}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_VERSION = 2
EMBED_BATCH_SIZE = 16

def _report(progress_callback, fraction: float, message: str) -> None:

    """Call the optional UI progress callback with a 0-1 fraction and a short status message. No-op if no callback was supplied."""
    if progress_callback is not None:
        progress_callback( min(max(fraction, 0.0), 1.0), message )

def _call_with_retries(fn, *args, attempts: int = 3, base_delay: float = 1.5,
                        non_retryable: tuple = (), **kwargs):
    """Call fn(*args, **kwargs), retrying on transient failures with exponential backoff. Exceptions listed in non_retryable (e.g. a video genuinely having no transcript) are raised immediately without retrying, since retrying them just wastes time on a non-transient problem."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except non_retryable:
            raise
        except Exception as e:
            last_exc = e
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs", getattr(fn, "__name__", "call"), attempt, attempts, e, delay, )
            time.sleep(delay)
    raise last_exc
 
def _get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your local .env file, or, if deployed on Streamlit Cloud, add it under the app's Settings -> Secrets.")
    return key

@lru_cache(maxsize=1)
def get_youtube_api() -> YouTubeTranscriptApi:
    """Build the YouTubeTranscriptApi client, routed through a Webshare proxy if WEBSHARE_PROXY_USERNAME/WEBSHARE_PROXY_PASSWORD are set."""

    proxy_username = os.getenv("WEBSHARE_PROXY_USERNAME")
    proxy_password = os.getenv("WEBSHARE_PROXY_PASSWORD")
 
    if proxy_username and proxy_password:
        logger.info("Using Webshare proxy for YouTube transcript requests.")
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
            )
        )
 
    logger.info("No proxy configured; connecting to YouTube directly.")
    return YouTubeTranscriptApi()
 
@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(model_name=LLM_MODEL,temperature=0.3,api_key=_get_groq_api_key())

@lru_cache(maxsize=1)
def get_classifier_llm():
    """A temperature-0 instance used only for quick yes/no style classification calls (deciding whether a question needs whole-video context)"""
    return ChatGroq(model_name=LLM_MODEL, temperature=0.0, api_key=_get_groq_api_key())
 
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "ru": "Russian", "it": "Italian", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi","ur": "Urdu", "gu": "Gujarati", "pa": "Punjabi",
}

def language_name(code: str) -> str:
    """Map a YouTube language code (e.g. 'hi', 'en-US') to a readable name. Falls back to returning the raw code if it isn't in the lookup table, so this never raises."""
    base = code.split("-")[0].lower()
    return LANGUAGE_NAMES.get(base, code)

@dataclass
class VideoIndex:
    video_id: str
    original_lang: str
    original_transcript: str
    translated_transcript: str
    vector_store: FAISS
    chunks: list
    retriever: object
    qa_chain: object
    from_cache: bool

# Step 0 — URL / ID parsing
def extract_video_id(raw: str) -> str | None:
    raw = raw.strip()
    if re.fullmatch( r"[A-Za-z0-9_-]{11}",raw): return raw

    match = re.search(
        r"(?:v=|/videos/|embed/|youtu\.be/|/v/|/shorts/)([A-Za-z0-9_-]{11})", raw
    )
    return match.group(1) if match else None


# Step 1 — Transcript fetch + translation
def fetch_transcript_text(video_id: str,llm,progress_callback=None) -> tuple[str, str, str]:

    api = YouTubeTranscriptApi()
    transcript_list = _call_with_retries(
        api.list, video_id, non_retryable=(TranscriptsDisabled, VideoUnavailable),)
    _report( progress_callback, 0.05, "Fetching transcript...")

    # Case 1 — English transcript exists

    try:
        transcript = transcript_list.find_transcript(["en"])
        fetched = _call_with_retries(transcript.fetch)
        original_transcript = " ".join(seg.text for seg in fetched)

        _report(progress_callback,0.20,"English transcript fetched.")
        logger.info("%s: transcript already in English, no translation needed.", video_id)
        return (original_transcript,original_transcript,"en")

    except NoTranscriptFound:
        pass

    # Case 2 — English transcript doesn't exist
    available = list(transcript_list)
    if not available:
        raise NoTranscriptFound(video_id,[],transcript_list)

    manual = [t for t in available if not t.is_generated]
    transcript = (manual[0] if manual else available[0])

    original_lang = transcript.language_code
    fetched = _call_with_retries(transcript.fetch)
    original_transcript = " ".join(seg.text for seg in fetched)

    if not original_transcript.strip():
        raise NoTranscriptFound(video_id,[],transcript_list)

    logger.info(
        "%s: transcript language=%s (%s), translating via Groq.", video_id, original_lang, "manual" if transcript in manual else "auto-generated",)
    _report(progress_callback, 0.08, f"Translating from {language_name(original_lang)} using Groq...")
    
    translated_transcript = llm_translate_long_text(original_transcript,llm,progress_callback)

    _report(progress_callback,0.20,"Transcript translated using Groq.")
    return ( original_transcript, translated_transcript, original_lang )

# Groq translation fallback
def llm_translate_long_text(text: str,llm,progress_callback=None) -> str:

    splitter = RecursiveCharacterTextSplitter(chunk_size=MAP_REDUCE_BATCH_CHARS, chunk_overlap=0)
    batches = splitter.split_text(text)
    if not batches:
        raise ValueError("Transcript could not be split into translation batches.")
    prompt = PromptTemplate.from_template(
        "Translate the following text to English. "
        "Return ONLY the translation, with no notes or explanations.\n\n {text}"
    )

    chain = (prompt | llm | StrOutputParser())

    translated_batches = []
    total = len(batches)

    for i, batch in enumerate(batches):
        translated = _call_with_retries(chain.invoke, {"text": batch})
        translated_batches.append(translated)
        fraction = (0.05 + 0.15 * ((i + 1) / total))
        _report(progress_callback,fraction,f"Translating with Groq... ")

    return " ".join(translated_batches)

# Disk cache
def _cache_paths(video_id: str) -> tuple[str, str]:
    """Return (faiss_index_dir, chunks_pickle_path) for a given video ID."""
    index_dir = os.path.join(CACHE_DIR,video_id)
    chunks_path = os.path.join(CACHE_DIR,f"{video_id}_chunks.pkl")

    return (index_dir,chunks_path)

def load_cached_index(video_id: str,embeddings) -> tuple[FAISS,list,str,str,str] | None:   
    index_dir, chunks_path = _cache_paths(video_id)

    if not (os.path.isdir(index_dir) and os.path.isfile(chunks_path)):
        return None
    try:
        vector_store = FAISS.load_local(index_dir,embeddings,allow_dangerous_deserialization=True)

        with open(chunks_path, "rb" ) as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict):
            logger.info("%s: cache payload is an old unversioned format, rebuilding.", video_id)
            return None
 
        if payload.get("version") != CACHE_VERSION:
            logger.info(
                "%s: cache version mismatch (found %r, expected %r), rebuilding.", video_id, payload.get("version"), CACHE_VERSION,)
            return None
        
        chunks = payload.get("chunks")
        original_lang = payload.get("original_lang","en")
        original_transcript = payload.get("original_transcript","")
        translated_transcript = payload.get("translated_transcript","")

        if (not original_transcript or not translated_transcript):
            logger.warning("%s: cached transcript fields are empty, rebuilding.", video_id)
            return None
        return ( vector_store, chunks, original_lang, original_transcript, translated_transcript )

    except Exception as e:
        logger.warning("%s: failed to load cache (%s), rebuilding.", video_id, e)
        return None

def save_index_to_cache(video_id: str,vector_store: FAISS,chunks: list,original_lang: str, original_transcript: str,translated_transcript: str) -> None:

    index_dir, chunks_path = _cache_paths(video_id)
    vector_store.save_local(index_dir)

    with open(chunks_path,"wb") as f:
        pickle.dump({
                "version": CACHE_VERSION,"chunks": chunks,"original_lang": original_lang,"original_transcript": original_transcript,"translated_transcript": translated_transcript, },f )
    logger.info("%s: saved to cache (lang=%s).", video_id, original_lang)

# Step 2 — Chunk + embed + index

def build_vector_store(transcript_text: str,embeddings,progress_callback=None) -> tuple[FAISS, list]:
    _report(progress_callback,0.22,"Splitting transcript into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.create_documents([transcript_text])
    texts = [c.page_content for c in chunks]
    metadatas = [ c.metadata for c in chunks]

    total = len(texts)
    
    all_embeddings = []
    for i in range(0,total,EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        batch_embeddings = _call_with_retries(embeddings.embed_documents, batch)
        all_embeddings.extend(batch_embeddings)
        done = i + len(batch)
        fraction = (0.25  + 0.60 * ( done / total ) )
        _report(progress_callback,fraction,f"Embedding chunks... ")
    _report(progress_callback,0.87,"Building vector index...")
    
    text_embeddings = list(zip(texts,all_embeddings) )
    vector_store = FAISS.from_embeddings(text_embeddings,embeddings,metadatas=metadatas)
    return (vector_store,chunks)

# Step 3 — Normal similarity retrieval

def build_retriever(vector_store: FAISS):
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K})

def format_docs_with_budget(docs,max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join retrieved chunks into one context string, deduping repeats and truncating once the char budget is hit."""

    seen, pieces, total = set(), [],0
    for doc in docs:
        content = (doc.page_content.strip())
        if content in seen:
            continue
        seen.add(content)
        if (total + len(content)> max_chars):
            remaining = (max_chars - total)
            if remaining > 200:
                pieces.append(content[:remaining])
            break
        pieces.append(content)
        total += len(content)
    return "\n\n".join(pieces)


QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions about a YouTube video using only the transcript excerpts provided below.
    
    The video's original spoken language is: {original_lang}.
    The transcript excerpts below have already been translated into English if the original language was not English.
    
    If the context is insufficient to answer, say:
    "This isn't explained in the video. Continue with other question"

    Note: this is a SPOKEN transcript. It describes what the presenter said, not what appeared on screen. If asked to print, show, or draw something like a visual pattern, and the transcript describes it in enough detail (e.g. how many rows, which characters, how the spacing/alignment works), reconstruct it as an actual grid inside a code block, and say clearly that this is a reconstruction based on the spoken description, not something read directly off the screen. If the transcript does not give enough detail to reconstruct it confidently, say "the model works only on transcript and the transcript do not contain enough information for this task" explicitly instead of guessing.
Transcript context:
{context}

Question: {question}

Answer:""",
    input_variables=["context","question", "original_lang"],)

def build_qa_chain(retriever, llm, original_lang: str):
    def run( question: str ) -> str:
        docs = retriever.invoke( question )
        context = format_docs_with_budget( docs )
        chain = (QA_PROMPT | llm | StrOutputParser())
        return _call_with_retries(chain.invoke, {"context": context, "question": question, "original_lang": language_name(original_lang),})
    return RunnableLambda(run)

# Whole-video summary

MAP_PROMPT = PromptTemplate.from_template(
    "A user asked this question about a video: {question}\n\n"
    "Below is one excerpt from the video's transcript (not the whole video). Extract only what's relevant to answering the question from this excerpt. Be specific and complete — if the excerpt describes something like a list item, a step, or a visual pattern, capture the concrete details given (names, numbers, order, structure), not just a vague topic label. If nothing in this excerpt is relevant to the question, respond with exactly: NOT RELEVANT.\n\n"
    "Transcript excerpt:\n{text}"
)

REDUCE_PROMPT = PromptTemplate.from_template(
    "A user asked this question about a video: {question}\n\n"
    "Below are notes extracted from sequential parts of the video's transcript, each attempting to capture information relevant to that question. Some notes may say NOT RELEVANT — ignore those.\n\n Combine the relevant notes into one complete, direct answer to the question. If the question asks to list, enumerate, or print multiple items (such as patterns), include ALL of them found across the notes, not just the first few. If an item is a visual pattern and the notes describe it in enough detail (rows, characters, spacing), reconstruct it as an actual grid inside a code block, and say clearly that this is a reconstruction based on the spoken description, for more bettwe visual refer to the video itself rather than something read directly off the screen. If none of the notes are relevant, say exactly: \"This isn't explained in the video. Continue with other question\"\n\n"
    "Notes:\n{text}"
)

def is_whole_video_summary_request(question: str) -> bool:
    """Free, fast keyword pre-check for the most obvious 'summarize the whole video' phrasings. Not exhaustive — see needs_full_video_context() below for the LLM-based fallback used when this comes back False."""
    q = question.lower()
    return (any(w in q for w in SUMMARY_TRIGGER_WORDS) and any(w in q for w in WHOLE_VIDEO_WORDS))

CLASSIFY_PROMPT = PromptTemplate.from_template(
    "A user is asking a question about a video's transcript.\n"
    "Decide whether answering this question WELL requires seeing the ENTIRE transcript — e.g. it asks to list/enumerate/count everything of some kind covered in the video, or wants an overall summary — versus a question that can be answered from just a few relevant excerpts, such as asking about one specific detail, moment, or topic.\n\n"
    "Question: {question}\n\n"
    "Respond with exactly one word: FULL or PARTIAL."
)

def _classify_needs_full_video(question: str) -> bool:
    """LLM fallback for when the keyword heuristic is inconclusive (e.g. list all the patterns' or 'what patterns are discussed', neither of which match the fixed SUMMARY_TRIGGER_WORDS/WHOLE_VIDEO_WORDS sets). Uses a dedicated temperature-0 model for a consistent yes/no call. On any failure, defaults to PARTIAL (the cheaper, more common path) rather than silently paying for a full map-reduce pass."""
    chain = CLASSIFY_PROMPT | get_classifier_llm() | StrOutputParser()
    try:
        result = _call_with_retries(chain.invoke, {"question": question}, attempts=2)
    except Exception as e:
        logger.warning("Whole-video classification call failed (%s); defaulting to PARTIAL.", e)
        return False
    return result.strip().upper().startswith("FULL")

def needs_full_video_context(question: str) -> bool:
    """Decide whether a question needs the whole-video map-reduce path. Tries the free keyword heuristic first; only falls back to a paid LLM classification call when that's inconclusive, so obvious cases like 'summarize the entire video' cost nothing extra, while less predictable phrasings like 'list all the X' still get routed correctly."""
    if is_whole_video_summary_request(question):
        return True
    return _classify_needs_full_video(question)
 
 
def map_reduce_summarize(question:str, chunks,llm) -> str:
    """Answer `question` using the ENTIRE transcript via a map-reduce pass:extract relevant details from each chunk (map), then combine them into one direct answer (reduce)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAP_REDUCE_BATCH_CHARS, chunk_overlap=0)

    full_text = "\n\n".join( c.page_content for c in chunks)
    batches = splitter.split_text(full_text)
    map_chain = ( MAP_PROMPT | llm | StrOutputParser())

    partial_notes = [
        _call_with_retries(map_chain.invoke, {"question": question, "text": batch})
        for batch in batches
    ]
    relevant_notes = [n for n in partial_notes if n.strip().upper() != "NOT RELEVANT"]
    if not relevant_notes:
        return "This isn't explained in the video. Continue with other question"
 
    combined = "\n\n".join(relevant_notes)
    if len(combined) > MAX_CONTEXT_CHARS:
        fake_docs = [type( "D", (), { "page_content": summary})() for summary in relevant_notes]
        combined = format_docs_with_budget(fake_docs,max_chars=MAX_CONTEXT_CHARS)

    reduce_chain = ( REDUCE_PROMPT | llm | StrOutputParser())
    return _call_with_retries(reduce_chain.invoke, {"question": question,"text": combined})


def load_or_build_video(url_or_id: str, progress_callback=None) -> VideoIndex:

    """Resolves a URL/ID to a video_id. Either loads the cached FAISS index, or fetches the transcript, translates it via Groq if needed, chunks, embeds and saves it to cache. Original and translated transcripts are stored separately."""

    _report(progress_callback,0.0,"Resolving video ID...")
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError( "Couldn't find a valid YouTube video ID in that input.")

    _report(progress_callback,0.02,"Loading models...")

    llm = get_llm()
    embeddings = get_embeddings()

    _report(progress_callback,0.04,"Checking cache...")
    cached = load_cached_index(video_id,embeddings)

    if cached is not None:
        (vector_store,chunks,original_lang, original_transcript,translated_transcript) = cached
        from_cache = True
        _report(progress_callback,0.85,"Loaded from cache.")

    else:
        ( original_transcript, translated_transcript, original_lang) = fetch_transcript_text( video_id, llm, progress_callback)
        vector_store, chunks = build_vector_store(translated_transcript,embeddings,progress_callback)

        _report(progress_callback,0.90,"Saving to cache...")

        save_index_to_cache(
            video_id, vector_store, chunks, original_lang, original_transcript,translated_transcript)
        from_cache = False
    _report(progress_callback,0.93,"Setting up retriever...")

    retriever = build_retriever( vector_store )
    qa_chain = build_qa_chain(retriever, llm, original_lang)
    _report(progress_callback, 1.0,"Ready.")

    return VideoIndex(
        video_id=video_id, original_lang=original_lang, original_transcript=original_transcript,translated_transcript=translated_transcript, vector_store=vector_store, chunks=chunks,retriever=retriever, qa_chain=qa_chain, from_cache=from_cache)

def answer_question(video_index: VideoIndex,question: str) -> str:
    llm = get_llm()
    if needs_full_video_context(question):
        logger.info("%s: routing to full-video map-reduce for question: %r",
                     video_index.video_id, question)
        return map_reduce_summarize(question, video_index.chunks, llm)
    return video_index.qa_chain.invoke(question)
