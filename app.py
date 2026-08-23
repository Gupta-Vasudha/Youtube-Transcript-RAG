import os
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
for _key in ("GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "WEBSHARE_PROXY_USERNAME", "WEBSHARE_PROXY_PASSWORD"):
    if _key not in os.environ and _key in st.secrets:
        os.environ[_key] = st.secrets[_key]
        
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",)

from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked, IpBlocked
 
import model

st.set_page_config(page_title="Chat with your Youtube video", page_icon="🎬", layout="wide")

def init_session_state():
    defaults = {"video_index": None, "chat_history": []}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def process_video(url_or_id: str):
    progress_bar = st.progress(0, text="Starting...")
 
    def on_progress(fraction: float, message: str):
        progress_bar.progress(int(fraction * 100), text=f"{message} ({int(fraction * 100)}%)")
 
    try:
        video_index = model.load_or_build_video(url_or_id, progress_callback=on_progress)
    except ValueError as e:
        progress_bar.empty()
        st.error(str(e))
        return
    except TranscriptsDisabled:
        progress_bar.empty()
        st.error("Captions are disabled for this video.")
        return
    except NoTranscriptFound:
        progress_bar.empty()
        st.error("No transcript could be found for this video.")
        return
    except VideoUnavailable:
        progress_bar.empty()
        st.error("This video is unavailable.")
        return
    except (RequestBlocked, IpBlocked):
        progress_bar.empty()
        st.error(
            "Server's IP request is being blocked by Youtube this app needs a proxy configured (WEBSHARE_PROXY_USERNAME / WEBSHARE_PROXY_PASSWORD) to work around it. This cannot be fixed by retrying." )
        return
    except RuntimeError as e:
        progress_bar.empty()
        st.error(str(e))
        return
    except Exception as e:
        progress_bar.empty()
        st.error(f"Failed to process video: {e}")
        return

    progress_bar.empty()
    st.session_state.video_index = video_index
    st.session_state.chat_history = []
    if video_index.from_cache:
        st.success("Ask your questions now...")
    else:
        st.success(f"Ask your questions now...")


def main():
    init_session_state()

    st.title("Interact with your Youtube Video")
    st.caption("Paste a YouTube URL or video ID at the side bar and press process.")

    with st.sidebar:
        st.header("Load a video")
        url_or_id = st.text_input(
            "YouTube URL or video ID",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        if st.button("Process video", type="primary", use_container_width=True):
            if url_or_id.strip():
                process_video(url_or_id)
            else:
                st.warning("Enter a URL or video ID first.")

        vi = st.session_state.video_index
        if vi:
            st.divider()
            st.subheader("Loaded video")
            st.write(f"**ID:** `{vi.video_id}`")
            st.write(f"**Source:** {'disk cache' if vi.from_cache else 'freshly indexed'}")
            st.video(f"https://www.youtube.com/watch?v={vi.video_id}")

    if not st.session_state.video_index:
        st.info("Load a video from the sidebar to start chatting.")
        return

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    question = st.chat_input("Ask something about the video...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = model.answer_question(st.session_state.video_index, question)
                except Exception as e:
                    answer = f"Something went wrong answering that: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
