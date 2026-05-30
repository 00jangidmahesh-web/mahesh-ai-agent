from dotenv import load_dotenv
load_dotenv(override=True)

import os
import tempfile
import streamlit as st
from openai import OpenAI

from ai_agent import get_ai_response, clear_conversation

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Mahesh AI Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.block-container{
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 900px;
}

.stChatMessage{
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 10px;
}

.stAudioInput{
    border-radius: 14px;
    padding: 10px;
    border: 1px solid #dcdcdc;
}

.main-title{
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0;
}

.sub-text{
    color: #6b7280;
    font-size: 1rem;
    margin-top: -5px;
}

.info-box{
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 14px;
    border-radius: 12px;
    margin-top: 15px;
    margin-bottom: 20px;
}

.footer-text{
    text-align:center;
    color:gray;
    font-size:0.85rem;
    margin-top:30px;
}

.stButton>button{
    border-radius: 10px;
    height: 42px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# OPENAI CLIENT
# =====================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("Mahesh AI ")   # graduation cap emoji

    if st.button("Clear Conversation", use_container_width=True):
        clear_conversation()
        st.session_state.history = []
        st.session_state.last_input = ""
        st.rerun()

    st.divider()

    st.markdown("""
    **Mahesh Kumar Jangid**  
    MSc Mathematics & Computing  
    IIT ISM Dhanbad
    """)

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "processing" not in st.session_state:
    st.session_state.processing = False

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div class="main-title">Mahesh AI Assistant 🤖</div>   <!-- robot emoji -->
    <div class="sub-text">
        AI Voice Assistant • IIT ISM Dhanbad
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Ask questions about my projects, background, skills, AI work, or experience.
    You can interact using voice or text.
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# MODE TOGGLE
# =====================================================
mode_col1, mode_col2 = st.columns([1, 4])

with mode_col1:
    chat_mode = st.toggle(
        "Chat Mode",
        value=False
    )

with mode_col2:
    if chat_mode:
        st.caption("Text Input Enabled")
    else:
        st.caption("Voice Input Enabled")

st.divider()

# =====================================================
# GENERATE RESPONSE
# =====================================================
def generate_response(user_input: str):

    if not user_input:
        return None

    if user_input == st.session_state.last_input:
        return None

    st.session_state.last_input = user_input
    st.session_state.processing = True

    with st.spinner("Generating response..."):

        try:
            ai_response = get_ai_response(user_input)

            if not ai_response:
                ai_response = "Unable to generate response."

        except Exception as e:
            ai_response = f"Error: {str(e)}"

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Mahesh", ai_response))

    st.session_state.processing = False

    return ai_response

# =====================================================
# TEXT TO SPEECH
# =====================================================
def text_to_speech(text: str):

    if not text:
        return None

    if "Error:" in text:
        return None

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            speech_path = f.name

        with st.spinner("Generating audio..."):

            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="onyx",
                input=text,
            ) as response:

                response.stream_to_file(speech_path)

        return speech_path

    except Exception as e:
        st.warning(f"TTS Error: {e}")
        return None

# =====================================================
# VOICE MODE
# =====================================================
if not chat_mode:

    st.subheader("🎤 Voice Input")   # microphone emoji added here

    audio = st.audio_input("Speak")

    if audio and not st.session_state.processing:

        with st.spinner("Transcribing audio..."):

            try:

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio.getbuffer())
                    audio_path = f.name

                with open(audio_path, "rb") as audio_file:

                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )

                user_text = transcription.text.strip()

                if user_text:

                    st.info(f"You said: {user_text}")

                    ai_text = generate_response(user_text)

                    if ai_text:

                        speech_file = text_to_speech(ai_text)

                        if speech_file:
                            st.audio(speech_file, format="audio/mp3")

            except Exception as e:
                st.error(f"Transcription Error: {e}")

# =====================================================
# CHAT MODE
# =====================================================
else:

    st.subheader("Text Input")

    user_text = st.text_input(
        "Ask something",
        placeholder="Tell me about your AI projects..."
    )

    send_btn = st.button("Send")

    if send_btn and user_text and not st.session_state.processing:

        ai_text = generate_response(user_text)

        if ai_text:

            speech_file = text_to_speech(ai_text)

            if speech_file:
                st.audio(speech_file, format="audio/mp3")

# =====================================================
# CHAT HISTORY
# =====================================================
st.divider()

st.subheader("Conversation")

if not st.session_state.history:

    st.info("No conversation yet.")

else:

    for role, msg in st.session_state.history:

        if role == "You":

            with st.chat_message("user"):
                st.markdown(msg)

        else:

            with st.chat_message("assistant"):
                st.markdown(msg)

# =====================================================
# AUTO SCROLL
# =====================================================
st.markdown("""
<script>
    var elements = window.parent.document.querySelectorAll('.stChatMessage');
    if(elements.length) {
        elements[elements.length-1].scrollIntoView({behavior: 'smooth'});
    }
</script>
""", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer-text">
    Built with Streamlit + OpenAI
    </div>
    """,
    unsafe_allow_html=True
)