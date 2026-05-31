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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* ================= MAIN ================= */

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

.stApp{
    background-color: #0b1120;
    color: white;
}

/* ================= CONTAINER ================= */

.block-container{
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 950px;
}

/* ================= TITLES ================= */

.main-title{
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0;
    color: white;
    letter-spacing: -1px;
}

.sub-text{
    color: #94a3b8;
    font-size: 1rem;
    margin-top: -4px;
    margin-bottom: 20px;
}

/* ================= INFO BOX ================= */

.info-box{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    margin-top: 10px;
    margin-bottom: 20px;
    color: #d1d5db;
    backdrop-filter: blur(10px);
}

/* ================= CHAT ================= */

.stChatMessage{
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
}

/* ================= AUDIO ================= */

.stAudioInput{
    border-radius: 16px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
}

/* ================= BUTTON ================= */

.stButton>button{
    border-radius: 12px;
    height: 44px;
    width: 100%;
    border: none;
    background: #7c3aed;
    color: white;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover{
    background: #6d28d9;
    transform: scale(1.02);
}

/* ================= INPUT ================= */

.stTextInput input{
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background-color: rgba(255,255,255,0.03) !important;
    color: white !important;
}

/* ================= TOGGLE ================= */

.stToggle{
    padding-top: 8px;
}

/* ================= FOOTER ================= */

.footer-text{
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 35px;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"]{
    background-color: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.06);
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

    st.title("🤖 Mahesh AI")

    st.markdown("### Voice Assistant")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        clear_conversation()

        st.session_state.history = []
        st.session_state.last_input = ""
        st.session_state.latest_audio = None

        st.rerun()

    st.divider()

    st.markdown("""
    ### 👨‍💻 About Me

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

if "audio_preview_active" not in st.session_state:
    st.session_state.audio_preview_active = False

if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None

if "audio_input_key" not in st.session_state:
    st.session_state.audio_input_key = 0

# NEW
if "latest_audio" not in st.session_state:
    st.session_state.latest_audio = None

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="main-title">
        Mahesh AI Assistant 🤖
    </div>

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
        st.caption("💬 Text Input Enabled")
    else:
        st.caption("🎤 Voice Input Enabled")

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

        if not os.path.exists(speech_path):
            return None

        return speech_path

    except Exception as e:

        st.warning(f"TTS Error: {e}")
        return None

# =====================================================
# PLAY AUDIO
# =====================================================

def play_audio(speech_file):

    if not speech_file:
        return

    try:

        with open(speech_file, "rb") as audio_file:
            audio_bytes = audio_file.read()

        st.session_state.latest_audio = audio_bytes

    except Exception as e:

        st.error(f"Audio playback error: {e}")

# =====================================================
# VOICE MODE
# =====================================================

if not chat_mode:

    st.subheader("🎤 Voice Input")

    audio_bytes = st.audio_input(
        "Speak",
        key=f"voice_recorder_{st.session_state.audio_input_key}"
    )

    # ---------- New Recording ----------
    if audio_bytes is not None and not st.session_state.audio_preview_active:

        st.session_state.recorded_audio = audio_bytes
        st.session_state.audio_preview_active = True

        st.rerun()

    # ---------- Preview Section ----------
    if (
        st.session_state.audio_preview_active
        and st.session_state.recorded_audio is not None
    ):

        st.markdown("### 🔍 Preview Recording")

        st.audio(
            st.session_state.recorded_audio,
            format="audio/wav"
        )

        col1, col2 = st.columns(2)

        # =================================================
        # SEND TO AI
        # =================================================

        with col1:

            if st.button(
                "✅ Send to AI",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Transcribing audio..."):

                    try:

                        # Save temporary wav
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".wav"
                        ) as f:

                            f.write(
                                st.session_state.recorded_audio.getbuffer()
                            )

                            audio_path = f.name

                        # Whisper transcription
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

                                play_audio(speech_file)

                    except Exception as e:

                        st.error(f"Voice processing error: {e}")

                # Reset state
                st.session_state.audio_preview_active = False
                st.session_state.recorded_audio = None

        # =================================================
        # DELETE RECORDING
        # =================================================

        with col2:

            if st.button(
                "🗑️ Delete & Re-record",
                use_container_width=True
            ):

                st.session_state.audio_preview_active = False
                st.session_state.recorded_audio = None
                st.session_state.audio_input_key += 1

                st.rerun()

# =====================================================
# CHAT MODE
# =====================================================

else:

    st.subheader("💬 Text Input")

    user_text = st.text_input(
        "Ask something",
        placeholder="Tell me about your AI projects..."
    )

    send_btn = st.button("Send")

    if (
        send_btn
        and user_text
        and not st.session_state.processing
    ):

        ai_text = generate_response(user_text)

        if ai_text:

            speech_file = text_to_speech(ai_text)

            play_audio(speech_file)

# =====================================================
# AUDIO PLAYER
# =====================================================

if st.session_state.latest_audio:

    st.audio(
        st.session_state.latest_audio,
        format="audio/mp3",
        autoplay=True
    )

# =====================================================
# CHAT HISTORY
# =====================================================

st.divider()

st.subheader("🧠 Conversation")

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
    elements[elements.length-1].scrollIntoView({
        behavior: 'smooth'
    });
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
