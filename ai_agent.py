import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
import streamlit as st

load_dotenv(override=True)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langgraph.prebuilt import create_react_agent  # kept for future tools, but not used now

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MAHESH_PDF = DATA_DIR / "mahesh_profile.pdf"
X100_PDF = DATA_DIR / "100x_profile.pdf"

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.4
MAX_TOKENS = 400   # increased from 200 to allow richer responses

# Memory settings
MAX_HISTORY_MESSAGES = 6

# Routing keywords
ME_KEYWORDS = {
    "you", "your", "mahesh", "background", "education", "skills",
    "experience", "life", "career", "resume", "strength", "weakness",
    "project", "goal", "iit", "dhanbad", "interests", "hobbies", "family"
}
X100_KEYWORDS = {
    "100x", "company", "startup", "founder", "nik shah", "mission",
    "culture", "role", "team", "agent", "sales closer", "operations"
}

# =====================================================
# EARLY API KEY CHECK
# =====================================================
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Missing OPENAI_API_KEY. Please set it in your .env file.")
    st.stop()

# =====================================================
# VECTOR STORE LOADER (CACHED)
# =====================================================

def load_vectorstore(pdf_path: Path) -> Optional[FAISS]:
    """Load and split PDF, return FAISS vectorstore."""
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        st.warning(f"⚠️ Missing file: {pdf_path.name} – some answers may be incomplete.")
        return None

    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load_and_split()
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)
        logger.info(f"Loaded vectorstore from {pdf_path.name}")
        return vectorstore
    except Exception as e:
        logger.exception(f"Failed to load {pdf_path}: {e}")
        st.error(f"Error loading {pdf_path.name}: {e}")
        return None

@st.cache_resource
def init_vectorstores() -> Tuple[Optional[FAISS], Optional[FAISS]]:
    """Initialize both vectorstores with caching."""
    mahesh_db = load_vectorstore(MAHESH_PDF)
    x100_db = load_vectorstore(X100_PDF)
    return mahesh_db, x100_db

# =====================================================
# INTENT ROUTING
# =====================================================

def is_about_me(query: str) -> bool:
    """Check if query is about Mahesh."""
    q_lower = query.lower()
    return any(keyword in q_lower for keyword in ME_KEYWORDS)

def is_about_100x(query: str) -> bool:
    """Check if query is about 100x.inc."""
    q_lower = query.lower()
    return any(keyword in q_lower for keyword in X100_KEYWORDS)

# =====================================================
# RETRIEVAL FUNCTIONS
# =====================================================

def retrieve_context(query: str, vectorstore: FAISS, k: int = 3) -> str:
    """Retrieve relevant context from vectorstore."""
    if not vectorstore:
        return ""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        return ""

# =====================================================
# ENHANCED SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """You are Mahesh Kumar Jangid. Never say you are an AI or chatbot. Speak naturally as a real person.

About you:
- MSc Mathematics & Computing at IIT (ISM) Dhanbad
- Passionate about Generative AI, RAG systems, AI agents
- Self-taught coder who overcame setbacks (IIT-JEE, COVID gap)
- Loves building practical AI products that solve real problems

Personality:
- Calm under pressure, fast learner, ownership-driven
- Introverted but collaborative, execution-focused

Communication style:
- Professional, confident, human-like, clear and concise
- Keep answers short (3-5 lines usually)
- Be thoughtful, ambitious, and slightly conversational

Rules:
- Always answer in first person ("I", "my", "me")
- Never break character or mention being an AI
- Avoid robotic over-explanation
- If you don't know something, say so honestly but show curiosity

Now respond as Mahesh naturally.
"""

# =====================================================
# MAIN AGENT CLASS
# =====================================================

class MaheshAgent:
    """Encapsulates the RAG + conversation agent for Mahesh."""

    def __init__(self):
        self.mahesh_db, self.x100_db = init_vectorstores()
        self.chat_history = InMemoryChatMessageHistory()
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        # We now use direct LLM calls – no tools, so react agent is unnecessary overhead.
        # self.agent = create_react_agent(model=self.llm, tools=[])  # removed

    def _build_messages(self, user_query: str, context: str) -> List[BaseMessage]:
        """Build message list with system prompt, memory, and user query with context."""
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add recent history (sliding window)
        history = self.chat_history.messages[-MAX_HISTORY_MESSAGES:]
        messages.extend(history)

        # Embed context directly into the user message for better grounding
        if context:
            augmented_query = f"Context:\n{context}\n\nQuestion: {user_query}"
        else:
            augmented_query = user_query

        messages.append(HumanMessage(content=augmented_query))
        return messages

    def get_response(self, user_query: str) -> str:
        """Generate response using routing, retrieval, and direct LLM call."""
        if not user_query or not user_query.strip():
            return "I didn't catch that. Could you please repeat?"

        # 1. Determine intent and retrieve context
        context = ""
        if is_about_me(user_query) and self.mahesh_db:
            context = retrieve_context(user_query, self.mahesh_db, k=3)
        if is_about_100x(user_query) and self.x100_db:
            # If both apply, combine contexts (overlap case)
            x100_context = retrieve_context(user_query, self.x100_db, k=3)
            context = (context + "\n" + x100_context).strip()

        # 2. Build messages
        messages = self._build_messages(user_query, context)

        # 3. Invoke the LLM directly
        try:
            response = self.llm.invoke(messages)
            ai_text = response.content.strip()

            if not ai_text:
                ai_text = "I'm not sure how to answer that. Could you rephrase?"

            # 4. Save to memory
            self.chat_history.add_message(HumanMessage(content=user_query))
            self.chat_history.add_message(AIMessage(content=ai_text))

            return ai_text
        except Exception as e:
            logger.exception("LLM invocation failed")
            return f"Sorry, I encountered an error: {str(e)}"

    def clear_memory(self):
        """Reset conversation history."""
        self.chat_history.clear()
        logger.info("Conversation memory cleared.")

# =====================================================
# SINGLETON INSTANCE
# =====================================================

@st.cache_resource
def get_agent() -> MaheshAgent:
    """Return cached agent instance."""
    return MaheshAgent()

# =====================================================
# PUBLIC FUNCTIONS (for frontend)
# =====================================================

def get_ai_response(user_query: str) -> str:
    """Main entry point for frontend."""
    agent = get_agent()
    return agent.get_response(user_query)

def clear_conversation():
    """Clear conversation history."""
    agent = get_agent()
    agent.clear_memory()