import streamlit as st
import asyncio
import logging
from dataclasses import dataclass

@dataclass
class ChatMessage:
    role: str
    content: str
    sources: list = None
from src.chatbot import CustomChatBot
import os

INDEX_DATA = os.environ.get("INDEX_DATA", "0")
PULL_EMBEDDING_MODEL = os.environ.get("PULL_EMBEDDING_MODEL", "0")

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console logs
    ],
)

logger = logging.getLogger(__name__)

# Initialize chatbot instance (avoid reloading)
if "bot" not in st.session_state:
    st.session_state["bot"] = CustomChatBot(index_data=bool(int(INDEX_DATA)), pull_embedding_model=bool(int(PULL_EMBEDDING_MODEL)))

# Streamlit UI setup
st.set_page_config(page_title="ChatDoc", page_icon="📄")
st.header("Chat with your Document")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

if len(st.session_state["messages"]) == 0 or st.sidebar.button("Clear message history"):
    st.session_state["messages"].clear()
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg.role):
        st.write(msg.content)
        # Wenn Quellen existieren, in einem aufklappbaren Menü anzeigen
        if msg.sources:
            with st.expander("📄 Verwendete Quellen anzeigen"):
                for i, source in enumerate(msg.sources):
                    # Text aus LangChain-Dokumenten extrahieren
                    content = source.page_content if hasattr(source, "page_content") else str(source)
                    st.info(f"**Quelle {i+1}:**\n\n{content}")

# Handle user input
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.session_state.messages.append(ChatMessage(role="user", content=user_query))
    logger.info(f"Write user message in session state {user_query}")
    st.chat_message("user").write(user_query)

    async def handle_user_query(user_query):
        container = st.empty()
        answer = ""
        try:
            async for chunk in st.session_state["bot"].astream(user_query):
                if chunk:
                    answer+=chunk
                    container.markdown(answer)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            container.error("An error occurred while processing your request.")

        # Store assistant response in session state
    if answer:
        logger.info(f"Write assistant message in session state {user_query}")
        
        # Quellen aus der Datenbank abrufen
        retrieved_sources = []
        try:
            retrieved_sources = st.session_state["bot"].vector_db.similarity_search(user_query, k=3)
        except Exception as e:
            logger.warning(f"Could not fetch sources: {e}")

        # Nachricht inkl. Quellen im Verlauf speichern
        st.session_state.messages.append(ChatMessage(role="assistant", content=answer, sources=retrieved_sources))
        
        # Quellen sofort unter der aktuellen Antwort anzeigen
        if retrieved_sources:
            with st.expander("📄 Verwendete Quellen anzeigen"):
                for i, source in enumerate(retrieved_sources):
                    content = source.page_content if hasattr(source, "page_content") else str(source)
                    st.info(f"**Quelle {i+1}:**\n\n{content}")
                    
    with st.chat_message("assistant"):
        with st.spinner("Searching for information in your documents and generation response..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handle_user_query(user_query))