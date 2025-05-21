import streamlit as st
from dataclasses import dataclass
from pipeline import rag_pipeline
import uuid
from common.models import Message
from chroma.chroma import store_chat_message, load_chat_history, store_chat_session, get_all_chat_sessions

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
CURRENT_CHAT_KEY = "chat_id"
CHAT_SELECTION_KEY = "selected_chat"

def init_session_state():
    if CURRENT_CHAT_KEY not in st.session_state:
        st.session_state[CURRENT_CHAT_KEY] = None
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = []
    if CHAT_SELECTION_KEY not in st.session_state:
        st.session_state[CHAT_SELECTION_KEY] = "New chat..."
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

def on_chat_selected():
    """Callback for when a chat is selected from the dropdown"""
    selected = st.session_state[CHAT_SELECTION_KEY]
    
    if selected == st.session_state[CURRENT_CHAT_KEY]:
        return
        
    if selected == "New chat...":
        new_id = str(uuid.uuid4())
        st.session_state[CURRENT_CHAT_KEY] = new_id
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]
        store_chat_session(new_id)
    else:
        st.session_state[CURRENT_CHAT_KEY] = selected
        loaded_messages = load_chat_history(selected)
        if loaded_messages:
            st.session_state[MESSAGES] = loaded_messages
        else:
            st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Chat history could not be loaded completely. How can I help you?")]
    
    st.session_state.initialized = True

init_session_state()

st.sidebar.header("Chat Sessions")
existing_chats = get_all_chat_sessions()
chat_options = ["New chat..."] + existing_chats

if not st.session_state.initialized:
    if st.session_state[CURRENT_CHAT_KEY] in existing_chats:
        st.session_state[CHAT_SELECTION_KEY] = st.session_state[CURRENT_CHAT_KEY]
    else:
        st.session_state[CHAT_SELECTION_KEY] = "New chat..."

st.sidebar.selectbox(
    "Select chat to load", 
    chat_options, 
    key=CHAT_SELECTION_KEY,
    on_change=on_chat_selected
)

st.title("ReZanAI")
st.write("Ask me about constitution of the Republic of Kazakhstan and I'll use RAG to answer your questions!")

if not st.session_state.initialized and st.session_state[CURRENT_CHAT_KEY] is None:
    on_chat_selected()

for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")
if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    store_chat_message(st.session_state[CURRENT_CHAT_KEY], USER, prompt)


    current_chat_buffer = st.session_state[MESSAGES].copy()
    
    with st.spinner("Thinking..."):
        response: str = rag_pipeline(prompt, chat_history=current_chat_buffer)
    
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)
    store_chat_message(st.session_state[CURRENT_CHAT_KEY], ASSISTANT, response)


