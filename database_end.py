import streamlit as st
from langraph_database_backend import chatbot, retrieve_all_threads, save_thread_title, retrieve_all_thread_titles
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
import uuid
from dotenv import load_dotenv
from langchain_ollama import ChatOllama  

load_dotenv()

llm = ChatOllama(
    model = "llama3.1:latest",
    temperature = 0
)

''' START YOUR CONVERSATION'''
# Utility functions --------------------------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, title="New Chat")
    st.session_state['message_history'] = []
    st.session_state.conversation_title = "New Chat"

def add_thread(thread_id, title="New Chat"):
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if 'thread_titles' not in st.session_state:
        st.session_state['thread_titles'] = retrieve_all_thread_titles()
        
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = title

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# **************************** Session setup **************************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = retrieve_all_thread_titles()

add_thread(st.session_state['thread_id'])

if 'conversation_title' not in st.session_state:
    st.session_state.conversation_title = st.session_state['thread_titles'].get(st.session_state['thread_id'], "New Chat")

# ******************************** Sidebar UI **************************************************
st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_threads']:
    thread_title = st.session_state['thread_titles'].get(thread_id, "New Chat")
    button_label = f"💬 {thread_title}" if thread_id == st.session_state['thread_id'] else thread_title
    
    if st.sidebar.button(button_label, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state.conversation_title = thread_title
        messages = load_conversation(thread_id)

        temp_message = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_message.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_message
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Active: **{st.session_state.conversation_title}**")

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# ***************************************** Main UI *******************************************

# Loading the conversation history 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input("type here : ")

if user_input:
    # Generate title only for new conversation
    if len(st.session_state['message_history']) == 0:
        title_prompt = f""" 
        Generate a short title for this conversation.
        Rules :
        - 3 to 6 words
        - Describe the main topic
        - Return only the title
        - No quotes
        user message: {user_input}
        """
        try:
            response = llm.invoke(title_prompt)
            title = response.content.strip().strip('"\'')
            st.session_state.conversation_title = title
            st.session_state['thread_titles'][st.session_state['thread_id']] = title
            save_thread_title(st.session_state['thread_id'], title)
        except Exception:
            pass


    # First add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        Ai_Message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                   {'messages': [HumanMessage(content=user_input)]},
                   config = CONFIG,
                   stream_mode = 'messages'
            )
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': Ai_Message})
    st.rerun()























