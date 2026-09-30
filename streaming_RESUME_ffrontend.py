import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
import uuid





''' to programmitically generate thread id use uuid , it help you to access old thread'''
# Utility function --------------------------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config=CONFIG)

# ****************************session setup **************************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
add_thread(st.session_state['thread_id'])

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# ******************************** Side bar UI **************************************************
st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()


st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)


        temp_message = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role='assistant'
            temp_message.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_message
# *****************************************  MAin UI *******************************************


# loading the conversation history 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input("type here : ")

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)


    
    with st.chat_message('assistant'):
        Ai_Message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                   {'messages': [HumanMessage(content= user_input)]},
                   config = CONFIG,
                   stream_mode = 'messages'

            )
        )
    st.session_state['message_history'].append({'role':'assistant', 'content': Ai_Message})

        





















