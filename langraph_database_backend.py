from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama  
from langgraph.checkpoint.sqlite import SqliteSaver



# to make database import 
import sqlite3

load_dotenv()



llm = ChatOllama(
    model = "llama3.1:latest",
    temperature = 0
)

from langgraph.graph.message import add_messages
class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # take user question from state
    messages = state['messages']


    # send to llm 
    response = llm.invoke(messages)


    # response store back to state
    return {'messages' : [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)



graph = StateGraph(ChatState)
# add node 
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def init_title_table():
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_titles (
                thread_id TEXT PRIMARY KEY,
                title TEXT
            )
        """)

init_title_table()

def save_thread_title(thread_id: str, title: str):
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO thread_titles (thread_id, title)
            VALUES (?, ?)
        """, (thread_id, title))

def retrieve_all_thread_titles() -> dict:
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id, title FROM thread_titles")
        return dict(cursor.fetchall())







