from RAG_MCP_ALl import embeddings
from RAG_MCP_ALl import vector_store
import tempfile
from typing import Any
from typing_extensions import Dict
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated, Any, Dict, Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import AsyncGenerator
from importlib import machinery
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama  
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite
import threading
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader
import sqlite3
import requests

load_dotenv()

# --------------
# LLM ----------

llm = ChatOllama(
    model = "llama3.1:latest ",
    temperature = 1
)


#------------------------------pdf retriever store (per thread)
_THREAD_RETRIEVER: Dict[str, Any] = {}
_THREAD_METADATA : Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """ Fetch the retriever for a thread if available """
    if thread_id and thread_id in _THREAD_RETRIEVER:
        return _THREAD_RETRIEVER[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str]=None) -> dict:
    """ Build a FAISS retriever for the uploaded PDF and Store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """

    if not file_bytes:
        raise ValueError("no bytes recieved for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try: 
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size =1000, chunk_overlap= 200, separators = ["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type = 'similarity', search_kwargs={'k':4}
        )

        _THREAD_RETRIEVER[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        return {
            "filename" : filename or os.path.basename(temp_path),
            "documents":len(docs),
            "chunks":len(chunks),
        }

    finally:
        # the faiss store keeps copies of the text , so the temp file is safe to remove.
        try: 
            os.remove(temp_path)
        except OSError:
            pass





# ---------------
# tools 

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(f_num : float, s_num : float, operation : str) -> dict:
    """
    Perform a basic arithmatic operation on two numbers.
    Supported operation : add , sub , mul, div
    """
    try:
        if operation == 'add':
            result = f_num + s_num
        elif operation == 'sub':
            result = f_num - s_num
        elif operation == 'mul':
            result = f_num * s_num
        elif operation == 'div':
            if s_num == 0:
                return {'error': "division by zero is not allowed"}
            result = f_num / s_num
        else:
            return {'error':f"unsupported operation'{operation}'"}
        
        return {'f_num':f_num, "s_num": s_num, 'operation': operation, 'result': result}

    except Exception as e:
        return {'error':str(e)}


# @tool
# def get_stock_price(symbol:str) -> dict:
#     """
#     fetch latest stock price for a given symbol (e.g. 'TSLA', 'AAPL', 'NIFTY-FIFTY')
#     using alpha vantage with API key in the url 
#     """

@tool
def rag_tool(query:str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevent information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            'error': "no document indexed for this chat. Upload a PDf first.",
            "query": query,
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context':context,
        'metadata':metadata,
        'source_file': _THREAD_METADATA.get(str(thread_id), {}).get('filename')
    }



tools = [search_tool, calculator, rag_tool]
llm_with_tool = llm.bind_tools(tools)



#------------------
# state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


#------------------------
# Node

def chat_node(state: ChatState):
    """ LLM node that may answer or reqeuest a tool call."""
    messages= state["messages"]
    response = llm_with_tool.invoke(messages)
    return {'messages':[response]}

tool_node = ToolNode(tools)

# -----------------------------
# checkpointer

conn = sqlite3.connect(database='chatbot_tool.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)




#-------------------------
# Graph

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)



#----------------
# helper

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def thread_has_documents(thread_id: str) -> bool:
    return str(thread_id)

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

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






