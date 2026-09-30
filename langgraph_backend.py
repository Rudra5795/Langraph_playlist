from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama  
from langgraph.checkpoint.memory import MemorySaver

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


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

# add node 
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)





