from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import TypedDict, Literal, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama  
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import ToolNode, tools_condition 
import os
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
os.environ['LANGCHAIN_PROJECT'] = 'RAG_BOT'
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


llm = ChatOllama(
    model = "llama3.2:latest",
    temperature = 0.5
)


#document ingestion
loader = PyPDFLoader("Rudra_Shriwas_BMW_Internship_Cover_Letter.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',

)


vector_store = FAISS.from_documents(chunks, embeddings)



retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})


@tool
def rag_tool(query):
    """
    Retrieve relevent information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context':context,
        'metadata':metadata
    }


tools = [rag_tool]
llm_with_tool = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tool.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)


graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile()


result = chatbot.invoke(
    {
        "messages":[
            HumanMessage(
                content=(
                    'Using the pdf , explain the intention behind that cover letter'
                )
            )
        ]
    }
)

print(result['messages'][-1].content)






















