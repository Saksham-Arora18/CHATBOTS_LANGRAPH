from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_names (
        thread_id TEXT PRIMARY KEY,
        name      TEXT NOT NULL
    )
""")
conn.commit()
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(str(checkpoint.config['configurable']['thread_id']))

    return list(all_threads)

def save_thread_name(thread_id: str, name: str):
    conn.execute(
        "INSERT OR IGNORE INTO thread_names (thread_id, name) VALUES (?, ?)",
        (thread_id, name)
    )
    conn.commit()

def retrieve_thread_names() -> dict:
    rows = conn.execute("SELECT thread_id, name FROM thread_names").fetchall()
    return {row[0]: row[1] for row in rows}