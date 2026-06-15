import streamlit as st
from chatbot_backend_tools import chatbot, retrieve_all_threads,save_thread_name,retrieve_thread_names
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid
from langchain_core.messages import SystemMessage

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = str(generate_thread_id())
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])
def get_chat_name(user_input):
    """Ask the LLM for a short chat title based on the first message."""
    try:
        from complete_chatbot_backend import llm  # import your llm instance
        messages = [
            SystemMessage(content=(
                "Generate a short 4-5 word title for a chat that starts with the "
                "following message. Reply with ONLY the title, no punctuation, no quotes."
            )),
            HumanMessage(content=user_input)
        ]
        response = llm.invoke(messages)
        title = response.content.strip()[:50]
        return title if title else " ".join(user_input.split()[:5])
    except Exception:
        # fallback to first 5 words if LLM call fails
        return " ".join(user_input.split()[:5])

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(generate_thread_id())

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = retrieve_thread_names()

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************
st.title("SAKSHAM KA CHATBOT")
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    tid_str = str(thread_id)
    name = st.session_state['thread_names'].get(tid_str, "New chat")
    if st.sidebar.button(name, key=tid_str):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun() 

# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    tid_str = str(st.session_state['thread_id'])
    if tid_str not in st.session_state['thread_names']:               
        name = get_chat_name(user_input)                   
        st.session_state['thread_names'][tid_str] = name  
        save_thread_name(tid_str, name)


    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # first add the message to message_history
    with st.chat_message('assistant'):

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.content and not getattr(message_chunk, "tool_calls", []):
                        yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )