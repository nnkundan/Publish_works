import streamlit as st
from keys1 import key_groq, key_langsmith
import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Set up environment variables for your keys
os.environ['GROQ_FACE_API'] = key_groq
os.environ['langchain_langsmith_API'] = key_langsmith
os.environ['langchain_tracing_v2'] = 'true'
os.environ['langchain_project'] = 'courselangraph'

# Initialize the LLM model
llm = ChatGroq(groq_api_key=key_groq, model_name="gemma2-9b-it")

# Define StateGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Define chatbot function
def chatbot(state: State):
    return {"messages": llm.invoke(state['messages'])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Streamlit UI for the Chatbot
st.title("💬 AI Chatbot")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation history
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"**🧑 User:** {msg}")
    elif role == "assistant":
        st.markdown(f"**🤖 Assistant:** {msg}")

# User input area
user_input = st.text_input("Type your message here...", key="user_input")

# Button to send the message
if st.button("Send"):
    if user_input:
        # Update conversation state with the user's message
        st.session_state.messages.append(("user", user_input))
        
        # Process through the graph
        for event in graph.stream({'messages': st.session_state.messages}):
            for value in event.values():
                # Append the assistant's response to the session state
                response = value["messages"].content
                st.session_state.messages.append(("assistant", response))
                st.markdown(f"**🤖 Assistant:** {response}")

# Optional clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
