import streamlit as st
from keys1 import key_groq, key_langsmith
import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Set up environment variables
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

# Modified chatbot function to handle travel parameters
def chatbot(state: State):
    # Get user inputs from session state
    inputs = st.session_state.inputs
    prompt = f"""
    Create a travel plan with these parameters:
    City: {inputs['city']}
    Days: {inputs['days']}
    Budget: ${inputs['budget']}
    Response must be concise and limited to {inputs['word_limit']} words.
    
    Include:
    - Top attractions
    - Daily itinerary
    - Budget breakdown
    - Local tips
    """
    
    response = llm.invoke([("system", prompt)]).content
    return {"messages": response}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Streamlit UI
st.title("🌍 Travel Planner AI")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# Input form
with st.form("travel_inputs"):
    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("City Name", key="city")
        days = st.number_input("Number of Days", min_value=1, max_value=30, key="days")
    with col2:
        budget = st.number_input("Budget (USD)", min_value=50, key="budget")
        word_limit = st.number_input("Word Limit", min_value=50, max_value=100, value=100, key="word_limit")
    
    submitted = st.form_submit_button("Generate Travel Plan")
    
    if submitted:
        # Store inputs
        st.session_state.inputs = {
            "city": city,
            "days": days,
            "budget": budget,
            "word_limit": word_limit
        }
        
        # Clear previous messages
        st.session_state.messages = []
        
        # Process through the graph
        for event in graph.stream({'messages': []}):  # Start with empty messages
            for value in event.values():
                response = value["messages"]
                st.session_state.messages.append(("assistant", response))

# Display output
st.subheader("Your Travel Plan")
for role, msg in st.session_state.messages:
    if role == "assistant":
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; margin:10px 0;'>
            {msg}
        </div>
        """, unsafe_allow_html=True)

# Clear button
if st.button("Clear All"):
    st.session_state.messages = []
    st.session_state.inputs = {}
    st.rerun()
