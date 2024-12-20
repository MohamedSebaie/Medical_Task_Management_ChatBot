import streamlit as st # type: ignore
import requests # type: ignore
import json
import plotly.graph_objects as go # type: ignore
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd # type: ignore

# Configure page
st.set_page_config(
    page_title="Medical NLP Bot",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def process_command(text: str) -> Dict[str, Any]:
    """Send command to API and get response"""
    api_url = "http://localhost:8000/api/process"
    
    try:
        response = requests.post(
            api_url,
            json={
                "text": text,
                "conversation_history": st.session_state.conversation_history
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {"success": False, "error": str(e)}

def visualize_entities(entities: Dict[str, List]) -> go.Figure:
    """Create visualization for extracted entities"""
    categories = []
    values = []
    
    for category, items in entities.items():
        if items:
            categories.append(category)
            values.append(len(items))
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=['rgb(158,202,225)', 'rgb(94,158,217)', 
                         'rgb(32,102,148)', 'rgb(4,52,100)']
        )
    ])
    
    fig.update_layout(
        title="Extracted Entities by Category",
        xaxis_title="Entity Category",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig

def display_intent_confidence(intent: Dict[str, Any]):
    """Display intent classification results"""
    st.write("### Intent Analysis")
    
    # Create gauge chart for confidence
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = intent["confidence"] * 100,
        title = {'text': f"Intent: {intent['primary_intent']}"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ]
        }
    ))
    
    st.plotly_chart(fig)

def main():
    st.title("Medical NLP Bot 🏥")
    
    # Sidebar
    st.sidebar.header("Settings")
    show_raw_output = st.sidebar.checkbox("Show Raw API Output")
    show_visualizations = st.sidebar.checkbox("Show Visualizations", value=True)
    
    # Main chat interface
    st.write("### Medical Assistant Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your medical command here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append(prompt)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the command
        with st.spinner("Processing..."):
            response = process_command(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            if response["success"]:
                result = response["result"]
                
                # Display main response
                st.write("### Analysis Results")
                
                # Show visualizations if enabled
                if show_visualizations:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_intent_confidence(result["intent"])
                    
                    with col2:
                        fig = visualize_entities(result["entities"])
                        st.plotly_chart(fig)
                
                # Display structured information
                st.write("#### Extracted Information")
                for category, items in result["entities"].items():
                    if items:
                        st.write(f"**{category.title()}:**")
                        for item in items:
                            st.write(f"- {item['text']} ({item['type']})")
                
                # Display temporal information
                if result["temporal_info"]["dates"] or result["temporal_info"]["patterns"]:
                    st.write("#### Temporal Information")
                    for key, values in result["temporal_info"].items():
                        if values:
                            st.write(f"**{key.title()}:** {', '.join(values)}")
                
                # Show raw output if enabled
                if show_raw_output:
                    st.write("#### Raw API Output")
                    st.json(response)
            else:
                st.error(f"Error: {response.get('error', 'Unknown error')}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Analysis completed. See results above."
        })

if __name__ == "__main__":
    main()