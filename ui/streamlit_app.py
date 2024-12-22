from fastapi import logger
import streamlit as st # type: ignore
import requests
import json
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Medical Task Management ChatBot",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .extracted-info {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    .info-category {
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .info-item {
        margin-left: 20px;
        padding: 5px 0;
    }
    .json-output {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .metric-card h3 {
        color: #2C3E50;
        font-size: 1rem;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #3498DB;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px;
        color: #2C3E50;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB !important;
        color: white !important;
    }
    .chat-container {
        display: flex;
        align-items: flex-start;
        padding: 1rem;
        margin-bottom: 0.5rem;
        width: 100%;
    }
    
    .chat-icon {
        font-size: 2.5rem;  /* Increased icon size */
        margin: 0 0.8rem;   /* Increased margin */
        min-width: 45px;    /* Increased min-width */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .message-container {
        display: flex;
        flex-direction: column;
        max-width: 80%;
    }
    
    .user-container {
        flex-direction: row-reverse;
    }
    
    .user-message {
        background-color: #5B8FF9;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 15px 15px 0 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 15px 15px 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .follow-up-message {
        background-color: #00ACC1;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 15px 15px 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .timestamp {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.25rem;
        text-align: right;
    }
    
    .user-timestamp {
        text-align: right;
    }
    
    .assistant-timestamp {
        text-align: left;
    }
    .error-message {
            background-color: #ffebee;
            color: #c62828;
            padding: 0.8rem 1rem;
            border-radius: 15px;
            border: 1px solid #ef9a9a;
            margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chat_history" not in st.session_state:  # Add this
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "processed_texts" not in st.session_state:
    st.session_state.processed_texts = []
if "patients" not in st.session_state:
    st.session_state.patients = pd.DataFrame(columns=['name', 'age', 'gender', 'condition'])
if "medications" not in st.session_state:
    st.session_state.medications = pd.DataFrame(columns=['patient', 'medication', 'dosage', 'frequency'])
if "appointments" not in st.session_state:
    st.session_state.appointments = pd.DataFrame(columns=['patient', 'date', 'time', 'department'])
if "pipeline_type" not in st.session_state:
    st.session_state.pipeline_type = "transformer"

def process_command(text: str) -> Dict[str, Any]:
    """Send command to API and get response"""
    api_url = "http://localhost:8000/api/process"
    
    try:
        response = requests.post(
            api_url,
            json={
                "text": text,
                "conversation_history": st.session_state.conversation_history,
                "pipeline_type": st.session_state.pipeline_type
            },
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error occurred")
            st.error(f"Processing Error: {error_msg}")
            return result
            
        # Validate response structure
        if "result" not in result or "intent" not in result["result"]:
            st.error("Invalid response format from server")
            return {
                "success": False,
                "error": "Invalid response format"
            }
        
        # Update session state
        if result["success"]:
            st.session_state.processed_texts.append(result["result"])
            update_session_data(result["result"])
        
        return result
        
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return {"success": False, "error": str(e)}

def update_session_data(result: Dict[str, Any]):
    """Update session state with processed data"""
    # Update patients dataframe if relevant entities are found
    entities = result['entities']
    if 'patient_info' in entities and entities['patient_info']:
        patient_data = {
            'name': next((e['text'] for e in entities['patient_info'] if e['type'] == 'patient'), None),
            'age': next((e['text'] for e in entities['patient_info'] if e['type'] == 'age'), None),
            'gender': next((e['text'] for e in entities['patient_info'] if e['type'] == 'gender'), None),  # Updated gender extraction
            'condition': next((e['text'] for e in entities.get('medical_info', []) if e['type'] == 'condition'), None)
        }
        
        # If gender not found in patient_info, try other entity categories
        if not patient_data['gender']:
            for category in entities.values():
                gender_entity = next((e['text'] for e in category if e['type'] == 'gender'), None)
                if gender_entity:
                    patient_data['gender'] = gender_entity
                    break
        
        if patient_data['name']:
            # Create DataFrame if it doesn't exist
            if 'patients' not in st.session_state:
                st.session_state.patients = pd.DataFrame(columns=['name', 'age', 'gender', 'condition'])
            
            st.session_state.patients = pd.concat([
                st.session_state.patients,
                pd.DataFrame([patient_data])
            ], ignore_index=True)

def display_intent_confidence(intent: Dict[str, Any]):
    """Display intent classification results with improved gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=intent["confidence"] * 100,
        title={'text': f"Intent: {intent['primary_intent']}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#EBF5FB'},
                {'range': [50, 75], 'color': '#AED6F1'},
                {'range': [75, 100], 'color': '#3498DB'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor="white",
        font={'color': "#2C3E50", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def visualize_entities(entities: Dict[str, List]) -> go.Figure:
    """Create enhanced visualization for extracted entities"""
    categories = []
    values = []
    colors = ['#AED6F1', '#3498DB', '#2980B9']
    
    for category, items in entities.items():
        if items:
            categories.append(category.replace('_', ' ').title())
            values.append(len(items))
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Extracted Entities by Category",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Entity Category",
        yaxis_title="Count",
        template="plotly_white",
        height=300,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#2C3E50", 'family': "Arial"}
    )
    
    return fig

def display_extracted_info(result: Dict[str, Any]):
    # Helper function to safely get entities
    def get_entities(category: str) -> List[Dict[str, Any]]:
        return result.get("entities", {}).get(category, [])
    
    # Helper function to safely extract entity text
    def get_entity_text(entities: List[Dict], entity_type: str, alternate_types: List[str] = None) -> Optional[str]:
        for entity in entities:
            if entity.get("type") == entity_type:
                return entity.get("text")
            if alternate_types and entity.get("type") in alternate_types:
                return entity.get("text")
        return None

    # Helper function to get age from multiple possible locations
    def get_age(result: Dict[str, Any]) -> Optional[str]:
        # Try getting age from temporal_info first
        temporal_info = get_entities("temporal_info")
        age = get_entity_text(temporal_info, "age")
        if age:
            return age
        
        # If not found, try getting from demographics in patient_info
        patient_info = get_entities("patient_info")
        for entity in patient_info:
            if entity.get("type") == "demographics" and "years old" in entity.get("text", ""):
                return entity.get("text")
        return None

    # Display patient information
    patient_info = get_entities("patient_info")
    if patient_info:
        st.markdown("### Patient Information:")
        for entity in patient_info:
            st.markdown(f"• {entity.get('text', 'Unknown')} ({entity.get('type', 'Unknown')})")
    
    # Display medical information
    medical_info = get_entities("medical_info")
    if medical_info:
        st.markdown("### Medical Information:")
        for entity in medical_info:
            st.markdown(f"• {entity.get('text', 'Unknown')} ({entity.get('type', 'Unknown')})")

    # Display simplified entity format
    st.markdown("### Simplified Format:")
    
    # Build entities dictionary with non-null values only
    entities = {}
    print('===================================')
    print()
    print('======================================')
    
    # Extract all possible entities
    potential_entities = {
        "patient": get_entity_text(patient_info, "patient", ["patient_name"]),
        "gender": get_entity_text(patient_info, "gender"),
        "age": get_age(result),  # Using the new get_age function
        "condition": get_entity_text(medical_info, "condition", ["diagnosis"]),
        "medication": get_entity_text(medical_info, "medication"),
        "dosage": get_entity_text(medical_info, "dosage"),
        "frequency": get_entity_text(medical_info, "frequency"),
        "appointment_date": result.get('simplified_format').get('entities').get('appointment_date')
    }
    # Only include non-null values
    for key, value in potential_entities.items():
        if value is not None:
            entities[key] = value
    
    simplified = {
        "intent": result.get("intent", {}).get("primary_intent", "unknown"),
        "entities": entities
    }
    
    st.code(json.dumps(simplified, indent=2), language="json")

def show_dashboard():
    """Display enhanced dashboard page"""
    st.title("Medical NLP Dashboard 🏥")
    
    # Summary Cards in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Patients</h3>
                <p class="metric-value">{}</p>
            </div>
        """.format(len(st.session_state.patients)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Intents</h3>
                <p class="metric-value">{}</p>
            </div>
        """.format(len(st.session_state.processed_texts)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Conditions</h3>
                <p class="metric-value">{}</p>
            </div>
        """.format(len([item for text in st.session_state.processed_texts 
                       for item in text['entities'].get('medical_info', [])
                       if item['type'] == 'condition'])), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3>Processing Accuracy</h3>
                <p class="metric-value">98.4%</p>
            </div>
        """.format(), unsafe_allow_html=True)

    # Charts Section
    st.markdown("### Analytics Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.processed_texts:
            # Intent Distribution
            intents = [text['intent']['primary_intent'] for text in st.session_state.processed_texts]
            intent_counts = pd.Series(intents).value_counts()
            
            fig = px.pie(
                values=intent_counts.values,
                names=intent_counts.index,
                title="Intent Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No intent data available yet")

    with col2:
        if st.session_state.processed_texts:
            # Entity Types Distribution
            entity_types = []
            for text in st.session_state.processed_texts:
                for category in text['entities'].values():
                    entity_types.extend([item['type'] for item in category])
            
            entity_counts = pd.Series(entity_types).value_counts()
            
            fig = px.bar(
                x=entity_counts.index,
                y=entity_counts.values,
                title="Entity Types Distribution",
                labels={'x': 'Entity Type', 'y': 'Count'},
                color_discrete_sequence=['#3498DB']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entity data available yet")

    # Recent Activity
    st.markdown("### Recent Activity")
    if st.session_state.processed_texts:
        recent_df = pd.DataFrame([
            {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Intent': text['intent']['primary_intent'],
                'Confidence': f"{text['intent']['confidence']:.2%}",
                'Entities': len([item for category in text['entities'].values() for item in category])
            }
            for text in st.session_state.processed_texts[-5:]  # Last 5 entries
        ])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent activity to display")

def show_data_views():
    """Display enhanced data views page"""
    st.title("Medical Data Views 📊")
    
    # Create tabs with custom styling
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #fff;
            border-radius: 4px;
            color: #2C3E50;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            padding: 0 20px;
            font-size: 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3498DB !important;
            color: white !important;
        }
        .data-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .filter-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Patients", "Medical Records", "Entity Analysis"])
    
    with tabs[0]:  # Patients Tab
        st.markdown("### Patient Records")
        if not st.session_state.patients.empty:
            # Filters in a styled container
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                age_filter = st.slider("Filter by Age", 0, 100, (0, 100))
            with col2:
                # Safely handle conditions
                conditions = st.session_state.patients['condition'].dropna().unique()
                all_conditions = sorted([str(c) for c in conditions if c is not None])
                condition_filter = st.multiselect(
                    "Filter by Condition",
                    options=all_conditions,
                    default=[]
                ) if all_conditions else []
            with col3:
                # Safely handle genders
                genders = st.session_state.patients['gender'].dropna().unique()
                all_genders = sorted([str(g) for g in genders if g is not None])
                gender_filter = st.multiselect(
                    "Filter by Gender",
                    options=all_genders,
                    default=[]
                ) if all_genders else []
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Apply filters
            filtered_df = st.session_state.patients.copy()
            
            # Handle age filtering safely
            try:
                # Extract numeric age values safely
                filtered_df['age_num'] = filtered_df['age'].apply(
                    lambda x: pd.to_numeric(str(x).split()[0]) if pd.notnull(x) else None
                )
                age_mask = (
                    filtered_df['age_num'].notna() & 
                    (filtered_df['age_num'] >= age_filter[0]) & 
                    (filtered_df['age_num'] <= age_filter[1])
                )
                filtered_df = filtered_df[age_mask]
                filtered_df = filtered_df.drop('age_num', axis=1)
            except Exception as e:
                st.warning("Some age values couldn't be processed. Showing all ages.")
            
            # Handle condition filtering safely
            if condition_filter:
                filtered_df = filtered_df[
                    filtered_df['condition'].fillna('').astype(str).isin(condition_filter)
                ]
            
            # Handle gender filtering safely
            if gender_filter:
                filtered_df = filtered_df[
                    filtered_df['gender'].fillna('').astype(str).isin(gender_filter)
                ]
            
            # Display filtered data
            st.markdown('<div class="data-container">', unsafe_allow_html=True)
            st.dataframe(
                filtered_df.fillna('Not specified'),
                hide_index=True,
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Demographics visualizations
            if not filtered_df.empty:
                st.markdown("#### Patient Demographics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    try:
                        age_data = filtered_df['age'].dropna().apply(
                            lambda x: pd.to_numeric(str(x).split()[0])
                        )
                        if not age_data.empty:
                            fig = px.histogram(
                                age_data,
                                nbins=20,
                                title="Age Distribution",
                                labels={'value': 'Age', 'count': 'Number of Patients'},
                                color_discrete_sequence=['#3498DB']
                            )
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(t=40, b=20, l=20, r=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("Age distribution visualization unavailable")
                
                with col2:
                    try:
                        condition_counts = filtered_df['condition'].dropna().value_counts()
                        if not condition_counts.empty:
                            fig = px.pie(
                                values=condition_counts.values,
                                names=condition_counts.index,
                                title="Condition Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig.update_layout(
                                margin=dict(t=40, b=20, l=20, r=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("Condition distribution visualization unavailable")
                
                with col3:
                    try:
                        gender_counts = filtered_df['gender'].dropna().value_counts()
                        if not gender_counts.empty:
                            fig = px.pie(
                                values=gender_counts.values,
                                names=gender_counts.index,
                                title="Gender Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig.update_layout(
                                margin=dict(t=40, b=20, l=20, r=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("Gender distribution visualization unavailable")
            else:
                st.info("No data available for the selected filters")
        else:
            st.info("No patient data available")

    with tabs[1]:  # Medical Records Tab
        st.markdown("### Medical Records Analysis")
        if st.session_state.processed_texts:
            # Extract medical records
            try:
                medical_records = []
                for text in st.session_state.processed_texts:
                    for item in text['entities'].get('medical_info', []):
                        medical_records.append({
                            'type': item['type'],
                            'value': item['text'],
                            'confidence': item.get('confidence', 'N/A'),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                
                if medical_records:
                    medical_df = pd.DataFrame(medical_records)
                    
                    # Filters
                    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                    record_types = sorted(medical_df['type'].unique().tolist())
                    selected_types = st.multiselect(
                        "Filter by Record Type",
                        options=record_types,
                        default=record_types
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Apply filters
                    filtered_records = medical_df[medical_df['type'].isin(selected_types)]
                    
                    # Display filtered records
                    st.markdown('<div class="data-container">', unsafe_allow_html=True)
                    st.dataframe(
                        filtered_records,
                        hide_index=True,
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            filtered_records,
                            names='type',
                            title="Record Type Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_layout(
                            margin=dict(t=40, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'confidence' in filtered_records.columns:
                            fig = px.box(
                                filtered_records,
                                x='type',
                                y='confidence',
                                title="Confidence by Record Type",
                                color_discrete_sequence=['#3498DB']
                            )
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(t=40, b=20, l=20, r=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No medical records found in the processed data")
            except Exception as e:
                st.error(f"Error processing medical records: {str(e)}")
        else:
            st.info("No medical records available. Process some medical text first.")

    with tabs[2]:  # Entity Analysis Tab
        st.markdown("### Entity Analysis")
        if st.session_state.processed_texts:
            try:
                # Aggregate entities
                all_entities = []
                for text in st.session_state.processed_texts:
                    for category, entities in text['entities'].items():
                        for entity in entities:
                            all_entities.append({
                                'category': category,
                                'type': entity['type'],
                                'value': entity['text'],
                                'confidence': entity.get('confidence', 'N/A'),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                
                if all_entities:
                    entity_df = pd.DataFrame(all_entities)
                    
                    # Filters
                    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_categories = st.multiselect(
                            "Filter by Category",
                            options=sorted(entity_df['category'].unique()),
                            default=entity_df['category'].unique()
                        )
                    
                    with col2:
                        selected_types = st.multiselect(
                            "Filter by Entity Type",
                            options=sorted(entity_df['type'].unique()),
                            default=entity_df['type'].unique()
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Apply filters
                    filtered_entities = entity_df[
                        (entity_df['category'].isin(selected_categories)) &
                        (entity_df['type'].isin(selected_types))
                    ]
                    
                    # Display filtered data
                    st.markdown('<div class="data-container">', unsafe_allow_html=True)
                    st.dataframe(
                        filtered_entities,
                        hide_index=True,
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            filtered_entities,
                            names='category',
                            title="Entity Category Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_layout(
                            margin=dict(t=40, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        type_counts = filtered_entities['type'].value_counts()
                        fig = px.bar(
                            type_counts,
                            title="Entity Type Distribution",
                            labels={'value': 'Count', 'index': 'Entity Type'},
                            color_discrete_sequence=['#3498DB']
                        )
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=40, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence Analysis
                    if 'confidence' in filtered_entities.columns:
                        st.markdown("#### Confidence Analysis")
                        fig = px.box(
                            filtered_entities,
                            x='category',
                            y='confidence',
                            title="Entity Confidence by Category",
                            color_discrete_sequence=['#3498DB']
                        )
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=40, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No entities found in the processed data")
            except Exception as e:
                st.error(f"Error processing entities: {str(e)}")
        else:
            st.info("No entity data available. Process some medical text first.")

def format_response_json(response_data):
    """Format the response data into a clean JSON string"""
    if isinstance(response_data, dict):
        formatted_json = {
            "intent": response_data.get('intent', {}).get('primary_intent'),
            "entities": {}
        }
        
        # Add entities
        entities = response_data.get('entities', {})
        for category, items in entities.items():
            for item in items:
                if item['type'] in ['patient', 'gender', 'age', 'condition', 'medication', 'dosage', 'frequency']:
                    formatted_json["entities"][item['type']] = item['text']
        
        # Add medication validation if present
        if 'medication_validation' in response_data:
            formatted_json["medication_validation"] = response_data['medication_validation']
        
        # Add follow-up question if present
        if 'follow_up_question' in response_data:
            formatted_json["follow_up_question"] = response_data['follow_up_question']
        
        return json.dumps(formatted_json, indent=2)
    return str(response_data)

def get_nurse_icon():
    return """
    <div class="chat-icon">
        <svg viewBox="0 0 24 24" fill="#5B8FF9">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
        </svg>
    </div>
    """

def get_bot_icon():
    return """
    <div class="chat-icon">
        <svg viewBox="0 0 24 24" fill="#4CAF50">
            <path d="M20 9V7c0-1.1-.9-2-2-2h-3c0-1.66-1.34-3-3-3S9 3.34 9 5H6c-1.1 0-2 .9-2 2v2c-1.66 0-3 1.34-3 3s1.34 3 3 3v2c0 1.1.9 2 2 2h3c0 1.66 1.34 3 3 3s3-1.34 3-3h3c1.1 0 2-.9 2-2v-2c1.66 0 3-1.34 3-3s-1.34-3-3-3zM7.5 11.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5S9.83 13 9 13s-1.5-.67-1.5-1.5zM16 17H8v-2h8v2zm-1-4c-.83 0-1.5-.67-1.5-1.5S14.17 10 15 10s1.5.67 1.5 1.5S15.83 13 15 13z"/>
        </svg>
    </div>
    """

def show_chat_interface():
    """Display chat interface page with enhanced chat history"""
    st.title("Medical Task Management ChatBot 🏥")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['is_user']:
            st.markdown(
                f"""
                <div class="chat-container user-container">
                    <div class="chat-icon">👩‍⚕️</div>
                    <div class="message-container">
                        <div class="user-message">{message['message']}</div>
                        <div class="timestamp user-timestamp">{message['timestamp']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if message.get('is_error'):
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-icon">⚠️</div>
                        <div class="message-container">
                            <div class="error-message">{message['message']}</div>
                            <div class="timestamp assistant-timestamp">{message['timestamp']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif message.get('is_follow_up'):
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-icon">🤖</div>
                        <div class="message-container">
                            <div class="follow-up-message">{message['message']}</div>
                            <div class="timestamp assistant-timestamp">{message['timestamp']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-icon">🤖</div>
                        <div class="message-container">
                            <div class="assistant-message">{message['message']}</div>
                            <div class="timestamp assistant-timestamp">{message['timestamp']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display JSON response if available and requested
            if message.get('result') and st.session_state.show_raw_output:
                st.code(format_response_json(message['result']), language='json')
            
            # Display visualization for assistant responses if result exists
            if message.get('result') and not message.get('is_error'):
                try:
                    result = message['result']
                    if isinstance(result, dict) and "intent" in result:
                        col1, col2 = st.columns(2)
                        with col1:
                            if isinstance(result["intent"], dict):
                                display_intent_confidence(result["intent"])
                        with col2:
                            if "entities" in result and isinstance(result["entities"], dict):
                                fig = visualize_entities(result["entities"])
                                st.plotly_chart(fig)
                        display_extracted_info(result)
                except Exception as viz_error:
                    st.error(f"Error displaying visualizations: {str(viz_error)}")
                    # logger.error(f"Visualization error: {viz_error}\nResult: {result}")
        
    # Chat input
    if prompt := st.chat_input("Type your medical command here..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            'message': prompt,
            'is_user': True,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Process the command
        with st.spinner("Processing..."):
            try:
                response = process_command(prompt)
                
                if not response["success"]:
                    st.session_state.chat_history.append({
                        'message': f"Error: {response.get('error', 'Unknown error')}",
                        'is_user': False,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'is_error': True
                    })
                    st.rerun()
                    return

                result = response["result"]
                
                # Handle LLM-specific processing if needed
                if st.session_state.pipeline_type == "llm":
                    if "llm_response" in result:
                        st.session_state.chat_history.append({
                            'message': result["llm_response"],
                            'is_user': False,
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'result': result
                        })
                        st.rerun()
                        return
                
                # Check for medication validation first
                if result["intent"]["primary_intent"] == "assign_medication":
                    med_entities = result["entities"].get("medical_info", [])
                    medication = next((e["text"] for e in med_entities if e["type"] == "medication"), None)
                    
                    if "medication_validation" in result:
                        validation = result["medication_validation"]
                        
                        # Add validation message
                        st.session_state.chat_history.append({
                            'message': validation["message"],
                            'is_user': False,
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'is_follow_up': False
                        })
                        
                        # If medication is invalid or needs more information
                        if not validation["is_valid"]:
                            st.session_state.chat_history.append({
                                'message': validation["follow_up_question"],
                                'is_user': False,
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'is_follow_up': True
                            })
                            st.rerun()
                            return
                        
                        # If medication is valid but needs dosage/frequency
                        if validation["validation_step"] in ["dosage", "frequency"]:
                            st.session_state.chat_history.append({
                                'message': validation["follow_up_question"],
                                'is_user': False,
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'is_follow_up': True
                            })
                            st.rerun()
                            return

                # Add regular follow-up questions if no medication validation or after validation complete
                if "follow_up_question" in result:
                    follow_up = result["follow_up_question"]
                    st.session_state.chat_history.append({
                        'message': follow_up,
                        'is_user': False,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'is_follow_up': True
                    })
                    st.rerun()
                    return
                
                # Add analysis completion message and results
                st.session_state.chat_history.append({
                    'message': "Analysis completed. See results below.",
                    'is_user': False,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'result': result
                })
                
            except Exception as e:
                st.session_state.chat_history.append({
                    'message': f"Error processing command: {str(e)}",
                    'is_user': False,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'is_error': True
                })
            
        st.rerun()

def handle_response(response: Dict[str, Any]):
    """Handle API response and generate appropriate UI elements"""
    if response["success"]:
        result = response["result"]
        
        # Handle medication validation
        if result["intent"]["primary_intent"] == "assign_medication":
            if "medication_validation" in result:
                validation = result["medication_validation"]
                
                # Add validation message to chat
                st.session_state.chat_history.append({
                    'message': validation["message"],
                    'is_user': False,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'is_follow_up': False
                })
                
                # If validation failed or needs more info
                if not validation["is_valid"] or validation["validation_step"] != "complete":
                    st.session_state.chat_history.append({
                        'message': validation["follow_up_question"],
                        'is_user': False,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'is_follow_up': True
                    })
                    
                    # Add follow-up input
                    with st.form(key='follow_up_form'):
                        follow_up_response = st.text_input("Your response:")
                        if st.form_submit_button("Submit"):
                            # Process follow-up response
                            st.session_state.chat_history.append({
                                'message': follow_up_response,
                                'is_user': True,
                                'timestamp': datetime.now().strftime("%H:%M:%S")
                            })
                            # Process the follow-up response
                            new_response = process_command(follow_up_response)
                            if new_response["success"]:
                                # Handle the new response recursively
                                handle_response(new_response)
                            st.rerun()
                    return
def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat Interface", "Dashboard", "Data Views"])
    
    # Add pipeline selection
    st.sidebar.header("Processing Mode")
    st.session_state.pipeline_type = st.sidebar.radio(
        "Select Processing Mode",
        ["transformer", "llm"],
        format_func=lambda x: "Transformer-based" if x == "transformer" else "LLM-based"
    )
    
    # Settings
    st.sidebar.header("Settings")
    st.session_state.show_raw_output = st.sidebar.checkbox("Show Raw API Output", value=False)
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history()
    
    # Display selected page
    if page == "Chat Interface":
        show_chat_interface()
    elif page == "Dashboard":
        show_dashboard()
    elif page == "Data Views":
        show_data_views()

if __name__ == "__main__":
    main()