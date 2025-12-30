"""
Streamlit Frontend Application
-------------------------------
User interface for SourceBot supplier search.
"""

import sys
from pathlib import Path
import streamlit as st

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------
ROOT_DIR = Path( __file__ ).resolve().parents[1]
if str( ROOT_DIR ) not in sys.path :
    sys.path.insert( 0, str( ROOT_DIR ) )

from sourcebot.orchestrator import handle
from sourcebot.responses import format_response

# -------------------------------------------------
# Streamlit UI Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="SourceBot — Supplier Search Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title( "🤖 SourceBot" )
st.caption( "AI-powered supplier sourcing assistant" )

# Session management
SESSION_ID = "user1"

# Initialize session state for chat history
if "messages" not in st.session_state :
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages :
    with st.chat_message( message["role"] ) :
        st.markdown( message["content"] )

# Chat input
query = st.chat_input(
    "Ask me about suppliers, e.g., 'LED bulb suppliers in Vietnam under $2 with ISO certification'"
)

if query :
    # Add user message to chat history
    st.session_state.messages.append( {"role" : "user", "content" : query} )

    # Display user message
    with st.chat_message( "user" ) :
        st.markdown( query )

    # Get bot response
    with st.chat_message( "assistant" ) :
        with st.spinner( "Searching suppliers..." ) :
            result = handle( SESSION_ID, query )

        # Format and display response
        if isinstance( result, list ) :
            response = format_response( result )
        else :
            response = str( result )

        st.markdown( response )

    # Add assistant response to chat history
    st.session_state.messages.append( {"role" : "assistant", "content" : response} )

# Sidebar with info
with st.sidebar :
    st.header( "About" )
    st.write( "SourceBot helps you find suppliers based on:" )
    st.write( "- Product type" )
    st.write( "- Price range" )
    st.write( "- Location" )
    st.write( "- Certifications" )

    if st.button( "Clear Chat History" ) :
        st.session_state.messages = []
        st.rerun()