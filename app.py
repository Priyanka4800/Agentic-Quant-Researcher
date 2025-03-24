import streamlit as st
import os
from agentic_market_explainer import InteractiveLangGraph
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

def initialize_agent():
    """Initialize the agent with API keys."""
    try:
        # Load API keys from Streamlit secrets or environment
        google_api_key = os.environ.get("GEMINI_API_KEY")
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        alphavantage_api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        
        # Initialize agent
        agent_system = InteractiveLangGraph(
            google_api_key=google_api_key,
            tavily_api_key=tavily_api_key,
            alphavantage_api_key=alphavantage_api_key
        )
        agent_system.initialize()
        return agent_system
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Market Research Assistant",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Market Research Assistant")
    st.markdown("Ask questions about stocks, companies, and market trends.")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()
        
    # If agent failed to initialize
    if st.session_state.agent is None:
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant can help you with:
        - Stock information and prices
        - Company overviews
        - Market research
        
        Built with LangGraph and Gemini.
        """)
        
        # Reset conversation button
        if st.button("Reset Conversation"):
            st.session_state.conversation = []
            st.session_state.agent.conversation_history = []
            st.rerun()
    
    # Display conversation history
    for message in st.session_state.conversation:
        role = message["role"]
        content = message["content"]
        
        # Map role to Streamlit chat role
        display_role = "user" if role == "user" else "assistant"
        
        with st.chat_message(display_role):
            st.write(content)
    
    # Input box for new queries
    query = st.chat_input("Ask about stocks, companies, or market trends...")
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": query})
        
        # Get agent response with error handling
        try:
            with st.spinner("Researching..."):
                if not hasattr(st.session_state.agent, 'conversation_history'):
                    st.session_state.agent.conversation_history = []
                    
                # Add user message to agent's history first
                st.session_state.agent.conversation_history.append({"role": "user", "content": query})
                
                # Process query
                st.session_state.agent.process_query(query)
                
                # Instead of get_last_response, get the last assistant message directly
                for msg in reversed(st.session_state.agent.conversation_history):
                    if msg["role"] == "assistant":
                        response = msg["content"]
                        break
                else:
                    response = "No response generated"
        except Exception as e:
            response = f"Error: {str(e)}"
            
        # Add agent response to conversation
        st.session_state.conversation.append({"role": "assistant", "content": response})
        
        # Display agent response
        with st.chat_message("assistant"):
            st.write(response)
        
        # Force Streamlit to rerun and update the UI
        st.rerun()

if __name__ == "__main__":
    main()