from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.tools import tool
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Union, Optional
import operator
import json
import time
import os
from IPython.display import Image, display
from langchain.chains import LLMChain
import requests
import dotenv

dotenv.load_dotenv()

PROMPT_DIR = "system_prompts"


class AgentState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    current_agent: str
    waiting_for_user: bool
    exit_requested: bool
    last_processed_message_idx: int
    timeout_seconds: float
    current_query: str  # Add this new field
    conversation_history: List[Dict[str, str]]  # Add this new field

class InteractiveLangGraph:
    def __init__(self, google_api_key=None, tavily_api_key=None, alphavantage_api_key=None,prompt_dir=PROMPT_DIR):
        """Initialize with API keys either from parameters or environment variables."""
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self.alphavantage_api_key = alphavantage_api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        self.prompt_dir = prompt_dir
        self.llm = None
        self.workflow = None
        self.workflow_state = None
        self.last_activity = time.time()
        self.message_counter = 0  # New global counter
        self.message_history = []
        self.conversation_history = [] 
    
    def _check_exit_request(self, query: str) -> bool:
        """Check if the user is requesting to exit the conversation."""
        exit_keywords = ["exit", "quit", "done", "bye", "goodbye", "q"]
        return any(keyword == query.lower().strip() for keyword in exit_keywords)
    
    def _load_prompt_from_file(self, filename: str) -> str:
        """Load a prompt from a file."""
        try:
            filepath = os.path.join(self.prompt_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Prompt file not found: {filepath}")
                
            with open(filepath, 'r') as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Error loading prompt from {filename}: {str(e)}")
    
    def _setup_tools_and_agents(self):
        """Set up the LLM, tools, and agents."""
        # Initialize LLM
        print(self.google_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key= self.google_api_key
        )
        
        # Initialize tools
        tavily_search_tool = TavilySearchResults(
            max_results=5,
            topic="general",
            tavily_api_key=self.tavily_api_key
        )
        
        alpha_vantage = AlphaVantageAPIWrapper(
            alphavantage_api_key=self.alphavantage_api_key
        )
        
        # Create specialized tools
        @tool
        def get_stock_data(symbol: str) -> str:
            """Get stock information for a specific company symbol."""
            try:
                api_key = self.alphavantage_api_key  # Use your API key variable
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
                response = requests.get(url)
                data = response.json()
                return str(data)
            except Exception as e:
                return f"Error retrieving stock data: {str(e)}"


        @tool
        def get_company_overview(symbol: str) -> str:
            """Get company overview for a specific stock symbol."""
            try:
                api_key = self.alphavantage_api_key  # Use your API key variable
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
                response = requests.get(url)
                data = response.json()
                return str(data)
            except Exception as e:
                return f"Error retrieving company overview: {str(e)}"

        # @tool
        # def get_price_change(symbol: str, days: int = 1) -> str:
        #     """Get price change information for a specific company symbol over a specified number of days."""
        #     try:
        #         # Get time series data
        #         data = alpha_vantage.run(f"time_series_daily_adjusted?symbol={symbol}")
                
        #         # Parse the response
        #         time_series = json.loads(data)["Time Series (Daily)"]
                
        #         # Get dates sorted in descending order
        #         dates = sorted(time_series.keys(), reverse=True)
                
        #         # Ensure we have enough data points
        #         if len(dates) <= days:
        #             return f"Not enough data available. Requested {days} days but only have {len(dates)} days of data."
                
        #         # Get closing prices
        #         current_price = float(time_series[dates[0]]["4. close"])
        #         previous_price = float(time_series[dates[days]]["4. close"])
                
        #         # Calculate change
        #         change = current_price - previous_price
        #         percent_change = (change / previous_price) * 100
                
        #         return f"Symbol: {symbol}\nCurrent price: ${current_price:.2f}\nPrice {days} day(s) ago: ${previous_price:.2f}\nChange: ${change:.2f} ({percent_change:.2f}%)"
        #     except Exception as e:
        #         return f"Error retrieving price change data: {str(e)}"

        # Create search tool wrapper
        search_tool = Tool(
            name="web_search",
            description="Search the web for information on a given topic.",
            func=tavily_search_tool.invoke
        )
        
        # Define tool collections
        self.research_tools = [search_tool]
        self.finance_tools = [get_stock_data, get_company_overview]
        
         # Load prompts from files
        research_system_prompt = self._load_prompt_from_file("research_system_prompt.txt")
        finance_system_prompt = self._load_prompt_from_file("finance_system_prompt.txt")
        supervisor_system_prompt = self._load_prompt_from_file("supervisor_system_prompt.txt")
        
        
        # Create agent prompts
        self.research_agent_prompt = PromptTemplate(template=research_system_prompt, 
                                                    input_variables=["input", "agent_scratchpad", "tools", "tool_names"])

        self.finance_agent_prompt = PromptTemplate(template=finance_system_prompt, 
                                                   input_variables=["input", "agent_scratchpad", "tools", "tool_names"])

        self.supervisor_prompt = PromptTemplate(template=supervisor_system_prompt,
                                                input_variables=["input", "agent_scratchpad", "tools", "tool_names"])
        print(self.research_agent_prompt)
        
        # Create agents
        self.research_agent = create_react_agent(
            llm=self.llm, 
            tools=self.research_tools, 
            prompt=self.research_agent_prompt
        )
        
        self.finance_agent = create_react_agent(
            llm=self.llm, 
            tools=self.finance_tools, 
            prompt=self.finance_agent_prompt
        )
        
        self.supervisor_prompt = PromptTemplate(
        template=supervisor_system_prompt + "\n\nUser Query: {input}",
        input_variables=["input"]
    )
    
        self.supervisor_chain = LLMChain(
            llm=self.llm,
            prompt=self.supervisor_prompt
        )
    
    def _create_workflow(self):
        """Create the workflow graph."""
        # Define node functions
        def supervisor(state: AgentState):
            """Supervisor that decides which agent should handle the query."""
            query = state["current_query"]
            history = state["conversation_history"]
            
            # Determine the appropriate agent based on both current query and history
            result = self.supervisor_chain.invoke({
                "input": query,
                "history": "\n".join([f"{m['role']}: {m['content']}" for m in history[:-1]])
            })
            
            # Determine next agent
            response_text = result["text"].strip().lower()
            next_agent = "finance" if "finance" in response_text else "research"
            
            return {**state, "current_agent": next_agent} # Update the state with the next agent       


        def finance(state: AgentState):
            """Finance agent that handles financial queries."""
            query = state["current_query"]
            history = state["conversation_history"]
            
            formatted_history = ""
            if len(history) > 0:
                formatted_history = "Previous conversation:\n" + "\n".join([
                    f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                    for msg in history[-4:]  # Last 4 messages for context
                ]) + "\n\nBased on this previous conversation, answer: "
            
            # Process the query with awareness of conversation history
            result = AgentExecutor.from_agent_and_tools(
                agent=self.finance_agent,
                tools=self.finance_tools,
                verbose=True,
                handle_parsing_errors=True  
            ).invoke({
                "input": formatted_history + query,
                "chat_history": history[:-1],  
                "agent_scratchpad": [],  
                "tools": "\n".join([f"{t.__name__ if hasattr(t, '__name__') else t.name}: {t.description if hasattr(t, 'description') else ''}" for t in self.finance_tools]),  
                "tool_names": ", ".join([t.__name__ if hasattr(t, '__name__') else t.name for t in self.finance_tools])  
            })
            
            self.conversation_history.append({"role": "assistant", "content": result["output"]})
            
            
            return {**state, "current_agent": "complete"}
        
        def research(state: AgentState):
            """Research agent that handles financial queries."""
            query = state["current_query"]
            history = state["conversation_history"]
            
            formatted_history = ""
            if len(history) > 0:
                formatted_history = "Previous conversation:\n" + "\n".join([
                    f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                    for msg in history[-4:]  # Last 4 messages for context
                ]) + "\n\nBased on this previous conversation, answer: "
    
            # Process the query with awareness of conversation history
            result = AgentExecutor.from_agent_and_tools(
                agent=self.research_agent,
                tools=self.research_tools,
                verbose=True,
                handle_parsing_errors=True  # Keep this parameter
            ).invoke({
                "input": formatted_history + query,
                "chat_history": history[:-1],  # Keep as a list instead of joining
                "agent_scratchpad": [],  # Keep this parameter
                "tools": "\n".join([f"{t.__name__ if hasattr(t, '__name__') else t.name}: {t.description if hasattr(t, 'description') else ''}" for t in self.research_tools]),  # Keep tools
                "tool_names": ", ".join([t.__name__ if hasattr(t, '__name__') else t.name for t in self.research_tools])  # Keep tool_names
            })
            
            # Add the response to the global history
            self.conversation_history.append({"role": "assistant", "content": result["output"]})
            
            # Mark as complete
            return {**state, "current_agent": "complete"}

        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor", supervisor)
        workflow.add_node("finance", finance)
        workflow.add_node("research", research)

        # Add conditional edges
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["current_agent"],
            {
                "finance": "finance",
                "research": "research"
            }
        )

        # Agents directly end the workflow
        workflow.add_edge("finance", END)
        workflow.add_edge("research", END)

        workflow.set_entry_point("supervisor")
        
        # Compile the graph
        self.workflow = workflow.compile()
        
    def initialize(self):
        """Set up the LLM, tools, agents, and workflow."""
        # Check for API keys
        if not self.google_api_key or not self.tavily_api_key or not self.alphavantage_api_key:
            raise ValueError("Missing API keys. Please provide Google, Tavily, and Alpha Vantage API keys.")
            
        # Set up tools and agents
        self._setup_tools_and_agents()
        
        # Create the workflow
        self._create_workflow()
        
        return self
        
    def visualize_graph(self):
        """Visualize the workflow graph."""
        if self.workflow is None:
            print("Workflow not initialized. Call initialize() first.")
            return
            
        # Get and display the graph visualization
        mermaid_code = self.workflow.get_graph().draw_mermaid()
        print("Mermaid graph code:")
        print(mermaid_code)
        print("\nYou can paste this code into a Mermaid renderer like https://mermaid.live")
        
    def process_query(self, query: str):
        """Process a user query as part of an ongoing conversation."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Exit early if requested
        if self._check_exit_request(query):
            return self.conversation_history
        
        # Create a state for this turn
        state = {
            "current_agent": "",  # Will be set by supervisor
            "waiting_for_user": False,
            "exit_requested": self._check_exit_request(query),
            "last_processed_message_idx": 0,
            "timeout_seconds": 300,
            "current_query": query,  # This was missing
            "conversation_history": self.conversation_history  # This was missing
        }
        
        # Run a single turn of conversation
        result = self.workflow.invoke(state)
        
        # Return the updated conversation history
        return self.conversation_history
    
    def get_last_response(self):
        """Get the last assistant response."""
        for message in reversed(self.message_history):
            if message["role"] == "assistant":
                return message["content"]
        return None
        
    def print_conversation(self):
        """Print the entire conversation history."""
        if self.workflow_state is None or not self.workflow_state.get("messages"):
            print("No conversation history.")
            return
            
        # Print all messages in the conversation
        for message in self.workflow_state["messages"]:
            role = "User" if message["role"] == "user" else "Assistant"
            print(f"\n[{role}]: {message['content']}")

# Example usage
def main():
    """Example of how to use the InteractiveLangGraph class."""
    # Initialize with your API keys
    agent_system = InteractiveLangGraph(
        google_api_key="your-gemini-api-key",  # Or set via environment variable
        tavily_api_key="your-tavily-api-key",  # Or set via environment variable
        alphavantage_api_key="your-alpha-vantage-api-key"  # Or set via environment variable
    )
    
    # Initialize the system
    agent_system.initialize()
    
    # Visualize the graph
    print("Graph visualization:")
    agent_system.visualize_graph()
    
    # Start the interactive loop
    print("\nChat with the agent system. Type 'exit' to end.")
    
    # First query
    query = input("\nYou: ")
    
    while not agent_system._check_exit_request(query):
        # Process the query
        agent_system.process_query(query)
        
        # Print the response
        response = agent_system.get_last_response()
        print(f"\nAssistant: {response}")
        
        # Get the next user input
        query = input("\nYou: ")
    
    # Print final message
    print("\nAssistant: Conversation ended.")

if __name__ == "__main__":
    main()