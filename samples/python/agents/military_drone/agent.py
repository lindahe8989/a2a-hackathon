import json
import random
from typing import Any, AsyncIterable, Dict, Optional
from google.adk.agents.llm_agent import LlmAgent # Base LLM agent class 
from google.adk.tools.tool_context import ToolContext # Context for tool execution 
from google.adk.artifacts import InMemoryArtifactService # temporary storage for artifacts 
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService # Memory management 
from google.adk.runners import Runner # executes agent operations 
from google.adk.sessions import InMemorySessionService # manages agent sessions 
from google.genai import types # google ai types for content handling 
# from google.adk.agents.agent_card import AgentCard  # Use this instead of common.types AgentCard
from common.server import A2AServer 
from common.types import (
    SendTaskRequest, 
    TaskSendParams, 
    Message,
    TextPart, 
    AgentSkill,  # Add this import
    AgentCard
)
from a2a.find_matching_agent import find_matching_agent 
from uuid import uuid4
import logging 

# Set up logging at the top of the file
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Cache for mission IDs
# Global set to track active mission IDs (note: this is temporary, resets on restart)
mission_ids = set()

def create_mission(location: Optional[str] = None, objective: Optional[str] = None, drone_type: Optional[str] = None) -> dict[str, Any]:
    """Create a new drone mission."""

    """ 
    Creates a new drone mission with unique ID and specified parameters. 

    Args: 
        location: Target coordinates or area description 
        objective: Mission purpose or goal 
        drone_type: Type of drone ('scout' or 'attack') 
    
    Returns: 
        Dictionary containing mission details and ID 
    """
    mission_id = "mission_" + str(random.randint(1000000, 9999999))
    mission_ids.add(mission_id)
    return {
        "mission_id": mission_id,
        "location": location or "<target location>",
        "objective": objective or "<mission objective>",
        "drone_type": drone_type or "<scout/attack>",
    }

def return_form(
    form_request: dict[str, Any],    
    tool_context: ToolContext,
    instructions: Optional[str] = None) -> dict[str, Any]:
    """Returns a structured mission form."""
    """ 
    Creates a structured form for mission data collection/confirmation. 

    Args: 
        form_request: Initial form data 
        tool_context: Context for tool execution 
        instructions: Optional instructions for form handling 

    Returns: 
        JSON string containing form structure and data 
    """
    if isinstance(form_request, str):
        form_request = json.loads(form_request)

    tool_context.actions.skip_summarization = True
    tool_context.actions.escalate = True
    form_dict = {
        'type': 'form',
        'form': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'Target location',
                    'title': 'Location',
                },
                'objective': {
                    'type': 'string',
                    'description': 'Mission objective',
                    'title': 'Objective',
                },
                'drone_type': {
                    'type': 'string',
                    'description': 'Type of drone (scout/attack)',
                    'title': 'Drone Type',
                },
                'mission_id': {
                    'type': 'string',
                    'description': 'Mission ID',
                    'title': 'Mission ID',
                },
            },
            'required': list(form_request.keys()),
        },
        'form_data': form_request,
        'instructions': instructions,
    }
    return json.dumps(form_dict)

def execute_mission(mission_id: str) -> dict[str, Any]:
    """Execute the drone mission."""

    """ 
    Executes a mission if the ID exists in our tracking set. 
    Simple implementation that just checks ID validity and returns status. 
    """
    if mission_id not in mission_ids:
        return {"mission_id": mission_id, "status": "Error: Invalid mission_id."}
    return {"mission_id": mission_id, "status": "in_progress"}

async def search_and_delegate(task_description: str, current_mission: dict) -> dict[str, Any]:
    """
    Search for specialized agents and delegate the mission if a match is found.
    
    Args:
        task_description: Description of the specialized capabilities needed
        current_mission: Current mission details
        
    Returns:
        dict: Delegation results including status and any error messages
    """
    try:
        # Log before search
        logger.debug(f"Searching for agent with task: {task_description}")
        
        matching_agent = find_matching_agent(task_description)
        
        # Log the found agent
        logger.debug(f"Found matching agent: {matching_agent}")
        logger.debug(f"Type of matching_agent: {type(matching_agent)}")

        if matching_agent:
            try:

                # Log before creating AgentCard
                logger.debug("Attempting to create AgentCard with data:")
                logger.debug(f"Name: {matching_agent.get('name', 'NOT FOUND')}")
                logger.debug(f"Skills: {matching_agent.get('skills', 'NOT FOUND')}")

                agent_card = AgentCard(
                    name=matching_agent["name"],
                    description=matching_agent["description"],
                    version=matching_agent["version"],
                    skills=matching_agent["skills"],
                    url=matching_agent["url"],
                    capabilities=matching_agent["capabilities"],
                    defaultInputModes=matching_agent["defaultInputModes"],
                    defaultOutputModes=matching_agent["defaultOutputModes"]
                )
                return await delegate_to_agent(agent_card, current_mission)
            except KeyError as ke:
                print("unable to find matching agent")
                return {
                    "status": "delegation_failed",
                    "message": f"Missing required field in agent data: {str(ke)}",
                    "skill_details": matching_agent.get("skills", [{}])[0]
                }
            except Exception as e:
                return {
                    "status": "delegation_failed",
                    "message": f"Failed to delegate to {matching_agent.get('name', 'unknown agent')}: {str(e)}",
                    "skill_details": matching_agent.get("skills", [{}])[0]
                }
        else:
            return {
                "status": "delegation_failed",
                "message": "No matching specialized agents found for this task",
                "skill_details": {}
            }
    except Exception as e:
        return {
            "status": "delegation_failed",
            "message": f"Error during agent search: {str(e)}",
            "skill_details": {}
        }

def delegate_to_agent(agent_card: AgentCard, current_mission: dict) -> dict[str, Any]:
    return 1 

class DroneAgent:
    """Agent that handles drone missions."""

    ## define what types of content this agent can handle 
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """ 
        Initialize the agent with necessary services and configuration. 
        Sets up memory, session management, and artifact storage. 
        """
        self._agent = self._build_agent() # Create LLM agent with specific instructions 
        self._user_id = "remote_agent" # identifier for this agent instance 
        # Instantiate runner with required services 
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def invoke(self, query, session_id) -> str:
        """ 
        Synchronously process a user query and return response. 

        Args: 
            query: the user's text input (e.g., "Send scout drone to coordinates X,Y") 
            session_id: Unique identifier for this conversation 

        Returns: 
            String response from the agent 
        """

        # Get existing session or None if it doesn't exist 
        session = self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )

        # Create content object from user's query 
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=query)]
        )

        # Create new session if one doesn't exist 
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        
        # Process the query and get response events 
        events = self._runner.run(
            user_id=self._user_id, session_id=session.id, new_message=content
        )
        
        # Return empty string if no valid response 
        if not events or not events[-1].content or not events[-1].content.parts:
            return ""
        
        # Combine all text parts of the response 
        return "\n".join([p.text for p in events[-1].content.parts if p.text])

    async def stream(self, query, session_id) -> AsyncIterable[Dict[str, Any]]:

        """ 
        Asynchronously process a user query with streaming updates. 

        Args: 
            query: The user's text input 
            session_id: Unique identifier for this conversation 

        Yield: 
            Dictionary containing status updates and responses 
        """

        session = self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=query)]
        )
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        # Process query with streaming updates 
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                # Handle final response from agent 
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join([p.text for p in event.content.parts if p.text])
                elif (
                    event.content
                    and event.content.parts
                    and any([True for p in event.content.parts if p.function_response])):
                    response = next((p.function_response.model_dump() for p in event.content.parts))
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "Processing the drone mission request...",
                }

    def _build_agent(self) -> LlmAgent:

        """ 
        Configure and create the LLM agent with specific instructions and tools. 

        Returns: 
            Configured LlmAgent instance ready to handle drone missions. 
        """
        return LlmAgent(
            model="gemini-2.0-flash-001",
            name="drone_agent",
            description="This agent handles military drone operations and can find specialized agents for specific tasks",
            instruction="""
            You are an agent who handles military drone operations.

            When you receive a mission request:
            1. Create a new mission form using create_mission(). Only provide default values if they are provided by the user, otherwise use an empty string.
                - Location: target coordinates/area
                - Objective: mission purpose
                - Drone Type: scout or attack

            Once you created the form, return the result of calling return_form with the form data from the create_mission call.

            When you receive the filled-out form back:
            1. Analyze if the mission requires specialized capabilities:
               - For specialized surveillance: thermal imaging, signal intelligence
               - For complex attack missions: precision targeting, EW capabilities
               - For reconnaissance: deep penetration, stealth operations
            
            2. If specialized capabilities are needed:
               - Use search_and_delegate() to find a more capable agent
               - Provide detailed task description for matching
               - Pass current mission details for context
               - Return delegation results to the user

            3. If basic mission capabilities are sufficient:
               - Validate all required information
               - Use execute_mission() to begin operations
               - Include mission_id and status in your response

            Examples of when to search for specialized agents:
            - "Need thermal imaging surveillance of target area"
            - "Require stealth drone for deep reconnaissance"
            - "Need electronic warfare capable drone"
            """,
            tools=[
                create_mission,
                execute_mission,
                return_form,
                search_and_delegate
            ],
        )