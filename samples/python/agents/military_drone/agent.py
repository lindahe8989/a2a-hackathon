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
from google.adk.agents.agent_card import AgentCard # for agent card creation 
from common.server import A2AServer 
from common.types import (
    SendTaskRequest, 
    TaskSendParams, 
    Message,
    TextPart, 
    AgentCard
)

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

def search_and_delegate(task_description: str, current_mission: dict) -> dict[str, Any]:
    """
    Search for agents that can better handle specific tasks and delegate using A2A protocol.
    
    Args:
        task_description: Description of the specific task needed
        current_mission: Current mission details
        
    Returns:
        dict with delegation status and results
    """
    try:
        # Use existing find_matching_agent function to search agent cards
        matching_agent = AgentCard = await find_matching_agent(task_description)
        
        if matching_agent:
            try: 
                # Create task request following A2A protocol 
                task_params = TaskSendParams(
                    id=str(uuid4()),  # Generated ID for the task
                    sessionId=str(uuid4().hex),  # Session ID
                    message=Message(
                        role="user",  # Must be "user" or "agent"
                        parts=[
                            TextPart(
                                type="text",  # This is fixed
                                text=task_description,  # Your actual message
                                metadata=None  # Optional
                            )
                        ],
                        metadata=None  # Optional
                    ),
                    acceptedOutputModes=matching_agent.defaultOutputModes,
                    metadata=current_mission  # Optional: can include context
                )

                request = SendTaskRequest(params=task_params)

                # Create server connection to the matched agent 
                delegate_server = A2AServer(
                    agent_card=matching_agent,
                    host=matching_agent.url.split("://")[1].split(":")[0], 
                    port=int(matching_agent.url.split(":")[-1].strip("/"))
                )

                # Send task and await response 
                response = await delegate_server.send_task(request) 

                return {
                    "status": "delegated", 
                    "delegated_to": matching_agent.name, 
                    "agent_url": matching_agent.url, 
                    "original_mission": current_mission, 
                    "delegation_message": f"Task delegated to {matching_agent.name} for specialized handling",
                    "delegation_response": response.result
                }
            except Exception as e:
                return {
                    "status": "delegation_failed",
                    "message": f"Failed to delegate to {matching_agent.name}: {str(e)}",
                    "agent_url": matching_agent.url
                }
        else:
            return {
                "status": "no_match",
                "message": "No agents found with required capabilities"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in agent discovery: {str(e)}"
        }
             

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
            model="claude-3-sonnet",
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
                search_and_delegate  # Added new tool
            ],
            card=AgentCard(
                name="Military Drone Agent",
                description="This agent handles military drone operations including mission planning and execution.",
                url="http://localhost:10002/",
                version="1.0.0",
                capabilities={
                    "streaming": True,
                    "pushNotifications": False,
                    "stateTransitionHistory": False
                },
                defaultInputModes=["text", "text/plain"],
                defaultOutputModes=["text", "text/plain"],
                skills=[
                    {
                        "id": "process_drone_mission",
                        "name": "Drone Mission Control",
                        "description": "Handles military drone operations including mission planning, execution, and monitoring.",
                        "tags": ["military", "drone", "surveillance", "reconnaissance"]
                    }
                ]
            )
        )