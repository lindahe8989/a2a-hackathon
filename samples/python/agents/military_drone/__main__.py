from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from task_manager import AgentTaskManager
from agent import DroneAgent
import click
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10002)
def main(host, port):
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        
        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="process_drone_mission",
            name="Drone Mission Control",
            description="Handles military drone operations including mission planning, execution, and monitoring.",
            tags=["military", "drone", "surveillance", "reconnaissance"],
            examples=[
                "Send a scout drone to coordinates 35.6895, 139.6917 for surveillance",
                "Deploy an attack drone to target location Alpha-7"
            ],
        )
        agent_card = AgentCard(
            name="Military Drone Agent",
            description="This agent handles military drone operations including mission planning and execution.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=DroneAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DroneAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=DroneAgent()),
            host=host,
            port=port,
        )
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)

if __name__ == "__main__":
    main()
