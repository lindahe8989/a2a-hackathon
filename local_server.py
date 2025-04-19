from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Agent server is running"}

@app.get("/agent-card")
async def agent_card():
    return {
        "name": "Test Agent",
        "description": "A test agent for development",
        "capabilities": ["test"],
        "version": "0.1.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)