# from fastapi import FastAPI, Header, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# import uvicorn

# app = FastAPI(title="Incident Dashboard API")


# # 3. Add the middleware to the app
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],            # Allows specific origins (or ["*"] for all)
#     allow_credentials=True,
#     allow_methods=["*"],              # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"],              # Allows all headers (including your X-API-Key)
# )

# # In-memory store for testing
# incidents_db = []

# # --- Models ---
# class Analysis(BaseModel):
#     hazards: List[str]
#     severity: str

# class Routing(BaseModel):
#     services: str

# class IncidentPayload(BaseModel):
#     event_id: str
#     timestamp: str
#     location: dict
#     analysis: Analysis
#     routing: Routing
#     license_plate:Optional[str]

# # --- POST: Receive incident from LangGraph ---
# @app.post("/api/v1/incidents")
# def receive_incident(payload: IncidentPayload, x_api_key: Optional[str] = Header(None)):
#     if x_api_key != "your_secure_hackathon_key":
#         raise HTTPException(status_code=401, detail="Invalid API Key")
    
#     incidents_db.append(payload.dict())
#     print(f"✅ Incident stored: {payload.event_id}")
#     return {"message": "Incident received", "event_id": payload.event_id}







from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json


app = FastAPI(title="Incident Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# --- Models & DB (Same as before) ---
incidents_db = []

class Analysis(BaseModel):
    hazards: List[str]
    severity: str

class Routing(BaseModel):
    services: str

class IncidentPayload(BaseModel):
    event_id: str
    timestamp: str
    location: dict
    analysis: Analysis
    routing: Routing
    license_plates: Optional[List[bytes]]
    detected_faces:Optional[List[bytes]]

# --- POST: Receive and then BROADCAST ---
@app.post("/api/v1/incidents")
async def receive_incident(payload: IncidentPayload, x_api_key: Optional[str] = Header(None)):
    if x_api_key != "your_secure_hackathon_key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    incident_data = payload.dict()
    incidents_db.append(incident_data)
    
    # 🔥 This is the magic part: Send to all open browser tabs
    await manager.broadcast({"type": "NEW_INCIDENT", "data": incident_data})
    
    return {"status": "sent to dashboard", "event_id": payload.event_id}

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- GET: View all incidents ---
@app.get("/api/v1/incidents")
def get_all_incidents():
    return {"total": len(incidents_db), "incidents": incidents_db}

# --- GET: View a specific incident by event_id ---
@app.get("/api/v1/incidents/{event_id}")
def get_incident(event_id: str):
    for inc in incidents_db:
        if inc["event_id"] == event_id:
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")

if __name__ == "__main__":
    uvicorn.run("dashboard_api:app", host="127.0.0.1", port=8000, reload=True)