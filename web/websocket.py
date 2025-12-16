import json
import asyncio
from typing import List
from fastapi import WebSocket
from openai import OpenAI

class InvestigationWebSocketManager:
    def __init__(self, client: OpenAI):
        self.active_connections: List[WebSocket] = []
        self.client = client

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific websocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def send_worker_progress(self, websocket: WebSocket, worker_num: int, total_workers: int, status: str):
        """Send worker progress update"""
        await self.send_message({
            "type": "worker_progress",
            "worker": worker_num,
            "total": total_workers,
            "status": status,
            "progress": worker_num / total_workers
        }, websocket)

    async def send_stage_update(self, websocket: WebSocket, stage: str, message: str):
        """Send stage update (workers, synthesizing, streaming)"""
        await self.send_message({
            "type": "stage",
            "stage": stage,
            "content": message
        }, websocket)

    async def send_stream_start(self, websocket: WebSocket):
        """Signal that streaming response is starting"""
        await self.send_message({
            "type": "stream_start"
        }, websocket)

    async def send_chunk(self, websocket: WebSocket, chunk: str):
        """Send streaming response chunk"""
        await self.send_message({
            "type": "chunk",
            "content": chunk
        }, websocket)

    async def send_stream_end(self, websocket: WebSocket, question: str):
        """Signal that streaming response is complete"""
        await self.send_message({
            "type": "stream_end",
            "question": question,
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)

    async def send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to client"""
        await self.send_message({
            "type": "error",
            "content": error_message
        }, websocket)

    async def send_response(self, websocket: WebSocket, response: str, question: str):
        """Send complete response (fallback for non-streaming)"""
        await self.send_message({
            "type": "response",
            "content": response,
            "question": question,
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)

