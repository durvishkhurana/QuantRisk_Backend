from collections import defaultdict
from fastapi import WebSocket


class PortfolioSocketManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, portfolio_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[portfolio_id].add(websocket)

    def disconnect(self, portfolio_id: str, websocket: WebSocket) -> None:
        self._connections[portfolio_id].discard(websocket)

    async def broadcast(self, portfolio_id: str, payload: dict) -> None:
        stale: list[WebSocket] = []
        for ws in self._connections.get(portfolio_id, set()):
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._connections[portfolio_id].discard(ws)


socket_manager = PortfolioSocketManager()
