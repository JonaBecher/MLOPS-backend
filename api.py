from typing import Dict, re

from fastapi import FastAPI, WebSocket
import base64

from starlette.websockets import WebSocketDisconnect

app = FastAPI()
connections: Dict[str, WebSocket] = {}

@app.get("/")
def test():
    return None


@app.get("/connectedDevices")
async def connectedDevices():
    return [key for key in connections.keys()]


def decode_base64(data, altchars='+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(r'[^a-zA-Z0-9%s]+' % altchars, '', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += '='* (4 - missing_padding)
    return base64.b64decode(data, altchars)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            imgdata = decode_base64(data["base64"])
            filename = 'some_image.png'  # I assume you have a way of picking unique filenames
            with open(filename, 'wb') as f:
                f.write(imgdata)
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print(f"""{client_id} disconnected.""")
        del connections[client_id]