import asyncio
import datetime
import math
import os
import json
import threading

import numpy as np
import pandas as pd
from mlflow.entities import Run
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

import db
from db import setup_db, insert_or_update_entity, Model, Device, Project, ModelMetrics, get_entity, update_entity
from sqlalchemy import text
from helper import decode_base64, make_dir, delete_dir, convert_model_to_js, zip_folder
from mlflow_api import get_experiments, get_runs, get_run, download_model

from typing import Dict
from fastapi import FastAPI, WebSocket, Request, HTTPException, Response

from starlette.websockets import WebSocketDisconnect

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    setup_db()


async def poll_backend(interval=10):
    while True:
        try:
            await sync_with_backend()
        except Exception as e:
            print(f"Error during sync: {e}")
            # Implement backoff strategy here if needed
        await asyncio.sleep(interval)


async def sync_with_backend():
    experiments = await get_experiments()
    for experiment in experiments:
        experiment_id = experiment["_experiment_id"]
        make_dir(f'./projects/{experiment_id}/')
        insert_or_update_entity(Project, experiment_id, {"name": experiment["_name"]})

        runs = await get_runs(experiment_id = experiment_id)
        for run in runs:
            run_id = run["_info"].run_id
            insert_or_update_entity(Model, run_id, {"name": run["_info"].run_name, "projectId": experiment_id})


app = FastAPI()
connections: Dict[int, WebSocket] = {}
global dark
dark = False


def start_polling_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(poll_backend())


# Start the polling in a separate thread
polling_thread = threading.Thread(target=start_polling_in_thread)
polling_thread.start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/projects")
def get_projects():
    return get_entity(Project, None)


@app.get("/{projectId}/models")
def get_models(projectId: int):
    models: [Model] = get_entity(Model, projectId, "projectId", first=False)
    for idx, model in enumerate(models):
        models[idx] = models[idx].__dict__
        additional_data: Run = asyncio.run(get_run(model.id))
        models[idx]["params"] = additional_data.data.params
        models[idx]["metrics"] = additional_data.data.metrics
    return models


@app.get("/{projectId}/getStats")
def get_models(projectId: int):

    sql = text(f"SELECT * FROM model_metrics WHERE created_at >= datetime('now', '-20 minutes') AND created_at <= datetime('now')")
    with db.engine.connect() as conn:
        result = conn.execute(sql).mappings().all()
        predicted_values = []
        rgb_values = []
        for row in result:
            predictions = json.loads(row.get("data"))
            rgb_value = json.loads(row.get("image"))
            rgb_values.append([rgb_value["avgRed"],rgb_value["avgGreen"], rgb_value["avgBlue"],])
            created_at = row.get("created_at")
            for prediction in predictions:
                label = prediction["predictedLabel"]
                score = prediction["score"]
                predicted_values.append([label, score, created_at])

        df = pd.DataFrame(predicted_values, columns=["label", "score", "created_at"])
        df_colors = pd.DataFrame(rgb_values, columns=["avgRed", "avgGreen", "avgBlue"])

        df["time"] = pd.to_datetime(df["created_at"]).dt.strftime('%Y-%m-%d %H:%M')
        agg_values = df.groupby(["time", "label"]).agg({"label": "count", "score": "mean"})
        agg_scores = df.groupby(["label"]).agg({"score": ["mean", np.std]})
        avgRed = round(df_colors["avgRed"].mean(), 1)
        avgGreen = round(df_colors["avgGreen"].mean(), 1)
        avgBlue = round(df_colors["avgBlue"].mean(), 1)
        stdRed = round(df_colors["avgRed"].std(), 1)
        stdGreen = round(df_colors["avgGreen"].std(), 1)
        stdBlue = round(df_colors["avgBlue"].std(), 1)
        if math.isnan(avgRed):
            avgRed = 0
            avgGreen = 0
            avgBlue = 0
            stdRed = 0
            stdGreen = 0
            stdBlue = 0

        print({
            "data": agg_values.to_json(),
            "image": {
                "avgRed": [avgRed, stdRed],
                "avgGreen": [avgGreen, stdGreen],
                "avgBlue": [avgBlue, stdBlue],
            },
            "agg_scores": {
                "mean": agg_scores["score"]["mean"].to_json(),
                "std": agg_scores["score"]["std"].to_json()
            }
        })
        return {
            "data": agg_values.to_json(),
            "image": {
                "avgRed": [avgRed, stdRed],
                "avgGreen": [avgGreen, stdGreen],
                "avgBlue": [avgBlue, stdBlue],
            },
            "agg_scores": {
                "mean": agg_scores["score"]["mean"].to_json(),
                "std": agg_scores["score"]["std"].to_json()
            }
        }



@app.get("/{projectId}/activeModel")
def get_active_model(projectId: int):
    project: Project = get_entity(Project, projectId)
    modelId = project.current_modelId

    zip_path = f'./projects/{projectId}/{modelId}/js_model.zip'

    with open(zip_path, 'rb') as file:
        data = file.read()

    return Response(content=data,
                    media_type="application/x-zip-compressed",
                    headers={"Content-Disposition": f"attachment; filename={os.path.basename(zip_path)}"})


@app.get("/{projectId}/download/{model_id}/{file_name}")
async def get_model(projectId: int, file_name: str):
    project: Project = get_entity(Project, projectId)
    modelId = project.current_modelId
    model_directory = f'./projects/{projectId}/{modelId}/js_model/'

    model_path = os.path.join(model_directory, file_name)
    print("test", model_path)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(model_path)


@app.post("/{projectId}/registerDevice")
def register_device(projectId: int):
    return {"id": insert_or_update_entity(Device, None, {"projectId": projectId})}


@app.get("/{projectId}/devices")
async def devices(projectId: int):
    devices: Device = get_entity(Device, projectId, "projectId", first=False)
    for idx, device in enumerate(devices):
        if connections.get(device.id) is not None:
            setattr(devices[idx], "online", True)
        else:
            setattr(devices[idx], "online", False)

    return devices


@app.post("/{projectId}/setDark")
async def setDark(projectId: int):
    global dark
    dark = not dark
    for key, websocket in connections.items():
        await websocket.send_json({"type": "setDark", "value": dark})
    return dark


@app.post("/{projectId}/setModel")
async def set_model(projectId: int, request: Request):
    request = await request.json()
    model_id = request.get("modelId")
    model: Model = get_entity(Model, model_id, first=True)
    print(model)
    if model.projectId != projectId:
        raise HTTPException(status_code=400, detail="Invalid project or model Id.")

    dir_path = f"./projects/{projectId}/{model_id}"
    delete_dir(dir_path)
    make_dir(dir_path)
    await download_model(model_id, dir_path)
    model_path = f"{dir_path}/best.onnx"
    js_model_path = f"{dir_path}/js_model"
    tmp_path = f"{dir_path}/tmp"
    make_dir(tmp_path)
    convert_model_to_js(model_path, js_model_path, tmp_path)
    delete_dir(tmp_path)
    zip_folder(js_model_path)
    project:Project = get_entity(Project, projectId, first=True)

    insert_or_update_entity(Project, projectId, {"current_modelId": model_id})
    insert_or_update_entity(Model, project.current_modelId, {"active": False})
    insert_or_update_entity(Model, model_id, {"active": True})

    for key, websocket in connections.items():
        await websocket.send_json({"type": "updateModel", "value": {"version": model_id}})

    return Response(status_code=200)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await websocket.accept()
    connections[client_id] = websocket
    insert_or_update_entity(Device, client_id, {"last_online": datetime.datetime.now(), "projectId": 300530342426941322})
    await websocket.send_json({"type": "modelVersion"})
    try:
        while True:
            data = await websocket.receive_json()
            match data["type"]:
                case "modelVersion":
                    device: Device = get_entity(Device, client_id, option="project")
                    project: Project = device.project
                    expected_version = project.current_modelId
                    actual_version = data["value"]
                    insert_or_update_entity(Device, client_id, {"modelId": actual_version})
                    if expected_version != actual_version:
                        await websocket.send_json({"type": "updateModel", "value": {"version": expected_version}})
                    await websocket.send_json({"type": "setDark", "value": dark})
                case "updateModelVersion":
                    new_version = data["value"]
                    print("UPDATE_MODEL_VERSION")
                    insert_or_update_entity(Device, client_id, {"modelId": new_version})
                case "image":
                    imgdata = decode_base64(data["base64"])
                    filename = 'some_image.png'  # I assume you have a way of picking unique filenames
                    with open(filename, 'wb') as f:
                        f.write(imgdata)
                    await websocket.send_text(f"Message text was: {data}")
                case "logs":
                    logs = data["value"]["metrics"]
                    modelId = data["value"]["modelId"]
                    deviceId = data["value"]["deviceId"]
                    image = data["value"]["image"]
                    insert_or_update_entity(ModelMetrics, None, {
                        "data": json.dumps(logs),
                        "modelId": modelId,
                        "deviceId": deviceId,
                        "image": json.dumps(image)
                    })

    except WebSocketDisconnect:
        insert_or_update_entity(Device, client_id, {"last_online": datetime.datetime.now()})
        print(f"""{client_id} disconnected.""")
        del connections[client_id]