import mlflow

mlflow_client = mlflow.tracking.MlflowClient("http://134.209.232.89:5000/")
mlflow.set_tracking_uri("http://134.209.232.89:5000/")


async def get_experiments():
    experiments = mlflow.search_experiments()
    return [experiment.__dict__ for experiment in experiments]


async def get_runs(experiment_id: str):
    runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list")
    return [run.__dict__ for run in runs]


async def get_run(run_id: str):
    return mlflow_client.get_run(run_id=run_id)


async def get_metric_history(run_id: str, metric_key:str):
    return mlflow_client.get_metric_history(run_id=run_id, key=metric_key)


async def get_artifacts(run_id: str):
    return mlflow_client.list_artifacts(run_id)


async def download_model(run_id: str, dir_path: str):
    mlflow_client.download_artifacts(run_id, 'best.onnx', dir_path)