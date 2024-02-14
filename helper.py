import base64
import os
from pathlib import Path
import re
import shutil
from onnx.backend.base import BackendRep
import onnx
from onnx_tf.backend import prepare


def decode_base64(data, altchars='+/'):
    data = re.sub(r'[^a-zA-Z0-9%s]+' % altchars, '', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += '='* (4 - missing_padding)
    return base64.b64decode(data, altchars)


def make_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def delete_dir(path: str):
    dirpath = Path(path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


def convert_model_to_js(model_path, export_path, tmp_path):
    onnx_model = onnx.load(model_path)
    tf_rep: BackendRep = prepare(onnx_model)

    tf_rep.export_graph(tmp_path)
    os.system(f'tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve "{tmp_path}" "{export_path}"')


def zip_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path does not point to a directory.")

    base_folder = os.path.basename(folder_path)
    zip_path = os.path.join(os.path.dirname(folder_path), f"{base_folder}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
