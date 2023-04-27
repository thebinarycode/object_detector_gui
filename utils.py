from pathlib import Path
import shutil
import os
import sys
import requests
from bs4 import BeautifulSoup
import tarfile
from zipfile import ZipFile

try:
    from models.research.object_detection.utils import config_util
except ImportError:
    print("Object detection import failed")

LABELIMG_URL = "https://github.com/heartexlabs/labelImg/archive/refs/tags/v1.8.1.zip"
PROTOBUF_URL = "https://github.com/protocolbuffers/protobuf/releases/download/v22.0/protoc-22.0-win64.zip"

OBJECT_DETECTION_URL = "https://github.com/tensorflow/models/archive/refs/heads/master.zip"

MODEL_ZOO_URL = "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md"
DOWNLOAD_BASE_URL = "http://download.tensorflow.org/models"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def data_check(dir_path):
    return True if os.path.exists(dir_path) else False

def create_folder(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("folder(s) created successfully")

def create_file(file_path, contents):
    f = open(file_path, "a")
    f.write(contents)
    f.close()
    print("file created successfully")

def list_directory(file_path):
    itr = iter(os.walk(file_path))
    root, dirs, files = next(itr)
    return dirs

def copy_files(src, dest):
    if os.path.isdir(src): shutil.copytree(src, dest, dirs_exist_ok=True)
    else: shutil.copy(src, dest)
    print("file(s) copied successfully")

def clean_files(dir):
    shutil.rmtree(dir)
    print("files cleaned successfully")

def download_model(models_list):
    print(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    response_content = requests.get(MODEL_ZOO_URL)
    parsed_html = BeautifulSoup(response_content.content, features="lxml")

    download_urls = {}
    for tags in parsed_html.find_all('a', href=True):
        if DOWNLOAD_BASE_URL in str(tags): download_urls[tags.get_text()] = str(tags.get('href'))

    print(f"total models available: {len(download_urls)}")

    filenames = []
    for model in models_list:
        print(f"process started for model: {model}")
        # download_url = next(url for url in download_urls if models_list in url)
        download_url = download_urls.get(model)
        filename = download_url.split("/")[-1]

        r = requests.get(download_url, allow_redirects=True)
        open("./workspace/training-demo/pre-trained-models/" + filename, 'wb').write(r.content)
        print(f"model file downloaded to local: {filename}")
        
        model_file = tarfile.open("./workspace/training-demo/pre-trained-models/" + filename)
        model_file.extractall("./workspace/training-demo/pre-trained-models/")
        model_file.close()
        create_folder("./workspace/training-demo/models/" + filename)
        copy_files("./workspace/training-demo/pre-trained-models/" + filename + "/pipeline.config" "./workspace/training-demo/models/" + filename + "/pipeline.config")
        
        print("model file extracted from compressed format successfully")
        filenames.append(filename)
    return filenames

def get_models_list():
    response_content = requests.get(MODEL_ZOO_URL)
    parsed_html = BeautifulSoup(response_content.content, features="lxml")

    download_urls = {}
    for tags in parsed_html.find_all('a', href=True):
        if DOWNLOAD_BASE_URL in str(tags): download_urls[tags.get_text()] = str(tags.get('href'))
    
    return download_urls

def download_zip(url, filename, remove_root=True):
    dirname = filename.replace(".zip", "")
    if not data_check(dirname):
        resp = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(resp.content)
        print(f"model file downloaded to local: {filename}")
        
        with ZipFile(filename, 'r') as zipdata:
            zipdata.extractall(path=dirname)

        if remove_root:
            subdir = list_directory(dirname)
            copy_files(dirname + "/" + subdir[0], dirname)
    else: print("data already present")

def update_config(file_path, config_dict):
    configs = config_util.get_configs_from_pipeline_file(file_path)
    for key, val in config_dict.items():
        if key == "num_classes": configs['model'].center_net.num_classes = val
        elif key == "fine_tune_checkpoint": configs['train_config'].fine_tune_checkpoint = val
        elif key == "label_map_path":
            configs['train_input_reader'].label_map_path = val
            configs['eval_input_reader'].label_map_path = val
        elif key == "train_input_path": configs['train_input_reader'].input_path = val
        elif key == "eval_input_path": configs['eval_input_reader'].input_path = val

    config_util.save_pipeline_config(configs, file_path)
    print("config file updated")


download_zip(LABELIMG_URL, "./addons/labelImg.zip")