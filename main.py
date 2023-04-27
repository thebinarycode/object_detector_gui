import os
import utils

OBJ_DET_SETUP_PATH = "./models/research/object_detection/packages/tf2/setup.py"

TRAIN_IMAGE_PATH = "./workspace/training-demo/images/train"
VAL_IMAGE_PATH =  "./workspace/training-demo/images/validation"

LABEMAP_PATH = "./workspace/training-demo/annotations/label_map.pbtxt"

TRAIN_TFRECORD_PATH = "./workspace/training-demo/annotations/train.record"
VAL_TFRECORD_PATH = "./workspace/training-demo/annotations/validation.record"

MODEL_RES_PATH = "./workspace/training-demo/models/"

def set_path():
    print(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Annotate images
def annotate_images():
    set_path()
    os.system("labelImg")

# Create label_map.pbtxt file
def create_labelmap(labels):
    set_path()
    contents = []
    if "," in labels:
        labels = labels.split(",")
    else: labels = [labels]

    msg = ''
    for id, name in enumerate(labels, start=1):
        msg = msg + "item {\n"
        msg = msg + " id: " + str(id) + "\n"
        msg = msg + " name: '" + name + "'\n}\n\n"

    utils.create_file(LABEMAP_PATH, msg[:-1])
    return labels

# TFRecord creation
def generate_tfrecord():
    set_path()
    os.system(f"python generate_tfrecord.py -x {TRAIN_IMAGE_PATH} -l {LABEMAP_PATH} -o {TRAIN_TFRECORD_PATH}")
    os.system(f"python generate_tfrecord.py -x {VAL_IMAGE_PATH} -l {LABEMAP_PATH} -o {VAL_TFRECORD_PATH}")

# Model Training
def train_model(model):
    set_path()
    os.system(f"python ./workspace/training_demo/model_main_tf2.py --model_dir=./workspace/training-demo/models/{model} --pipeline_config_path=./workspace/training-demo/models/{model}/pipeline.config")

# Dependencies setup 
def setup():
    set_path()
    utils.create_folder("./addons")
    utils.create_folder(TRAIN_IMAGE_PATH)
    utils.create_folder(VAL_IMAGE_PATH)
    utils.create_folder("./workspace/training-demo/annotations")
    utils.create_folder("./workspace/training-demo/pre-trained-models")

    utils.download_zip(utils.LABELIMG_URL, "./addons/labelImg.zip")
    utils.download_zip(utils.PROTOBUF_URL, "./addons/protoc.zip")
    utils.download_zip(utils.OBJECT_DETECTION_URL, "./models.zip")

    utils.copy_files(OBJ_DET_SETUP_PATH, "./models/research/setup.py")
    # os.system("python -m pip install --use-feature=2020-resolver ./models/research/")
    os.system("python -m pip install ./models/research/")
    utils.copy_files("./models/research/object_detection/model_main_tf2.py", "./workspace/training-demo/model_main_tf2.py")

    os.system("protoc models/research/object_detection/protos/*.proto --python_out=./models/research/")

# Environment check - monitor for exceptions/errors in console logs
def env_check():
    set_path()
    os.system("python -c 'import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))'")
    os.system("python ./models/research/object_detection/builders/model_builder_tf2_test.py")

# Update pipeline.config file
def get_model_and_configure(models, labels):
    modelname = None
    label_list = create_labelmap(labels)
    filenames = utils.download_model(models)
    for filename in filenames:
        config_details = {}
        config_details["num_classes"] = len(label_list)
        config_details["fine_tune_checkpoint"] = "./workspace/training-demo/pre-trained-models/{filename}/checkpoint/ckpt-0".format(filename)
        config_details["label_map_path"] = LABEMAP_PATH
        config_details["train_input_path"] = TRAIN_TFRECORD_PATH
        config_details["eval_input_path"] = VAL_TFRECORD_PATH

        utils.update_config("", config_details)
        modelname = filename
    return modelname


def monitor_results():
    set_path()
    model_name = utils.list_directory(MODEL_RES_PATH)

    logdir = MODEL_RES_PATH + model_name[0]
    os.system(f"tensorboard --logdir={logdir}")
