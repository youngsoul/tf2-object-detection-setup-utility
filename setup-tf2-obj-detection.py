import os
import subprocess
import pathlib
import argparse
import platform
import logging
import model_config as mc
import tarfile
import re

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

base_tf2_pretrained_weights_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711"

this_file_dir = os.path.dirname(os.path.abspath(__file__))


def clone_models_repo(root_dir:str):
    if not pathlib.Path(f"{root_dir}/models").exists():
        os.chdir(root_dir)
        os.system("git clone https://github.com/tensorflow/models.git")

def install_tf_detection_api(research_dir:str):
    os.chdir(research_dir)
    logger.info(f"CWD: {research_dir}")
    logger.info("Running protoc")
    os.system("protoc object_detection/protos/*.proto --python_out=.")
    os.system("cp object_detection/packages/tf2/setup.py .")
    logger.info("pip install Tensorflow Dependendencies. see setup.py")
    os.system("pip install .")

def verify_tf_obj_detection_install(reseach_dir:str):
    os.chdir(reseach_dir)
    os.system("python object_detection/builders/model_builder_tf2_test.py")

def fix_tf_utils_bug():
    print(this_file_dir)
    if pathlib.Path(f"{this_file_dir}/tf_util_fixed.txt").exists():
        logger.info("tf_util.py already updated....")
        return

    logger.warning(f"NOTE:  Updating tf_utils.py")
    # ./python3.8/site-packages/tensorflow/python/keras/utils/tf_utils.py
    venv_python_path = subprocess.check_output("which python", shell=True)
    x = str(venv_python_path).split("/")[:-2]
    p = "/".join(x[1:])
    py_version_tuple = platform.python_version_tuple()
    path_to_tf_utls = "/" + p + f"/lib/python{py_version_tuple[0]}.{py_version_tuple[1]}/site-packages/tensorflow/python/keras/utils/tf_utils.py"

    print(path_to_tf_utls)

    with open(path_to_tf_utls) as f:
        tf_utils = f.read()

    with open(path_to_tf_utls, 'w') as f:
        throw_statement = "raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))"
        tf_utils = tf_utils.replace(throw_statement, "if not isinstance(x, str):" + throw_statement)
        f.write(tf_utils)

    os.chdir(this_file_dir)
    os.system(f"touch {this_file_dir}/tf_util_fixed.txt")

def pip_install_labelimg():
    os.system("pip install labelImg")


def download_pretrained_model_weights(research_dir:str):
    deploy_dir = pathlib.Path(f"{research_dir}/object_detection/{mc.training_dir_name}")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(str(deploy_dir))
    logger.info(f"Downloading pretrained weights to: {str(deploy_dir)}")

    download_tar_url = f"{base_tf2_pretrained_weights_url}/{mc.pretrained_checkpoint}"
    logger.info(f"execute: wget {download_tar_url}")
    os.system(f"wget {download_tar_url}")

    tar = tarfile.open(f"{str(deploy_dir)}/{mc.pretrained_checkpoint}")
    tar.extractall()
    tar.close()

    pathlib.Path(f"{str(deploy_dir)}/{mc.pretrained_checkpoint}").unlink()



def copy_base_training_config(research_dir:str):
    # models/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config
    deploy_dir = pathlib.Path(f"{research_dir}/object_detection/{mc.training_dir_name}")

    os.chdir(deploy_dir)
    source_base_config_path = f"{research_dir}/object_detection/configs/tf2/{mc.base_pipeline_file}"
    logger.info(f"cp {source_base_config_path} {deploy_dir}")
    os.system(f"cp {source_base_config_path} {deploy_dir}")

    pipeline_fname = deploy_dir / mc.base_pipeline_file
    return str(pipeline_fname)

def create_label_map_file(models_repo_path):
    """
    item {
    id: 1
    name: 'Raspberry_Pi_3'
}
item {
    id: 2
    name: 'Arduino_Nano'
}
item {
    id: 3
    name: 'ESP8266'
}
item {
    id: 4
    name: 'Heltec_ESP32_Lora'
}
    :return:
    :rtype:
    """

    with open(f"{models_repo_path}/models/research/object_detection/{mc.training_dir_name}/{mc.label_map_fname}", "w" ) as f2:
        for i, line in enumerate(mc.classes):
            f2.write("item {\n")
            f2.write(f"\tid: {i+1}\n")
            f2.write(f"\tname: '{line.strip()}'\n")
            f2.write("}\n")

def update_model_config_file(research_dir:str, pipeline_fname:str):


    logger.info('writing custom configuration file')
    fine_tune_checkpoint = f'{research_dir}/object_detection/training/' + mc.model_name + '/checkpoint/ckpt-0'

    tfrecord_output_path = f"{research_dir}/object_detection"
    train_record_fname = f"{tfrecord_output_path}/train.record"
    test_record_fname = f"{tfrecord_output_path}/test.record"
    label_map_pbtxt_fname = f"{research_dir}/object_detection/{mc.training_dir_name}/{mc.label_map_fname}"

    with open(pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(mc.batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(mc.num_steps), s)

        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                   'num_classes: {}'.format(len(mc.classes)), s)

        # fine-tune checkpoint type
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

        f.write(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf2-obj-det-root", required=False, default="../tf2_models_repo", help="Root directory to clone TF2 Models Repo")
    args = vars(ap.parse_args())

    models_repo_root = args['tf2_obj_det_root']
    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()
    abs_models_repo_path.mkdir(parents=True, exist_ok=True)
    research_path = f"{abs_models_repo_path}/models/research"
    object_detection_path = f"{research_path}/object_detection"

    print(abs_models_repo_path)
    print(research_path)

    # 1 Clone Tensorflow Repos Dir
    logger.info(f"Cloning Tensorflow Models Repo")
    clone_models_repo(root_dir=models_repo_root)
    logger.info(f"Done")

    # 2 Install the Object Detection API
    logger.info("Installing Tensorflow Detection API")
    install_tf_detection_api(research_dir=research_path)

    # 3 Verify TF ObjDet Install
    logger.info("Verify Tensorflow Object Detection Install...")
    verify_tf_obj_detection_install(reseach_dir=research_path)

    # 4 Fix tf_util bug
    logger.info("Fix tf_utils bug.  Watch for this step to no longer be needed")
    fix_tf_utils_bug()

    # 5 install LabelImg
    logger.info("Install LabelImg")
    pip_install_labelimg()

    # 6 download pretrained weights for selected model
    logger.info(f"Download pretrained model weights for model: {mc.pretrained_checkpoint}")
    download_pretrained_model_weights(research_dir=research_path)

    # 7 copy base config for model
    logger.info(f"Copy base model configuration: {mc.base_pipeline_file}")
    pipeline_filename = copy_base_training_config(research_dir=research_path)
    print(pipeline_filename)

    # 8 create label_map.pbtxt file, if it not not already there
    create_label_map_file(abs_models_repo_path)

    # 9 update model configuration file
    update_model_config_file(research_dir=research_path, pipeline_fname=pipeline_filename)