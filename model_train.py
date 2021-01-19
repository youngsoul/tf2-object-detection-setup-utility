import os
import pathlib
import argparse
import model_config as mc


"""
call the training script which is in 
models/research/object_detection/model_main_tf2.py
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf2-obj-det-root", required=False, default="../tf2_models_repo", help="Root directory to clone TF2 Models Repo")
    args = vars(ap.parse_args())

    models_repo_root = args['tf2_obj_det_root']
    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()

    research_path = f"{abs_models_repo_path}/models/research"
    object_detection_path = f"{research_path}/object_detection"

    os.chdir(object_detection_path)

    os.system(f"python model_main_tf2.py \
    --pipeline_config_path {mc.training_dir_name}/{mc.base_pipeline_file} \
    --model_dir={mc.training_dir_name} \
    --alsologtostderr"
    )



