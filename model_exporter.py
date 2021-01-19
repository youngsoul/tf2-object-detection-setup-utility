import os
import pathlib
import argparse
import model_config as mc


"""
ASSUMES you are in the object_detection directory
python exporter_main_v2.py \
    --trained_checkpoint_dir training \
    --output_directory inference_graph \
    --pipeline_config_path training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config
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

    pathlib.Path(f"{object_detection_path}/inference_graph").mkdir(parents=True, exist_ok=True)

    os.system(f"python exporter_main_v2.py \
    --trained_checkpoint_dir {mc.training_dir_name} \
    --output_directory inference_graph \
    --pipeline_config_path training/{mc.base_pipeline_file}")



