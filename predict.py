"""
Script to run inference while following along with the blog:

https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model

and YouTube

https://www.youtube.com/watch?v=cvyDYdI2nEI

"""
import numpy as np
import glob
import cv2
from six import BytesIO
import os
from PIL import Image

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import pathlib

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf2-obj-det-root", required=False, default="../tf2_models_repo", help="Root directory to clone TF2 Models Repo")
    args = vars(ap.parse_args())

    models_repo_root = args['tf2_obj_det_root']
    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()

    research_path = f"{abs_models_repo_path}/models/research"
    object_detection_path = f"{research_path}/object_detection"

    # base_dir = "."  # assumes you are running this from object_detection directory
    labelmap_path = f"{object_detection_path}/training/label_map.pbtxt"

    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    # ./training/efficientdet_d0_coco17_tpu-32/saved_model/saved_model.pb

    tf.keras.backend.clear_session()
    model = tf.saved_model.load(
        f'{object_detection_path}/inference_graph/saved_model')

    for image_path in glob.glob(f'/Users/patrickryan/Development/tf2_object_detection/images/test/*.jpg'):
      image_np = load_image_into_numpy_array(image_path)
      output_dict = run_inference_for_single_image(model, image_np)
      print(output_dict)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          min_score_thresh=0.5,
          max_boxes_to_draw=3,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)

      cv2.imshow("Inference", image_np)
      cv2.waitKey(0)
