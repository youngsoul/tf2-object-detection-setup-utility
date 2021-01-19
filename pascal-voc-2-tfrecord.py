# based on https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

"""
Usage:
python pascal-voc-2-tfrecord.py --images-dir ../images --tf2-obj-det-root ../tf2_models_repo


"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import pathlib
import model_config as mc


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def process_all_images(images_dir_path, models_repo_root):
    tfrecord_output_path = f"{models_repo_root}/models/research/object_detection"

    for folder in ['train', 'test']:
        image_path = os.path.join(images_dir_path,  folder)
        xml_df = xml_to_csv(image_path)
        path_to_csv = images_dir_path+'/'+folder+'_labels.csv'
        xml_df.to_csv(path_to_csv, index=None)
        tfrecord_cmd = f"python generate_tfrecord.py --csv_input={path_to_csv} --output_path={tfrecord_output_path}/{folder}.record --image_dir={image_path}"
        os.system(tfrecord_cmd)

    print('Successfully converted xml to csv.')

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



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Root directory where subfolders called train and test contain images and pascal_voc.xml files")
    ap.add_argument("--tf2-obj-det-root", required=False, default="../tf2_models_repo", help="Root directory to clone TF2 Models Repo")

    args = vars(ap.parse_args())

    images_dir = args['images_dir']
    models_repo_root = args['tf2_obj_det_root']

    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()

    #make sure the training directory exists incase this is run before some of the model setup
    pathlib.Path(f"{abs_models_repo_path}/models/research/object_detection/{mc.training_dir_name}").mkdir(parents=True, exist_ok=True)

    create_label_map_file(abs_models_repo_path)

    process_all_images(images_dir, models_repo_root)



