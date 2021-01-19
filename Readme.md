# TF2 Detection API Setup / Run Steps

## Step 1

Edit model_config.py

## Step 2

Create Dataset with pascal_voc.xml annotations

## Step 3

Run setup-tf2-obj-detection.py

## Step 4

Run pascal-voc-2-tfrecord.py --images-dir
## Step 5

Model Training

cd <tf2 model repo root>/

## Step 6

Model Export


Run model_exporter.py

## step 7

Inference

run  predict.py
