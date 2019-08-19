import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob
from chessconfig import ChessConfig
from chessdataset import ChessDataset
from sklearn.model_selection import KFold
from skimage.io import imread

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


root_dir = '/mnt/0A6A9B246A9B0B97/ITC/Deep Learning project'
train_dicom_dir = os.path.join(root_dir, 'Dataset/train')
test_dicom_dir = os.path.join(root_dir, 'Dataset/test')
COCO_WEIGHTS_PATH = os.path.join(root_dir, "mask_rcnn_coco.h5")
TEST_PATH = root_dir + '/chess_export_test.csv'
TRAIN_PATH = root_dir + '/chess_export_train.csv'
LEARNING_RATE = 0.006


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*')
    return list(set(dicom_fps))


def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['filename'])
        image_annotations[fp].append(row)
    return image_fps, image_annotations


def main():
    config = ChessConfig()
    config.display()

    # training dataset
    anns_train = pd.read_csv(TRAIN_PATH)
    image_fps_train, image_annotations_train = parse_dataset(train_dicom_dir, anns=anns_train)

    ORIG_SIZE = config.IMAGE_SHAPE[0]
    dataset_train = ChessDataset(image_fps_train, image_annotations_train, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()

    # testing dataset
    anns_test = pd.read_csv(TEST_PATH)
    image_fps_test, image_annotations_test = parse_dataset(test_dicom_dir, anns=anns_test)

    dataset_test = ChessDataset(image_fps_test, image_annotations_test, ORIG_SIZE, ORIG_SIZE)
    dataset_test.prepare()

    model = modellib.MaskRCNN(mode='training', config=config, model_dir=root_dir)

    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_test,
                learning_rate=LEARNING_RATE,
                epochs=2,
                layers='heads')


if __name__ == '__main__':
    main()
