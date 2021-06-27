import os
import sys
import yaml
import tfrecord
import tensorflow.compat.v1 as tf
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

path = "/media/SENSETIME\qiaozhijian/Extreme SSD/datasets/waymo/data_v1_0/tar_files_v1_0/training_0000/segment-1208303279778032257_1360_000_1380_000_with_camera_labels.tfrecord"
FILE_NAME = path
record_index = 0
dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')

frame_num = 0
pcs = dict()
gt_info = dict()
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    context = frame.context.name
    ts = frame.timestamp_micros
    # name = str(context) + '/' + str(ts)

    # extract the points
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=0)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
