import os
import logging
import cv2
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
from tqdm import tqdm
import imageio
import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.path import get_cwd
import numpy as np
import cv2 as cv
from math import sqrt


device = torch.device("cpu")


import torch
import statistics

from utils.matcher import feature_match, feature_extraction, evaluate_matches
from part2_model.arch_factory import model_factory
from part1_data.dataLoader import PatchDataset
import configs.config_loader as cfg_loader


def export_gif(frame_path_list, out_gif_filename, fps=24):
    frame_list = []
    for frame_path in tqdm(frame_path_list):
        frame_list.append(cv2.imread(frame_path))
        # os.remove(frame_path)
    imageio.mimsave(out_gif_filename, frame_list, fps=fps)

def main(cfg):
    args = cfg_loader.get_config()
    output_dir = "outputs"
    print(f"Working dir: {os.getcwd()}")
    print(f"Export dir: {output_dir}")

    print("Loading parameters from config file")
    validation_data_root = args.demo_sequence_data
    model_name = args.modelname
    model_weights_path = args.weights_path
    image_size = args.image_size
    patch_size = args.patch_size
    distance_matching_threshold = args.distance_matching_threshold
    matching_threshold = args.matching_threshold

    # load the model to be evaluated
    # 加载要评估的模型
    model = model_factory(model_name, model_weights_path)

    # generate patches from video frames
    # generate testing data
    # 生成来自视频帧的补丁
    # 生成测试数据
    test_dataset = PatchDataset(validation_data_root, patch_size)

    # list to store output git frames
    # 用于存储输出gif帧的列表
    frame_path_list = []

    # go through the patches, frame by frame
    # 逐帧遍历patches
    for id, data in enumerate(tqdm(test_dataset)):

        patch_src = data["patch_src"]
        patch_dst = data["patch_dst"]
        keypoint_src = data["keypoint_src"]
        keypoint_dst = data["keypoint_dst"]
        frame_src = data["image_src_name"]
        Next_frame = data["image_dst_name"]
        # extract feature vector for all patches
        # 提取所有补丁的特征向量
        list_desc_src = feature_extraction(patch_src, model, image_size)
        list_desc_dst = feature_extraction(patch_dst, model, image_size)

        # do matching
        # 进行匹配
        matches, distance_list = feature_match(
            list_desc_src, list_desc_dst, matching_threshold
        )
        h, w = frame_src.shape

        # -------------------------------------------------------------------------------------------------
        # 合并图像以显示匹配结果
        image_match = np.concatenate((frame_src, Next_frame), axis=1)
        image_match_rgb = cv.cvtColor(image_match, cv.COLOR_GRAY2BGR)

        for i in range(0, len(keypoint_src)):

            # if matches[i] != -1:
            xa = int(keypoint_src[i][0])
            ya = int(keypoint_src[i][1])
            xp = int(keypoint_dst[i][0])
            yp = int(keypoint_dst[i][1])
            x = int(keypoint_dst[matches[i]][0])
            y = int(keypoint_dst[matches[i]][1])
            dist = sqrt((yp - y) ** 2 + (xp - x) ** 2)

            cv.circle(
                image_match_rgb, (xa, ya), radius=2, color=(255, 0, 0), thickness=2
            )
            cv.circle(
                image_match_rgb, (xp + w, yp), radius=2, color=(255, 0, 0), thickness=2
            )
            if dist > distance_matching_threshold:
                cv.line(image_match_rgb, (xa, ya), (x + w, y), (0, 0, 255), thickness=1)
            else:
                cv.line(image_match_rgb, (xa, ya), (x + w, y), (0, 255, 0), thickness=1)

        file_name = os.path.join(output_dir, f"matches{id}_{id + 1}.png")
        frame_path_list.append(file_name)
        cv.imwrite(file_name, cv2.resize(image_match_rgb, (0, 0), fx=0.6, fy=0.6))

        # break

    print("Start exporting demo GIF")
    export_git_filename = os.path.join(output_dir, "matching_demo.gif")
    export_gif(
        frame_path_list=frame_path_list, out_gif_filename=export_git_filename, fps=20
    )


if __name__ == "__main__":
    main()
