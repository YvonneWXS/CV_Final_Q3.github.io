import os
import hydra
import json
from omegaconf import OmegaConf
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import configs.config_loader as cfg_loader
from part2_model.arch_factory import model_factory
from part1_data.dataLoader import PatchDataset
from utils.matcher import feature_match
from utils.matcher import evaluate_matches
import torch
import statistics

device = torch.device("cpu")

def main(cfg):
    args = cfg_loader.get_config()
    validation_data = args.validation_data
    root_folder = os.path.abspath(os.path.split(__file__)[0] + "/")
    validation_data_root = validation_data
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

    # back metrics
    # 用于存储评估指标的列表
    precision = []
    matching_score = []

    # go through the patches, frame by frame
    # for i, data in enumerate(test_dataset):
    # 逐帧遍历补丁
    for i, data in enumerate(tqdm.tqdm(test_dataset)):

        patch_src = data["patch_src"]
        patch_dst = data["patch_dst"]
        gt_keypoint_src = data["keypoint_src"]
        gt_keypoint_dst = data["keypoint_dst"]

        # extract feature vector for all patches
        # 提取所有补丁的特征向量
        list_desc_src = feature_extraction(patch_src, model, image_size)
        list_desc_dst = feature_extraction(patch_dst, model, image_size)

        # do matching
        # 进行匹配
        matches, distance_list = feature_match(
            list_desc_src, list_desc_dst, matching_threshold
        )

        # compute evaluation metrics
        # 计算评估指标
        nb_false_matching, nb_true_matches, nb_rejected_matches = evaluate_matches(
            gt_keypoint_src,
            gt_keypoint_dst,
            matches,
            distance_matching_threshold,
            distance_list,
            matching_threshold,
        )

        # # do matching
        # matches, _ = feature_match(list_desc_src, list_desc_dst, matching_threshold)
        #
        # # compute evaluation metrics
        # nb_false_matching, nb_true_matches, nb_rejected_matches = evaluate_matches(
        #     gt_keypoint_src, gt_keypoint_dst, matches, distance_matching_threshold
        # )
        precision.append(nb_true_matches / (nb_false_matching + nb_true_matches))
        matching_score.append(
            nb_true_matches
            / (nb_false_matching + nb_true_matches + nb_rejected_matches)
        )


if __name__ == "__main__":
    main()
