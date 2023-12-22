import glob
import os
import os.path as osp
import logging
import hydra
import json
from omegaconf import OmegaConf
import configs.config_loader as cfg_loader
import cv2
from scipy.spatial import distance
from scipy.ndimage import zoom
import random
import numpy as np

def main():
    # 加载配置文件
    cfg = cfg_loader.get_config()
    print("Start Processing raw sequence")

    # 存储每个输入文件夹的图像列表
    input_image_lists = []
    for input_folder in cfg.raw_data_dirs:
        # print(glob.glob(input_folder + "/*.png"))
        input_image_lists.append(sorted(glob.glob(input_folder + "/*.png")))

    # process sequence folder one by one
    # 逐个处理输入图像文件夹

    for input_image_list, folder_name in zip(input_image_lists, cfg.raw_data_dirs):
        # print(input_image_list)
        # assert input_image_list

        # 只获取文件夹名称并创建输出文件夹
        # get only the folder name and create the output folder
        sequence_folder_name = osp.split(folder_name)[1]
        export_folder = osp.join(cfg.export_dir, sequence_folder_name)
        if not osp.exists(export_folder):
            os.makedirs(export_folder)

        # 逐个处理输入图像
        # go through the input image list
        for idx, image_path in enumerate(input_image_list):

            # read and enhance current image frame
            # 读取和增强当前图像帧
            image = cv2.imread(image_path, 3)
            enhanced_image = enhance_image(image)
            keypoints, _ = extract_image_keypoints(enhanced_image, cfg.params.keypoint_extractor)
            nb_keypoints = len(keypoints)

            # convert keypoints to numpy fp32
            # 将关键点转换为numpy fp32类型
            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)

            # keep sparse keypoints by removing closest points under threshold in a greedy way
            # 通过贪婪方式删除距离阈值内最近的点，以保留稀疏的关键点
            to_be_removed_ids = []
            for i in range(nb_keypoints):
                for j in range(nb_keypoints):
                    if i != j and j not in to_be_removed_ids:
                        dist = distance.euclidean(source_keypoints_coords[i], source_keypoints_coords[j])
                        if dist < cfg.params.keypoint_dist_threshold:
                            to_be_removed_ids.append(j)

            keypoints = list(keypoints)

            for el_idx in sorted(to_be_removed_ids, reverse=True):
                del keypoints[el_idx]

            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            nb_keypoints = len(source_keypoints_coords)
            image_height, image_width = enhanced_image.shape
            (center_x, center_y) = (image_width // 2, image_height // 2)

            # 随机选择预定义的变换之一
            # select random transformation between predefined transformation list
            transformation = random.choice(cfg.params.transformation_list)

            # TODO just for debug
            transformation = "rotation"

            if transformation == "rotation":
                rotation_angle = random.choice(cfg.params.predefined_angle_degrees)
                rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
                warped_image = cv2.warpAffine(enhanced_image, rotation_matrix, (image_width, image_height))

                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    rotated_point = rotation_matrix.dot(
                        np.array((int(source_keypoints_coords[b][0, 0]), int(source_keypoints_coords[b][0][1])) + (1,))
                    )

                    xp = int(rotated_point[0])
                    yp = int(rotated_point[1])
                    xa = int(source_keypoints_coords[b][0][0])
                    ya = int(source_keypoints_coords[b][0][1])
                    xn = int(source_keypoints_coords[b + 1][0][0])
                    yn = int(source_keypoints_coords[b + 1][0][1])
                    z = cfg.params.patch_size

                    # check if the the patch is inside the image canvas
                    if (
                        ((yp - z) > 0)
                        & ((xp - z) > 0)
                        & ((yp + z) < image_height)
                        & ((xp + z) < image_width)
                        & ((ya - z) > 0)
                        & ((xa - z) > 0)
                        & ((ya + z) < image_height)
                        & ((xa + z) < image_width)
                        & ((yn - z) > 0)
                        & ((xn - z) > 0)
                        & ((yn + z) < image_height)
                        & ((xn + z) < image_width)
                    ):
                        # do crop patch from the the warped image
                        crop_img_p = warped_image[yp - z : yp + z, xp - z : xp + z]
                        crop_img_a = enhanced_image[ya - z : ya + z, xa - z : xa + z]
                        crop_img_n = enhanced_image[yn - z : yn + z, xn - z : xn + z]

                        # construct output filenames for triplet patches
                        curr_output_folder = osp.join(export_folder, f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"

                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)

                        # save the triplet patches
                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)
                        triplet_counter += 1


if __name__ == "__main__":
    main()
