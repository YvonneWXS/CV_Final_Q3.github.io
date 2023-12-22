import numpy as np

import torch
import os
import cv2 as cv
from torch.utils.data import Dataset  #???
from scipy.spatial import distance
from numpy import loadtxt
import glob

import torch.utils.data as data
import os.path as osp

import sys
from PIL import Image
from torchvision import transforms

sys.path.append("../")
supported_image_ext = [".png", ".bmp", ".jpg", ".jpeg"]
labels = []

class TripletDataset(data.Dataset):
    """
       初始化TripletDataset对象。

       参数：
       - root: 数据集根目录的路径
       - image_size: 图像的大小，用于裁剪图像
       - shuffle: 是否在加载数据时对数据进行随机打乱
       - use_cache: 是否使用缓存

       属性：
       - root: 数据集根目录的路径
       - list: 数据集文件列表
       - nb_samples: 数据集样本数量
       - phase: 数据集阶段
       - transform: 是否进行图像变换
       - image_size: 图像的大小，用于裁剪图像
       """
    def __init__(self, root, image_size, shuffle=False, use_cache=True):
        self.root = root
        self.list = self.get_file_list()
        self.nb_samples = len(self.list)
        self.phase = 0
        self.transform = True
        self.image_size = image_size

    def get_file_list(self):
        """
            获取数据集文件列表。

            返回：
            - local_list: 数据集文件列表
        """
        local_list = []
        folder_list = glob.glob(self.root + "/*")

        for d in folder_list:
            bOk = True
            image_list = glob.glob(d + "/*")

            # TODO what is this if statement ??
            if not len(image_list) == 3:
                continue

            for i in image_list:
                bOk = bOk and osp.splitext(i)[1] in supported_image_ext

            # check same extension
            ext_0 = osp.splitext(image_list[0])[1]
            ext_1 = osp.splitext(image_list[1])[1]
            ext_2 = osp.splitext(image_list[2])[1]

            if not (ext_0 == ext_1 == ext_2):
                pass

            # check for a, n, and p
            folder = osp.split(image_list[0])[0]
            anchor = folder + "/a" + ext_0
            positive = folder + "/p" + ext_0
            negative = folder + "/n" + ext_0

            # Check whether the specified path exists or not
            bOk = bOk and osp.exists(anchor)
            bOk = bOk and osp.exists(positive)
            bOk = bOk and osp.exists(negative)

            if bOk:
                local_list.append({"a": anchor, "p": positive, "n": negative})

        return local_list

    def __getitem__(self, item):
        """
            获取数据集中指定索引的样本。
            参数：
            - item: 数据集中的索引
            返回：
            - item: 数据集中的索引
            - input_tensor: 包含锚点、正例和负例图像的张量
        """
        triplet_dict = self.list[item]
        # anchor patch
        img = Image.open(triplet_dict["a"])
        img_cropped = img.crop((0, 0, self.image_size, self.image_size))
        anchor_image = transforms.ToTensor()(img_cropped)

        # positive patch
        img = Image.open(triplet_dict["p"])
        img_cropped = img.crop((0, 0, self.image_size, self.image_size))
        positive_image = transforms.ToTensor()(img_cropped)

        # negative patch
        img = Image.open(triplet_dict["n"])
        img_cropped = img.crop((0, 0, self.image_size, self.image_size))
        negative_image = transforms.ToTensor()(img_cropped)

        input_tensor = torch.cat([anchor_image, positive_image, negative_image])
        return item, input_tensor

    def __len__(self):
        return self.nb_samples



class PatchDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, root_path, patch_size):
        """
            初始化PatchDataset对象。

            参数：
            - root_path: 数据集根目录的路径
            - patch_size: 图像块的大小
        """
        self.image_name = []
        self.keypoints_GT = []
        self.root_path = root_path
        self.patch_size = patch_size

        self.all_frames_per_sequence = []
        self.all_keypoints = []
        self.data = []
        self.load_data_path()

    def __len__(self):
        return len(self.data)

    def load_data_path(self):
        """
           加载数据集的路径信息。
       """
        # get list of  sequences
        sequence_list = glob.glob(self.root_path + "/*")

        # got through sequences
        for sequence in sequence_list:
            #  first get all gt matches for the current sequence
            curr_matches = sorted(glob.glob(sequence+"/matches/*"))
            for match in curr_matches:
                # get src and dst frame filenames
                _, src_filename, dst_filename = osp.splitext(osp.split(match)[1])[0].split("_")
                src_abs_path = osp.join(sequence, "frames", src_filename)
                dst_abs_path = osp.join(sequence, "frames", dst_filename)

                # if all related data exist
                if osp.exists(src_abs_path) and osp.exists(dst_abs_path):
                    self.data.append({"match": match, "src_frame": src_abs_path, "dst_frame": dst_abs_path})

    def enhance(self, img):
        """
            图像增强函数。

            参数：
            - img: 输入图像

            返回：
            - image_enhanced: 增强后的图像
        """
        crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 40]
        gray2 = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=5)
        image_enhanced = clahe.apply(gray2)
        # image_enhanced = cv.equalizeHist(gray2)
        return image_enhanced

    def __getitem__(self, idx):
        """
            获取数据集中指定索引的样本。

            参数：
            - idx: 数据集中的索引

            返回：
            - 字典包含样本信息：源图像、目标图像、源图像块、目标图像块、目标图像关键点、源图像关键点
        """
        frame_src = cv.imread(self.data[idx]["src_frame"], 3)
        Next_frame = cv.imread(self.data[idx]["dst_frame"], 3)

        frame_src = self.enhance(frame_src)
        Next_frame = self.enhance(Next_frame)

        # keypoints filenames
        matches = self.data[idx]["match"]

        list_matches = loadtxt(matches, dtype='int')

        list_keypoints_src = []
        list_keypoints_next_frame = []

        for i in range(0, len(list_matches)):
            list_keypoints_src.append((list_matches[i][0], list_matches[i][1]))
            list_keypoints_next_frame.append((list_matches[i][2], list_matches[i][3]))
        h, w = frame_src.shape

        # ---------------------------------------------------Generate_data-----------------------------------------------------------
        i = 0
        gt_key_src = []
        gt_key_next_frame = []
        patches_src = []
        patches_next_frame = []
        i = 0
        # ---------------------------------------------------Generate_data-----------------------------------------------------------

        for b in range(0, len(list_keypoints_src)):

            xa = int(list_keypoints_src[b][0])
            ya = int(list_keypoints_src[b][1])
            xp = int(list_keypoints_next_frame[b][0])
            yp = int(list_keypoints_next_frame[b][1])

            if (
                    ((ya - self.patch_size) > 0)
                    & ((xa - self.patch_size) > 0)
                    & ((ya + self.patch_size) < h)
                    & ((xa + self.patch_size) < w)
                    & ((yp - self.patch_size) > 0)
                    & ((xp - self.patch_size) > 0)
                    & ((yp + self.patch_size) < h)
                    & ((xp + self.patch_size) < w)
            ):
                crop_patches_src = frame_src[
                    ya - self.patch_size : ya + self.patch_size, xa - self.patch_size : xa + self.patch_size
                ]
                crop_patches_next_frame = Next_frame[
                    yp - self.patch_size : yp + self.patch_size, xp - self.patch_size : xp + self.patch_size
                ]

                patches_next_frame.append(crop_patches_next_frame)
                patches_src.append(crop_patches_src)
                gt_key_src.append(list_keypoints_src[i])
                gt_key_next_frame.append(list_keypoints_next_frame[i])
                i += 1
        return {
            "image_src_name": frame_src,
            "image_dst_name": Next_frame,
            "patch_src": patches_src,
            "patch_dst": patches_next_frame,
            "keypoint_dst": gt_key_next_frame,
            "keypoint_src": gt_key_src
        }

if __name__ == '__main__':
    dataset = TripletDataset('./data/base_train2',128)


