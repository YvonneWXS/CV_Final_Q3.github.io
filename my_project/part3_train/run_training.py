import os
import hydra
import warnings
import configs.config_loader as cfg_loader
import part1_data.dataLoader as dataLoader
from part2_model.train_model.train_model import Trainer
import statistics
import torch
from torch.utils.tensorboard import SummaryWriter

from part2_model.models.hardnet_model import HardNet128


# 我是intel的核显,根本没有cuda, 就直接cpu吧orz
device = torch.device("cpu")
warnings.filterwarnings("ignore")


# @hydra.main(version_base=None, config_path="../configs", config_name="config_loader")
def main():
    args = cfg_loader.get_config()
    # 这应该不打需要, 都不用yaml文件了
    # 直接设置文件路径吧

    traindatapath = args.traindatapath
    batch_size = args.batch_size
    image_size = args.image_size

    # 创建数据集和数据加载器
    # creates dataset and datalaoder
    dataset = dataLoader.TripletDataset(traindatapath, image_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = HardNet128()


    print("Start Epochs ...")
    trainer = Trainer(train_loader, model,args)
    trainer.train_model()


if __name__ == "__main__":
    main()


