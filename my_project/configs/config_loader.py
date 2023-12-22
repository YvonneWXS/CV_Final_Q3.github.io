import sys
#sys.path.append('../lightNDF')
import numpy as np
import os
import configargparse

def config_parser():
    # 创建配置解析器
    parser = configargparse.ArgumentParser()
    """
    Parse input arguments
    """
    parser = configargparse.ArgumentParser(description='myproject')

    # parser.add_argument('--batch_size', type=int, default=4,
    #                     help='batch')
    # 是否从上次中断的地方继续训练
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume')

    ###config Training
    parser.add_argument('--logpath', type=str, default="runs",  #log
                        help='path of logs')
    parser.add_argument('--traindatapath', type=str, default="../data/base_train2",  #train_data
                        help='path of traindata')
    parser.add_argument('--modelname', type=str, default="HardNet128",   #model_name
                        help='model name')
    parser.add_argument('--nb_epoch', type=int, default=300,
                        help='epoch')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='learning')
    parser.add_argument('--image_size', type=int, default=128,
                        help='image_size')
    parser.add_argument('--margin_value', type=int, default=1,
                        help='margin_value')
    parser.add_argument('--loss_weight', type=float, default=0.5,
                        help='loss_weight')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay')
    parser.add_argument('--chdir', type=bool, default=True,
                        help='chdir')
    parser.add_argument('--loss_layer', type=str, default="HardNetLoss",
                        help='loss_layer')
    parser.add_argument('--output_dir', type=str, default="./trained_models",
                        help='loss_layer')

    ######matching_demo_para
    parser.add_argument('--paths', type=str, default="runs", #paths
                        help='paths')
    parser.add_argument('--demo_sequence_data', type=str, default="./data/test_dataset", #paths
                        help='demo_sequence_data')
    parser.add_argument('--weights_path', type=str, default="./trained_models/HardNet128.pth",
                        help='weights_path')

    # parser.add_argument('--image_size', type=int, default=128,
    #                     help='image_size')
    parser.add_argument('--matching_threshold', type=float, default=0.5,
                        help='matching_threshold')

    parser.add_argument('--patch_size', type=int, default= 64 ,
                        help='patch_size')

    parser.add_argument('--distance_matching_threshold', type=int, default=5,
                        help='distance_matching_threshold')


    ###triplet_generation_para

    # 呃, 这是什么文件夹啊
    parser.add_argument('--raw_data_dirs', type=list, default=["/media/achraf/data/workspace/crns/optimendoscopy/seq1",
                        "/media/achraf/data/workspace/crns/optimendoscopy/seq2",
                    "/media/achraf/data/workspace/crns/optimendoscopy/seq3",
                    "/media/achraf/data/workspace/crns/optimendoscopy/seq4"],
                        help='raw_data_dirs')
    parser.add_argument('--export_dir', type=str, default="/media/achraf/data/workspace/crns/optimendoscopy/triplets",
                        help='export_dir')
    parser.add_argument('--keypoint_extractor', type=str, default="SIFT",
                        help='keypoint_extractor')
    parser.add_argument('--keypoint_dist_threshold', type=int, default=['zoom', 'rotation', 'translation'],
                        help='keypoint_dist_threshold')
    parser.add_argument('--transformation_list', type=list, default=3,
                        help='transformation_list')
    parser.add_argument('--predefined_angle_degrees', type=list, default=[5, 10, 15],
                        help='predefined_angle_degrees')
    parser.add_argument('--predefined_zoom_factors', type=list, default=[0.9, 0.95, 1.05, 1.1, 1.15],
                        help='predefined_zoom_factors')


    parser.add_argument('--predefined_translation_pixels', type=list, default=[8],
                        help='predefined_translation_pixels')

    ###evaluation
    parser.add_argument('--validation_data', type=str, default="./data/test_dataset",
                        help='validation_data')
    parser.add_argument('--log', type=str, default="runs",
                        help='log')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    return cfg

