from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入必要的库和模块
import sys
import os
import os.path as osp
import argparse  # 用于解析命令行参数
import time
import numpy as np  # 数组和数值计算
from tqdm import tqdm  # 进度条显示
import json
import torch  # PyTorch库
import torch.backends.cudnn as cudnn  # CUDA加速设置
import cv2  # OpenCV库，用于视频处理
import copy  # 用于对象深拷贝

# 从HRNet库导入实用工具和模型
# from lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from PoseDetector2d.lib.hrnet.lib.config import cfg, update_config
from PoseDetector2d.lib.hrnet.lib.utils.transforms import *
from PoseDetector2d.lib.hrnet.lib.utils.inference import get_final_preds
from PoseDetector2d.lib.hrnet.lib.models import pose_hrnet
from PoseDetector2d.lib.hrnet.lib.utils.utilitys import PreProcess
# HRNet配置和模型路径
cfg_dir = 'F:/Pycharm/GCNTransformer/PoseDetector2d/lib/hrnet/experiments/'  # 配置文件目录
model_dir = 'F:/Pycharm/GCNTransformer/PoseDetector2d/lib/checkpoint/'  # 模型文件目录

# 导入YOLOv3人类检测器和SORT跟踪库
from PoseDetector2d.lib.yolov3.human_detector import load_model as yolo_model
from PoseDetector2d.lib.yolov3.human_detector import yolo_human_det as yolo_det
from PoseDetector2d.lib.sort.sort import Sort  # 多目标跟踪算法

#
# def parse_args():
#     """解析命令行参数。"""
#     parser = argparse.ArgumentParser(description='Train keypoints network')
#     # 一般配置
#     parser.add_argument('--cfg', type=str, default=cfg_dir+'w48_384x288_adam_lr1e_3.yaml',
#                         help='实验配置文件名')
#     parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
#                         help="使用命令行修改配置选项")
#     parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
#                         help='模型目录')
#     parser.add_argument('--det-dim', type=int, default=416,
#                         help='检测图像的输入维度')
#     parser.add_argument('--thred-score', type=float, default=0.30,
#                         help='对象置信度的阈值')
#     parser.add_argument('-a', '--animation', action='store_true',
#                         help='输出动画')
#     parser.add_argument('-np', '--num-person', type=int, default=1,
#                         help='估计姿态的最大人数')
#     parser.add_argument("-v", "--video", type=str, default='camera',
#                         help="输入视频文件名")
#     parser.add_argument('--gpu', type=str, default='0', help='使用的GPU')
#     args = parser.parse_args()
#
#     return args
#
#
# def reset_config(args):
#     """更新配置并设置CUDA选项。"""
#     update_config(cfg, args)
#
#     # cudnn相关设置
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK  # 启用基准测试以提高性能
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  # 确保结果可重复
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED  # 启用cudnn
#
#
# def model_load(config):
#     """加载HRNet模型。"""
#     model = pose_hrnet.get_pose_net(config, is_train=False)  # 获取HRNet模型
#     if torch.cuda.is_available():
#         model = model.cuda()  # 将模型移动到GPU
#
#     state_dict = torch.load(config.OUTPUT_DIR)  # 加载模型权重
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k  # 去掉module前缀
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)  # 加载新字典中的权重
#     model.eval()  # 设置模型为评估模式
#
#     return model
#
#
# def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False):
#     """从视频生成关键点。"""
#     # 使用解析的参数更新配置
#     args = parse_args()
#     reset_config(args)
#
#     cap = cv2.VideoCapture(video)  # 打开视频文件
#
#     # 加载YOLO模型和姿态估计模型
#     human_model = yolo_model(inp_dim=det_dim)  # 加载YOLO人类检测模型
#     pose_model = model_load(cfg)  # 加载HRNet姿态估计模型
#     people_sort = Sort(min_hits=0)  # 初始化SORT跟踪
#
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧数
#
#     kpts_result = []  # 存储关键点结果
#     scores_result = []  # 存储分数结果
#     for ii in tqdm(range(video_length)):  # 逐帧处理视频
#         ret, frame = cap.read()  # 读取帧
#
#         if not ret:
#             continue  # 如果读取失败，跳过此帧
#
#         # 检测帧中的人
#         bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)
#
#         if bboxs is None or not bboxs.any():
#             print('No person detected!')
#             bboxs = bboxs_pre  # 使用前一帧的检测结果
#             scores = scores_pre
#         else:
#             bboxs_pre = copy.deepcopy(bboxs)  # 深拷贝检测结果
#             scores_pre = copy.deepcopy(scores)
#
#         # 使用SORT跟踪人
#         people_track = people_sort.update(bboxs)
#
#         # 跟踪视频中的前两个人，并去除ID
#         if people_track.shape[0] == 1:
#             people_track_ = people_track[-1, :-1].reshape(1, 4)
#         elif people_track.shape[0] >= 2:
#             people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
#             people_track_ = people_track_[::-1]  # 翻转顺序
#         else:
#             continue
#
#         track_bboxs = []
#         for bbox in people_track_:
#             bbox = [round(i, 2) for i in list(bbox)]  # 四舍五入边界框坐标
#             track_bboxs.append(bbox)
#
#         with torch.no_grad():  # 关闭梯度计算
#             # 预处理帧并获取预测
#             inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)
#             inputs = inputs[:, [2, 1, 0]]  # 改变通道顺序
#
#             if torch.cuda.is_available():
#                 inputs = inputs.cuda()  # 将输入移至GPU
#             output = pose_model(inputs)  # 获取模型输出
#
#             # 从输出计算坐标
#             preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
#
#         kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)  # 初始化关键点数组
#         scores = np.zeros((num_peroson, 17), dtype=np.float32)  # 初始化分数数组
#         for i, kpt in enumerate(preds):
#             kpts[i] = kpt  # 保存每个人的关键点
#
#         for i, score in enumerate(maxvals):
#             scores[i] = score.squeeze()  # 保存每个人的置信度
#
#         kpts_result.append(kpts)  # 添加当前帧的结果
#         scores_result.append(scores)
#
#     keypoints = np.array(kpts_result)  # 转换为numpy数组
#     scores = np.array(scores_result)  # 转换为numpy数组
#
#     keypoints = keypoints.transpose(1, 0, 2, 3)  # 转换维度：(T, M, N, 2) -> (M, T, N, 2)
#     scores = scores.transpose(1, 0, 2)  # 转换维度：(T, M, N) -> (M, T, N)
#
#     return keypoints, scores  # 返回关键点和分数



# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import sys
# import os
# import os.path as osp
# import argparse
# import time
# import numpy as np
# from tqdm import tqdm
# import json
# import torch
# import torch.backends.cudnn as cudnn
# import cv2
# import copy
#
# from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
# from lib.hrnet.lib.config import cfg, update_config
# from lib.hrnet.lib.utils.transforms import *
# from lib.hrnet.lib.utils.inference import get_final_preds
# from lib.hrnet.lib.models import pose_hrnet
#
# cfg_dir = 'demo/lib/hrnet/experiments/'
# model_dir = 'demo/lib/checkpoint/'
#
# # Loading human detector model
# from lib.yolov3.human_detector import load_model as yolo_model
# from lib.yolov3.human_detector import yolo_human_det as yolo_det
# from lib.sort.sort import Sort
#
#
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e_3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')

    return model


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    cap = cv2.VideoCapture(video)

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()

        if not ret:
            continue

        bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

        if bboxs is None or not bboxs.any():
            print('No person detected!')
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs)

            # compute coordinate
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)
    # print(keypoints.shape,keypoints)
    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores
