import argparse
import copy
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from PoseDetector2d.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from PoseDetector2d.lib.preprocess import h36m_coco_format
from PoseDetector2d.lib.utils import normalize_screen_coordinates, denormalize_screen_coordinates
from PoseDetector3d.model.MiEFormer import MiEFormer


sys.path.append(os.getcwd())
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def video2pose(videopath):
    keypoints2d = get_pose2D(video_path=videopath)
    keypoints3d = get_pose3D(keypoints2d=keypoints2d, video_path=videopath)
    return keypoints3d


def get_pose2D(video_path):
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)
    return keypoints


def get_pose3D(keypoints2d, video_path):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 12, 3, 64, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value = True, 0.00001
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_adaptive_fusion = True
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 3, 1
    args.use_tcn, args.graph_only, args.ablation = False, False, 0
    args.n_frames = 96
    args = vars(args)

    model = nn.DataParallel(MiEFormer(**args)).cuda()
    model_path = "PoseDetector3d/checkpoint/MiE.pth.tr"
    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    clips, downsample = turn_into_clips(keypoints2d)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_size = []
    for i in range(video_length):
        ret, img = cap.read()
        if ret:
            img_size = img.shape
            break

    all_3d_poses = []
    all_3d_poses_denormalized = []
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    for idx, clip in tqdm(enumerate(clips)):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[..., 0] *= -1
        input_2D_aug[..., joints_left + joints_right, :] = input_2D_aug[..., joints_right + joints_left, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip = model(input_2D)
        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        # print(output_3D.shape,output_3D)
        if idx == len(clips) - 1 and downsample is not None:
            output_3D = output_3D[:, downsample]

        all_3d_poses.append(output_3D[0].cpu().detach().numpy())
        all_3d_poses_denormalized.append(
            denormalize_screen_coordinates(output_3D[0].cpu().detach().numpy(), w=img_size[1],
                                           h=img_size[0]))  # 将[0,1]区间的坐标重新映射回对应图像的坐标
        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

    all_3d_poses = np.concatenate(all_3d_poses, axis=0)
    all_3d_poses_denormalized = np.concatenate(all_3d_poses_denormalized, axis=0)
    output3d_denormalized = all_3d_poses_denormalized

    return all_3d_poses


def resample(n_frames):
    even = np.linspace(0, n_frames, num=96, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    downsample = None
    print(n_frames)
    if n_frames <= 96:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 96):
            keypoints_clip = keypoints[:, start_idx:start_idx + 96, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 96:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
        print(downsample)
    return clips, downsample
