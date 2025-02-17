import numpy as np
from PoseSequenceCompare.video2pose import video2pose
from PoseSequenceCompare.tools import DTW as caculate_distance

instructive_pose="instructive_pose/ChensTaiChi.npz"
if __name__ == '__main__':
    pose1=np.load(instructive_pose)['reconstruction']
    video=input()
    pose2=video2pose(videopath=video)
    score=caculate_distance(pose1,pose2,method='itakura',options={'max_slope':2.0})
    print(score)