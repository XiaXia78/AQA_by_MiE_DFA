***Citation:***  

Aoyu Xia, "Interpretable Two-Stage Action Quality Assessment via 3D Human Pose Estimation and Dynamic Feature Alignment" *Submitted to The Visual Computer*, 2025.


***3D HPE***

  **Dataset**
  
  Human3.6M(http://vision.imar.ro/human3.6m/description.php)
  
  **Train**
  
  %run PoseDetector3d/train.py --config PoseDetector3d/configs/MiEFormer.yaml


***AQA**

  **source**
  
  Chen Style Tai Chi Essential 18 Forms(https://www.bilibili.com/video/BV1aW41167dV/?spm_id_from=333.337.search-card.all.click&vd_source=daacd9dfcabc91048b8fa9d59550e9b1)


***How to use***

  %run PoseSequenceCompare/AQA.py --video-path your_video_path
