# SlowFast
python run.py configs/slowfast_r50_4x16x1_256e_kinetics400_rgb.py\
    checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth \
    example.mp4 map/label_map_k400.txt --batch_size 1 --batch_num 11

# # I3D
# python run.py configs/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb.py \
#     checkpoints/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200815-17f84aa2.pth \
#     example.mp4 map/label_map_k400.txt --batch_size 1 --batch_num 11

# # C3D
# python run.py configs/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
#     checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth \
#     example.mp4 map/label_map.txt --batch_size 1 --batch_num 11

# # TSN
# python run.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
#     checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
#     example.mp4 map/label_map_k400.txt --batch_size 1 --batch_num 11
