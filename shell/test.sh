# CUDA_VISIBLE_DEVICES=3 python ./tools/test.py  \
#   /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_dota_le90.py \
#   /data5/laiping/tianzhibei/checkpoint/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth --format-only \
#   --eval-options submission_dir=work_dirs/Task1_results1
# python ./tools/test.py  \
#   /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
#   /data5/laiping/tianzhibei/checkpoint/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth --eval mAP --gpu-ids 1

# CUDA_VISIBLE_DEVICES=3 python ./tools/test.py  \
#   /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_sar_le90.py \
#   /data5/laiping/tianzhibei/exp/LSKNet-plane-sar/latest.pth --format-only \
#   --eval-options submission_dir=/data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/work_dirs/plane_results_sar_new --output_path "/data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/output_path/SAR"

# python ./tools/run_plane.py  /data5/laiping/tianzhibei/code/input_path/plane /data5/laiping/tianzhibei/code/output_path
python ./tools/run_car.py  /data5/laiping/tianzhibei/code/input_path/car /data5/laiping/tianzhibei/code/output_path/car
