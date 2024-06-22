# CUDA_VISIBLE_DEVICES=2 python tools/train.py \
# /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_dota_le90.py \
# --work-dir /data5/laiping/tianzhibei/exp/LSKNet 

# ./tools/dist_train.sh /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_dota_le90.py \
# 2 --work-dir /data5/laiping/tianzhibei/exp/LSKNet 

# CUDA_VISIBLE_DEVICES=2 python tools/train.py \
# /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_le90.py \
# --work-dir /data5/laiping/tianzhibei/exp/LSKNet-plane-all

# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_all_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet-plane-all

#  CUDA_VISIBLE_DEVICES=4 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_sar_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet-plane-sar

 CUDA_VISIBLE_DEVICES=5 python train.py \
 /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_optical_le90.py \
 --work-dir /data5/laiping/tianzhibei/exp/LSKNet-plane-optical