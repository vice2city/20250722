# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
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
# 
#  CUDA_VISIBLE_DEVICES=1 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_optical_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet-plane-optical-all-1

# CUDA_VISIBLE_DEVICES=0 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/mobilenet/trans.py \
#  --work-dir /data5/laiping/tianzhibei/exp/swin-transformer

# "/data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/mobilenet/trans.py","--work-dir","/data5/laiping/tianzhibei/exp/swin-transformer"

# CUDA_VISIBLE_DEVICES=9 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/LSKNet_self_attention/lsk_s_self_attention_fpn_1x_dota_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet_self_attention

# CUDA_VISIBLE_DEVICES=9 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/LSKNet_multihead/lsk_s_multihead_fpn_1x_dota_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet_multihead


# CUDA_VISIBLE_DEVICES=1 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/swin_transformer_fpn_1x_dota_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet-swin

# CUDA_VISIBLE_DEVICES= python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2

# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,8,9 python -u -m torch.distributed.launch \
#     --nnodes=1  \
#     --nproc_per_node=8 \
#     --master_port=5568 \
#     train.py \
#     /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
#     --work-dir /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2-11-8 \
#     --launcher pytorch


# CUDA_VISIBLE_DEVICES=9 python train.py \
#  /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
#  --work-dir /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2_new-11-8


CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,8,9 python -u -m torch.distributed.launch \
    --nnodes=1  \
    --nproc_per_node=8 \
    --master_port=5566 \
    train.py \
    /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
    --work-dir /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2_wlsknet_1 \
    --launcher pytorch

# CUDA_VISIBLE_DEVICES=4,5,6,8,9 python -u -m torch.distributed.launch \
#     --nnodes=1  \
#     --nproc_per_node=5 \
#     --master_port=5568 \
#     train.py \
#     /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90_new_retrieval.py \
#     --work-dir /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2-11-13-less \
#     --launcher pytorch