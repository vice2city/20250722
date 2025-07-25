
# CUDA_VISIBLE_DEVICES=9 python test.py /data5/laiping/tianzhibei/pretrain_datasets/FAIR1M/test/images /data5/laiping/tianzhibei/output_path/Fair1mv2-11-9
# 
# CUDA_VISIBLE_DEVICES=0,3,6,8,9 python -u -m torch.distributed.launch \
#     --nnodes=1  \
#     --nproc_per_node=5 \
#     --master_port=5577 \
#     test.py \
#     /data5/laiping/tianzhibei/pretrain_datasets/FAIR1M/test/images \
#     /data5/laiping/tianzhibei/output_path/Fair1mv2-11-9 \
#     --launcher pytorch


#  CUDA_VISIBLE_DEVICES=8 python /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/test_two_stage_retrieval.py \
#  --gallery_path /data5/laiping/tianzhibei/test-retrieval-11-13/gallery/images \
#  --gallery_gt /data5/laiping/tianzhibei/test-retrieval-11-13/gallery/gt  \
#  --xml_output_path /data5/laiping/tianzhibei/test-retrieval-11-13/output_path \
#  --refine_output_path /data5/laiping/tianzhibei/test-retrieval-11-13/refine_output_path \
#  --eval-options submission_dir=/data5/laiping/tianzhibei/test-retrieval-11-13/work_dirs \
#  --checkpoint /data5/laiping/tianzhibei/checkpoint/lsknet_s_fair_epoch12.pth \
#  --config /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
#  --gallery_crop_path /data5/laiping/tianzhibei/test-retrieval-11-13/gallery_crop \
#  --query_path /data5/laiping/tianzhibei/test-retrieval-11-13/query/images \
#  --query_gt /data5/laiping/tianzhibei/test-retrieval-11-13/query/labelxml \
#  --query_crop_path /data5/laiping/tianzhibei/test-retrieval-11-13/query_crop \
#  --retrieval True 

#  CUDA_VISIBLE_DEVICES=0 python /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/test_two_stage_retrieval.py \
#  --gallery_path /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/gallery/images \
#  --gallery_gt /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/gallery/gt_26  \
#  --xml_output_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/output_path \
#  --refine_output_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/refine_output_path \
#  --eval-options submission_dir=/data5/laiping/tianzhibei/test-retrieval-11-14-unseen/work_dirs \
#  --checkpoint /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2-11-10-less/best_epoch_14.pth \
#  --config /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90_new.py \
#  --gallery_crop_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/gallery_crop \
#  --query_path /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/query/images \
#  --query_gt /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/query/labelxml-26 \
#  --query_crop_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/query_crop \
#  --retrieval True 


 CUDA_VISIBLE_DEVICES=0 python /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/test_two_stage_retrieval.py \
 --gallery_path /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/gallery/images \
 --gallery_gt /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/gallery/gt_26  \
 --xml_output_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/output_path \
 --refine_output_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/refine_output_path \
 --eval-options submission_dir=/data5/laiping/tianzhibei/test-retrieval-11-14-unseen/work_dirs \
 --checkpoint /data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2-11-10-less/best_epoch_14.pth \
 --config /data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90_new.py \
 --gallery_crop_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/gallery_crop \
 --query_path /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/query/images \
 --query_gt /data5/laiping/tianzhibei/test-retrieval-11-13-unseen/query/labelxml-26 \
 --query_crop_path /data5/laiping/tianzhibei/test-retrieval-11-14-unseen/query_crop \
 --retrieval True 