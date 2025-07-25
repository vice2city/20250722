python /data5/laiping/tianzhibei/code/object_detection/test_one_stage_recognition.py \
 --gallery_path /data5/laiping/tianzhibei/pretrain_datasets/FAIR1M/val/input_path \
 --gallery_gt /data5/laiping/tianzhibei/pretrain_datasets/FAIR1M/val/gt  \
 --xml_output_path /data5/laiping/tianzhibei/test-recogntion-5-20-1/output_path \
 --eval-options submission_dir=/data5/laiping/tianzhibei/test-recogntion-5-20-1/work_dirs \
 --checkpoint /data5/laiping/tianzhibei/checkpoint/lsknet_s_fair_epoch12.pth \
 --config /data5/laiping/tianzhibei/code/object_detection/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
 --save_dirs /data5/laiping/tianzhibei/test-recogntion-5-20-1/test \
 --evaluation True

# gallery_path="/code/dataset/images",
# gallery_gt="/code/dataset/xml",
# xml_output_path="/code/output_path",
# eval-options="submission_dir=/code/work_dirs",
# checkpoint="/code/checkpoint/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth",
# config="code/object_detection/configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py",
# evaluation="True"

# python /code/object_detection/test_one_stage_recognition_cpu.py \
#  --gallery_path /code/datset/DOTA/images \
#  --gallery_gt /code/datset/DOTA/xml \
#  --xml_output_path /code/output_path/test-12-20/output_path \
#  --eval-options submission_dir=/code/work_dirs/test-12-20 \
#  --checkpoint /code/checkpoint/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth \
#  --config /code/object_detection/configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py \
#  --save_dirs /code/temp/test \
#  --evaluation True