python /home/xueyijun/data/Vicy/object_detection/test_one_stage_recognition.py \
 --gallery_path /home/xueyijun/data/Vicy/datasets/20250721/ \
 --gallery_gt /home/xueyijun/data/Vicy/datasets/20250721/gt/  \
 --xml_output_path /home/xueyijun/data/Vicy/object_detection/outputs/20250721/xml_output \
 --eval-options submission_dir=/home/xueyijun/data/Vicy/object_detection/outputs/work_dirs \
 --checkpoint /home/xueyijun/data/Vicy/object_detection/data/checkpoint/lsknet_s_fair_epoch12.pth \
 --config /home/xueyijun/data/Vicy/object_detection/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
 --save_dirs /home/xueyijun/data/Vicy/object_detection/outputs/20250721/test \
 --evaluation False