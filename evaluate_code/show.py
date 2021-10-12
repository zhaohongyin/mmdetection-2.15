from mmdet.apis import init_detector, inference_detector
import os
import json
from mmdet.apis import (async_inference_detector, inference_detector,init_detector, show_result_pyplot)
if __name__ == '__main__':
    config_file = '../Cascade_r50_autoaugment_4x_multi/Cascade_r50_autoaugment_4x_multitrain.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = '../Cascade_r50_autoaugment_4x_multi/epoch_48.pth'
    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    test_dir = '/mnt/data/zhaohongyin/road_detect/test/image'
    test_list = os.listdir(test_dir)
    for test_img in test_list:
        result = inference_detector(model, os.path.join(test_dir,test_img))
        model.show_result(os.path.join(test_dir,test_img),result,out_file=os.path.join("./show_dir", test_img))
    #show_result_pyplot(model, test_dir, result, score_thr=0.3)
