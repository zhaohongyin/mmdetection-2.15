from mmdet.apis import init_detector, inference_detector
import os
import json
import cv2
if __name__ == '__main__':
    config_file = '/root/wangjinqiao/zhaohongyin/mmdetection-master/road_model/Cascade_res50_auto_dcnv2_4x_multi_0813/Cascade_r50_auto+dcnv2_4x_multitrain.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = '/root/wangjinqiao/zhaohongyin/mmdetection-master/road_model/Cascade_res50_auto_dcnv2_4x_multi_0813/epoch_48.pth'
    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    test_dir = '/root/wangjinqiao/zhaohongyin/dataset/Road_detect/test/JPEGImages'
    test_list = os.listdir(test_dir)
    json_dir = '../result_semi'
    for test_img in test_list:
        img=cv2.imread(os.path.join(test_dir, test_img)).shape
        W=img[1]
        H=img[0] 
        class_name = ['Crack', 'Net', 'AbnormalManhole', 'Pothole', 'Marking']
        output = []
        result = inference_detector(model, os.path.join(test_dir, test_img))
        for i in range(len(result)):
            category_name = class_name[i]
            bboxes = result[i]
            if not bboxes.shape[0]:
                continue
            else:
                for bbox in bboxes:
                    x0, y0, x1, y1, score = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(bbox[4])
                    if x1>=W:
                       x1=W-1
                    if y1>=H:
                       y1=H-1
                    json_bbox = {"category": category_name, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                                 "score": score}
                    output.append(json_bbox)
        with open(os.path.join(json_dir, test_img.split('.')[0] + '.json'), 'w') as f:
            json.dump(output, f)
