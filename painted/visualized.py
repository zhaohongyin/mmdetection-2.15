import numpy as np
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import mmcv
import os

config_area='/root/wangjinqiao/zhaohongyin/mmdetection-master/work_dirs/r50_3x_204_rotate_eqlv2_0903/r50_3x_204_rotate_eqlv2_0903.py'
checkpoint_area = '/root/wangjinqiao/zhaohongyin/mmdetection-master/work_dirs/r50_3x_204_rotate_eqlv2_0903/epoch_72.pth'

model_area = init_detector(config_area, checkpoint_area, device='cuda:0')


root="/root/wangjinqiao/zhaohongyin/mmdetection-master/painted/test_tiaohe/"
for root, dirs, files in os.walk(root):

    for i in range(len(files)):
        img=root+files[i]
        import pdb
        #pdb.set_trace()
        result = inference_detector(model_area, img)
        #for j in range(len(result)):
        #    if j!=56 and j!=60:
        #        result[j]=np.empty(shape=(0,5),dtype=np.float32)

        model_area.show_result(img, result, out_file='./output/'+files[i])
        
