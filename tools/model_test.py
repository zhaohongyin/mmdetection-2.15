from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import mmcv
import os
config_file = '/root/wangjinqiao/zhaohongyin/mmdetection-master/work_dir/res50_faster_207_6x_cenn_1105_baseline/res50_6x_207_baseline.py'
checkpoint_file = '/root/wangjinqiao/zhaohongyin/mmdetection-master/work_dir/res50_faster_207_6x_cenn_1105_baseline/epoch_6.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:4')
import pdb
pdb.set_trace()

root="/root/wangjinqiao/zhaohongyin/mmdetection-master/painted/test_yanhe/"
for root, dirs, files in os.walk(root):

    for i in range(len(files)):
        img=root+files[i]
        result = inference_detector(model, img)
        model.show_result(img, result, out_file='./pic/'+files[i])
