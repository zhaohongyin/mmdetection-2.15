## 更新日志

- **10.12 加入类间nms选项.**  
在config->dict(rcnn) -> interClass=True  
模型输出预测结果后会将所有类别的bbox进行nms操作  

## 

<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)

Documentation: https://mmdetection.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.3+**.
The old v1.x branch works with PyTorch 1.1 to 1.4, but v2.0 is strongly recommended for faster speed, higher performance, better design and more friendly usage.

![demo image](resources/coco_test_12510.jpg)

