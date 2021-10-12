import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class TiaoheDataset(CocoDataset):

    CLASSES = ('tiaohe', '555_j_tiaohe', '555_xkxz_tiaohe', '555jin_r_tiaohe', 'aixi_hb_tiaohe', 'baisha_htx_tiaohe',
               'baisha_xzhtx_tiaohe', 'baisha_yjpsd_tiaohe', 'baisha_yttxsxz_tiaohe', 'baisha_yxjped_tiaohe',
               'changbaishan_777_tiaohe', 'changbaishan_h_tiaohe', 'changcheng_cybh_tiaohe', 'changcheng_hs132_tiaohe',
               'daqianmen_dz_tiaohe', 'daqianmen_r_tiaohe', 'diaoyutai_84mm_tiaohe', 'diaoyutai_zz_tiaohe',
               'dongchongxiacao_hr_tiaohe', 'dongchongxiacao_j_tiaohe', 'dongchongxiacao_szz_tiaohe',
                'fenghuang_xz_tiaohe', 'furongwang_y_tiaohe', 'furongwang_yxz_tiaohe',
               'furongwang_yzz_tiaohe', 'guiyan_c_tiaohe', 'guiyan_gjx30_tiaohe', 'guiyan_ky_tiaohe',
               'guiyan_xzgjx30_tiaohe', 'guiyan_xzxz_tiaohe', 'hademen_cx_tiaohe', 'hademen_jd_tiaohe',
               'hademen_r_tiaohe', 'haerbin_lbd_tiaohe', 'haomao_jsh_tiaohe', 'haomao_xzcl_tiaohe',
               'hongqiqu_mg_tiaohe', 'hongqiqu_xj_tiaohe', 'hongshuangxi_lf_tiaohe', 'hongtashan_rjd_tiaohe',
               'hongtashan_ycq_tiaohe', 'huangguoshu_cz_tiaohe', 'huanghelou_gezz_tiaohe', 'huanghelou_rl_tiaohe',
               'huanghelou_ryy_tiaohe', 'huanghelou_txml_tiaohe', 'huanghelou_xzj9h_tiaohe', 'huanghelou_y1916_tiaohe',
               'huanghelou_y8d_tiaohe', 'huanghelou_yjsly_tiaohe', 'huanghelou_yl_tiaohe', 'huanghelou_ypa_tiaohe',
               'huanghelou_yqj_tiaohe', 'huanghelou_yxgqxz_tiaohe', 'huanghelou_yxgrq_tiaohe', 'huanghelou_yyz_tiaohe',
               'huangjinye_as_tiaohe', 'huangjinye_jmt_tiaohe', 'huangjinye_lt_tiaohe', 'huangjinye_txxz_tiaohe',
               'huangjinye_ty_tiaohe', 'huangjinye_tyxz_tiaohe', 'huangjinye_xmb_tiaohe', 'huangjinye_xmt_tiaohe',
               'huangshan_dhfy_tiaohe', 'huangshan_hfyxz_tiaohe', 'huangshan_hsxgnxz_tiaohe', 'huangshan_jy_tiaohe',
               'huangshan_xhfy_tiaohe', 'huangshan_xyp_tiaohe', 'huangshan_xzhy_tiaohe', 'huangshan_yxyp_tiaohe',
               'jianpai_bh4_tiaohe', 'jianpai_bhzk4_tiaohe', 'jiaozi_gdxz_tiaohe', 'jiaozi_kzhyxz_tiaohe',
               'jiaozi_kzxyxz_tiaohe', 'jiaozi_wlcx_tiaohe', 'jiaozi_wlcxzz_tiaohe', 'jiaozi_x_tiaohe',
               'jinqiao_bb_tiaohe', 'jinsheng_twgzg_tiaohe', 'jinsheng_yzyw_tiaohe', 'liqun_jny_tiaohe',
               'liqun_lt_tiaohe', 'liqun_lwl_tiaohe', 'liqun_rcz_tiaohe', 'liqun_rhcz_tiaohe', 'liqun_xb_tiaohe',
               'liqun_xhl_tiaohe', 'liqun_xxyd_tiaohe', 'liqun_xzyg_tiaohe', 'liqun_yxh_tiaohe',
               'longfengchengxiang_hkfg_tiaohe', 'mudan_jxz_tiaohe', 'mudan_lzz_tiaohe', 'mudan_qnxz_tiaohe',
               'mudan_r_tiaohe', 'nanjing_dgybb_tiaohe', 'nanjing_h_tiaohe', 'nanjing_hlj_tiaohe', 'nanjing_jp_tiaohe',
               'nanjing_jw_tiaohe', 'nanjing_rjw_tiaohe', 'nanjing_secbh_tiaohe', 'nanjing_secky_tiaohe',
               'nanjing_seczshhx_tiaohe', 'nanjing_xhm_tiaohe', 'nanjing_xhmxc_tiaohe', 'nanjing_xzjw_tiaohe',
               'nanjing_yhs_tiaohe', 'qipilang_b_tiaohe', 'qipilang_cj_tiaohe', 'qipilang_gtjzz_tiaohe',
               'qipilang_hq_tiaohe', 'renmindahuitang_sjzz_tiaohe', 'renmindahuitang_yhxz_tiaohe', 'shuangxi_hy_tiaohe',
               'suyan_cx_tiaohe', 'suyan_cz_tiaohe', 'suyan_js2_tiaohe', 'suyan_rjs_tiaohe', 'suyan_ty_tiaohe',
               'suyan_wxhss_tiaohe', 'taishan_3cx_tiaohe', 'taishan_bhm_tiaohe', 'taishan_bjj_tiaohe',
               'taishan_bjxz_tiaohe', 'taishan_bx_tiaohe', 'taishan_csjj_tiaohe', 'taishan_df_tiaohe',
               'taishan_dj_tiaohe', 'taishan_fg_tiaohe', 'taishan_fgxz_tiaohe', 'taishan_hb_tiaohe',
               'taishan_hbxz_tiaohe', 'taishan_hdmyh_tiaohe', 'taishan_hg_tiaohe', 'taishan_hhxx_tiaohe',
               'taishan_hjj_tiaohe', 'taishan_hjlp21x_tiaohe', 'taishan_hkxz_tiaohe', 'taishan_hp_tiaohe',
               'taishan_ht_tiaohe', 'taishan_jjzz_tiaohe', 'taishan_jpxz_tiaohe', 'taishan_mlxy_tiaohe',
               'taishan_pa_tiaohe', 'taishan_qx_tiaohe', 'taishan_rf_tiaohe', 'taishan_rfxz_tiaohe',
               'taishan_wy_tiaohe', 'taishan_xbssz_tiaohe', 'taishan_xbxz_tiaohe', 'taishan_xp_tiaohe',
               'taishan_xy_tiaohe', 'taishan_yhbx_tiaohe', 'taishan_ym_tiaohe', 'taishan_yy_tiaohe',
               'taishan_zhyy_tiaohe', 'taishan_zzjj_tiaohe', 'tianzi_j_tiaohe', 'tianzi_zz_tiaohe',
               'wanbaolu_rh2_tiaohe', 'wanbaolu_yh2_tiaohe', 'wangguan_s10_tiaohe', 'yanan_zz1935_tiaohe',
               'yunyan_crhy_tiaohe', 'yunyan_f_tiaohe', 'yunyan_rdzj_tiaohe', 'yunyan_rz_tiaohe', 'yunyan_xxmjy_tiaohe',
               'yunyan_xzdzj_tiaohe', 'yunyan_xzyl_tiaohe', 'yunyan_xzzp_tiaohe', 'yunyan_z_tiaohe', 'yuxi_108_tiaohe',
               'yuxi_ck_tiaohe', 'yuxi_gpb_tiaohe', 'yuxi_hy_tiaohe', 'yuxi_r_tiaohe', 'yuxi_rasm_tiaohe',
               'yuxi_xz108_tiaohe', 'yuxi_xzcx_tiaohe', 'yuxi_xzqxsj_tiaohe', 'yuxi_zzhx_tiaohe', 'zhenlong_jdh_tiaohe',
               'zhenlong_lsj_tiaohe', 'zhenlong_ly_tiaohe', 'zhenlong_zzly_tiaohe', 'zhonghua_jdz_tiaohe',
               'zhonghua_jxz_tiaohe', 'zhonghua_jzz_tiaohe', 'zhonghua_qks_tiaohe', 'zhonghua_r_tiaohe',
               'zhonghua_szz_tiaohe', 'zhonghua_xz_tiaohe', 'zhonghua_y_tiaohe', 'zhongnanhai_5mgxz_tiaohe',
               'zhongnanhai_j8mg_tiaohe', 'zuanshi_hh_tiaohe', 'zuanshi_rhh_tiaohe', 'zuanshi_xzhh_tiaohe',
               'zuanshi_xzzhh_tiaohe')
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,#defualt False
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=[0.50], #defualt None add for cigarette evaluate
                 #iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
