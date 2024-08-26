"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
from util.misc import NestedTensor
from models.pet import PET
from util.misc import save_tensor_to_image


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def save_split_map(save_path, gt_map, pred_map):

    # 创建一个全零的数组作为间隔
    H, W = gt_map.shape
    gap = np.zeros((H, 10))

    # 使用 numpy.concatenate 拼接 gt、间隔和 pred
    combined = np.concatenate((gt_map, gap, pred_map), axis=1) * 255  # 按宽度（W）方向拼接
    combined = combined.astype(np.uint8)

    # 使用 cv2.imwrite 保存图像
    cv2.imwrite(save_path, combined)


def visualization(samples, targets, pred, vis_dir, gt_split_map=None, gt_seg_head_map=None, pred_split_map=None, pred_seg_head_map=None, **kwargs):
    """
    Visualize predictions
    """
    gts = [t['points'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(1): # images.shape[0]
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw ground-truth points (red)
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1)

        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)

        name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
        # draw split map
        if pred_split_map is not None:
            save_path = os.path.join(vis_dir, '{}_gt{}_pred{}_split_map.jpg'.format(name, len(gts[idx]), len(pred[idx])))
            save_split_map(save_path, gt_split_map, pred_split_map)

        # draw seg_head_map map
        if pred_seg_head_map is not None:
            save_path = os.path.join(vis_dir, '{}_gt{}_pred{}_seg_head_map.jpg'.format(name, len(gts[idx]), len(pred[idx])))
            save_split_map(save_path, gt_seg_head_map, pred_seg_head_map)
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)


# training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    maes = []
    mses = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items() } for t in targets]
        gt_points = [target['points'] for target in targets]

        losses_data, outputs = model(samples, epoch=epoch, train=True,
                                        criterion=criterion, targets=targets)
        loss_dict, weight_dict, losses = losses_data['loss_dict'], losses_data['weight_dict'], losses_data['losses']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # with torch.autograd.detect_anomaly():
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # del losses
        # torch.cuda.empty_cache()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 显示数据
        test_outputs = PET.test_forward_process(outputs, 0.4, targets=targets)
        outputs_scores = torch.nn.functional.softmax(test_outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = test_outputs['pred_points'][0]
        outputs_offsets = test_outputs['pred_offsets'][0]

        # process predicted points
        predict_cnt = len(outputs_scores)
        gt_cnt = sum([tgt['points'].shape[0] for tgt in targets]) # targets[0]['points'].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(mae)
        mses.append(mse)

        # 可视化
        if False:
            vis_dir = "outputs/SHA/vis_train"
            img_h, img_w = samples.tensors.shape[-2:]
            pred_points_first = test_outputs['pred_points_first'].squeeze(0)
            points = [[point[0] * img_h, point[1] * img_w] for point in pred_points_first] # if pred_points_first.shape[1] > 0 else pred_points_first # recover to actual points
            gt_split_map = (test_outputs['gt_split_map'][0].detach().cpu().squeeze(
                0) > 0.5).float().numpy() if 'gt_split_map' in test_outputs else None
            gt_seg_head_map = (test_outputs['gt_seg_head_map'][0].detach().cpu().squeeze(
                0)).float().numpy() if 'gt_seg_head_map' in test_outputs else None
            pred_split_map = (test_outputs['pred_split_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            pred_seg_head_map = (test_outputs['pred_seg_head_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir,
                          gt_split_map=gt_split_map, gt_seg_head_map=gt_seg_head_map,
                          pred_split_map=pred_split_map, pred_seg_head_map=pred_seg_head_map)


        # break

    one_epoch_mae = sum(maes) / len(maes)
    one_epoch_mse = math.sqrt(sum(mses) / len(mses))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("===================Averaged stats===================")
    print(f"one_epoch_mae:{one_epoch_mae:2f} one_epoch_mse:{one_epoch_mse:2f}")
    print(metric_logger)
    print("====================================================")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]

        # inference
        split = True
        if split:
            max_batch = 4
            split_samples_list = split_test_batch(max_batch, samples)
            split_output_result = []
            for split_sample in split_samples_list:
                split_output = model(split_sample, test=True, targets=targets)
                split_output_result.append(split_output)

            outputs = merge_dicts(split_output_result)
        else:
            outputs = model(samples, test=True, targets=targets)

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        outputs_offsets = outputs['pred_offsets'][0]
        
        # process predicted points
        predict_cnt = len(outputs_scores)
        gt_cnt = targets[0]['points'].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

        # record results
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda()
        results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        metric_logger.update(mae=results['mae'], mse=results['mse'])

        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'])

        # torch.cuda.empty_cache()

        # visualize predictions
        if vis_dir:
            # vis_dir = "outputs/SHA/vis"
            test_outputs = outputs
            img_h, img_w = samples.tensors.shape[-2:]
            pred_points_first = test_outputs['pred_points_first'].squeeze(0)
            points = [[point[0] * img_h, point[1] * img_w] for point in
                      pred_points_first]  # if pred_points_first.shape[1] > 0 else pred_points_first # recover to actual points
            gt_split_map = (test_outputs['gt_split_map'][0].detach().cpu().squeeze(
                0) > 0.5).float().numpy() if 'gt_split_map' in test_outputs else None
            gt_seg_head_map = (test_outputs['gt_seg_head_map'][0].detach().cpu().squeeze(
                0)).float().numpy() if 'gt_seg_head_map' in test_outputs else None
            pred_split_map = (test_outputs['pred_split_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            pred_seg_head_map = (test_outputs['pred_seg_head_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir,
                          gt_split_map=gt_split_map, gt_seg_head_map=gt_seg_head_map,
                          pred_split_map=pred_split_map, pred_seg_head_map=pred_seg_head_map)

            # points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
            # gt_split_map = (outputs['gt_split_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy() if 'gt_split_map' in outputs else None
            # gt_seg_head_map = (outputs['gt_seg_head_map'][0].detach().cpu().squeeze(0)).float().numpy() if 'gt_seg_head_map' in outputs else None
            # pred_split_map = (outputs['pred_split_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            # pred_seg_head_map = (outputs['pred_seg_head_map'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            # visualization(samples, targets, [points], vis_dir,
            #               gt_split_map=gt_split_map, gt_seg_head_map=gt_seg_head_map,
            #               pred_split_map=pred_split_map, pred_seg_head_map=pred_seg_head_map)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mse'] = np.sqrt(results['mse'])
    return results




def split_test_batch(max_batch, samples):

    total_num = samples.tensors.shape[0]
    last_batch_num = total_num % max_batch
    group_num = total_num // max_batch
    size_list = [max_batch for _ in range(group_num)]
    if last_batch_num > 0:
        size_list.append(last_batch_num)

    def split_nest_tensor(nested_tensor):
        samples_tensors_list = torch.split(samples.tensors, size_list, dim=0)
        samples_masks_list = torch.split(samples.mask, size_list, dim=0)
        res = []
        for tensors, mask in zip(samples_tensors_list, samples_masks_list):
            res.append(NestedTensor(tensors, mask))
        return res

    split_samples_result = split_nest_tensor(samples)
    # split_features_result = []
    # for fea_4x, fea_8x in zip(split_nest_tensor(features['4x']), split_nest_tensor(features['8x'])):
    #     split_features_result.append({'4x': fea_4x, '8x': fea_8x})
    #
    # split_pos_result = []
    # for pos_4x, pos_8x in zip(split_nest_tensor(features['4x']), split_nest_tensor(features['8x'])):
    #     split_features_result.append({'4x': fea_4x, '8x': fea_8x})
    #
    # dense_input_embed_result = kwargs['dense_input_embed']

    return split_samples_result #, split_features_result, dense_input_embed_result


def merge_dicts(list_of_dicts):
    if not list_of_dicts:
        return {}

    # 获取第一个字典的键，假设所有字典的键相同
    example_dict = list_of_dicts[0]
    merged_dict = {key: [] for key in example_dict.keys()}

    # 遍历列表，收集每个键对应的张量
    for d in list_of_dicts:
        for key, tensor in d.items():
            merged_dict[key].append(tensor)

    # 将列表中的张量堆叠起来
    for key in merged_dict.keys():
        if "pred_" in key and "map" not in key:
            merged_dict[key] = torch.cat(merged_dict[key], dim=1)

    return merged_dict