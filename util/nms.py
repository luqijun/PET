import torch
from torchvision.ops import nms

def get_boxes_from_depths(points, depths, base_size=20, scale=1.0, min_size=8.0, img_h=None, img_w=None):
    box_sizes = base_size * scale * depths
    box_sizes = torch.clamp(box_sizes, min=min_size)
    lt_points = points - box_sizes.unsqueeze(-1)
    rb_points = points + box_sizes.unsqueeze(-1)
    anchor_bboxes = torch.cat([lt_points, rb_points], dim=-1) # (y1, x1, y2, x2)

    if img_h is not None:
        anchor_bboxes[:, 0] = torch.clamp(anchor_bboxes[:, 0], max=img_h, min=0)
        anchor_bboxes[:, 2] = torch.clamp(anchor_bboxes[:, 2], max=img_h, min=0)
    if img_w is not None:
        anchor_bboxes[:, 1] = torch.clamp(anchor_bboxes[:, 1], max=img_w, min=0)
        anchor_bboxes[:, 3] = torch.clamp(anchor_bboxes[:, 3], max=img_w, min=0)

    return anchor_bboxes


# def get_box_from_depth(point, depth, base_size=60, scale=1.0, min_size=8):
#     # 这里只是一个示例，你需要根据实际情况来计算box
#     # 假设box是一个以点为中心，深度值为半径的圆形区域
#     box_size = max(base_size * scale * depth, min_size)  # 根据实际情况调整box的大小
#     box = [
#         point[1] - box_size / 2,
#         point[0] - box_size / 2,
#         point[1] + box_size / 2,
#         point[0] + box_size / 2,
#     ]
#     return box

def nms_on_boxes(boxes, scores, iou_threshold):
    # 将boxes转换为tensor
    boxes_tensor = torch.tensor(boxes, device=scores.device)
    # 执行NMS
    keep = nms(boxes_tensor, scores, iou_threshold)
    return keep


if __name__ == '__main__':
    # 假设pred_points和scores是已经给出的预测点和得分
    pred_points = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    depths = torch.tensor([0.5, 0.4, 0.3, 0.2])  # 假设深度值

    # 计算每个点的box
    boxes = get_boxes_from_depths(pred_points, depths)

    # 执行NMS
    iou_threshold = 0.5  # 假设的IOU阈值
    keep = nms_on_boxes(boxes, scores, iou_threshold)

    # 根据NMS后的索引，获取最终的点和得分
    final_points = pred_points[keep]
    final_scores = scores[keep]

    print("Final Points:", final_points)
    print("Final Scores:", final_scores)