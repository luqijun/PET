import torch
from torchvision.ops import nms

def get_box_from_depth(point, depth):
    # 这里只是一个示例，你需要根据实际情况来计算box
    # 假设box是一个以点为中心，深度值为半径的圆形区域
    box_size = depth * 5  # 根据实际情况调整box的大小
    box = [
        point[1] - box_size / 2,
        point[0] - box_size / 2,
        point[1] + box_size / 2,
        point[0] + box_size / 2,
    ]
    return box

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
    boxes = [get_box_from_depth(point, depth) for point, depth in zip(pred_points, depths)]

    # 执行NMS
    iou_threshold = 0.5  # 假设的IOU阈值
    keep = nms_on_boxes(boxes, scores, iou_threshold)

    # 根据NMS后的索引，获取最终的点和得分
    final_points = pred_points[keep]
    final_scores = scores[keep]

    print("Final Points:", final_points)
    print("Final Scores:", final_scores)