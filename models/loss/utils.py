import torch

from util.misc import check_and_clear_memory


def split_and_compute_cdist(points1, points2, n=1, p=2):
    """
    将 points1 分成 n 份，分别与 points2 计算距离，最后将结果合并返回。

    :param points1: 形状为 (N, D) 的 Tensor，表示 N 个 D 维点。
    :param points2: 形状为 (M, D) 的 Tensor，表示 M 个 D 维点。
    :param n: 将 points1 分成的份数。
    :return: 形状为 (N, M) 的 Tensor，表示 points1 和 points2 之间的距离矩阵。
    """
    # 计算每一份的大小
    points1 = points1.detach()
    points2 = points2.detach()
    batch_size = points1.size(0) // n
    remainder = points1.size(0) % n

    # 初始化结果矩阵
    result = torch.zeros((points1.size(0), points2.size(0)))

    # 分批计算距离
    start_idx = 0
    for i in range(n):
        end_idx = start_idx + batch_size + (1 if i < remainder else 0)
        batch_points1 = points1[start_idx:end_idx]
        dist_matrix = torch.cdist(batch_points1, points2, p=p)
        result[start_idx:end_idx] = dist_matrix
        start_idx = end_idx
        # if i >= 8:
        #     check_and_clear_memory(0.9)

    return result.to(points1.device)


def split_and_compute_cdist2(points1, points1_sizes, points2, points2_sizes, p=2):
    """
    将 points1 分成 n 份，分别与 points2 计算距离，最后将结果合并返回。

    :param points1: 形状为 (N, D) 的 Tensor，表示 N 个 D 维点。
    :param points2: 形状为 (M, D) 的 Tensor，表示 M 个 D 维点。
    :param n: 将 points1 分成的份数。
    :return: 形状为 (N, M) 的 Tensor，表示 points1 和 points2 之间的距离矩阵。
    """
    # 计算每一份的大小
    points1_tuple = torch.split(points1, points1_sizes, dim=0)
    points2_tuple = torch.split(points2, points2_sizes, dim=0)

    # 初始化结果矩阵
    result = torch.zeros((points1.size(0), points2.size(0)), device=points1.device)
    pts1_sizes = torch.cumsum(torch.tensor([0] + points1_sizes), dim=0)
    pts2_sizes = torch.cumsum(torch.tensor([0] + points2_sizes), dim=0)
    for idx, (pts1, pts2) in enumerate(zip(points1_tuple, points2_tuple)):
        dist_matrix = torch.cdist(pts1, pts2, p=p)
        result[pts1_sizes[idx]:pts1_sizes[idx + 1], pts2_sizes[idx]:pts2_sizes[idx + 1]] = dist_matrix
        # if idx >= 8:
        #     check_and_clear_memory(0.9)

    return result
