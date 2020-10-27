#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import *
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters, triangulate_correspondences, pose_to_view_mat3x4, build_correspondences,
    rodrigues_and_translation_to_view_mat3x4)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corners: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    prms = TriangulationParameters(max_reprojection_error=1., min_triangulation_angle_deg=2., min_depth=0.1)
    track = [None for _ in range(len(corners))]
    points = [None for _ in range(corners.max_corner_id() + 1)]
    track[known_view_1[0]], track[known_view_2[0]] = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(
        known_view_2[1])
    new_points, ids, _ = triangulate_correspondences(
        build_correspondences(corners[known_view_1[0]], corners[known_view_2[0]]),
        track[known_view_1[0]], track[known_view_2[0]],
        intrinsic_mat, prms)
    new_ids = list(filter(lambda el: points[el[1]] is None, zip(new_points, ids)))
    for point, i in new_ids:
        points[i] = point
    if len(new_ids) > 0:
        print(f' {len(new_ids)} new points')
    flag = True
    while flag:
        flag = False
        for i in range(len(track)):
            if track[i] is None:
                flag = tracker(i, track, points, corners, prms, intrinsic_mat)

    for i in range(1, len(track)):
        if track[i] is None:
            track[i] = track[i - 1]
    view_mats = np.array(track)
    ids = np.array([i for i in range(len(points)) if points[i] is not None])
    points = np.array([points[i] for i in range(len(points)) if points[i] is not None])
    point_cloud_builder = PointCloudBuilder(ids=ids, points=points)
    calc_point_cloud_colors(point_cloud_builder, rgb_sequence, view_mats, intrinsic_mat, corners, 5.0)
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


def tracker(tid: int,
            track: List[Optional[np.ndarray]],
            pts: List[Optional[np.ndarray]],
            corners: CornerStorage,
            params: TriangulationParameters,
            intrinsic_mat: np.ndarray) -> bool:
    ids = corners[tid].ids.reshape(-1)
    cur = [(i, pt, pts[i]) for i, pt in zip(ids, corners[tid].points) if
           pts[i] is not None]
    if len(cur) < 5:
        return False
    cur_corner_ids, fr_points, cur_points = zip(*cur)
    cur_corner_ids, fr_points, cur_points = np.array(cur_corner_ids), np.array(fr_points), np.array(cur_points)
    a, b, c, d = cv2.solvePnPRansac(cur_points, fr_points, intrinsic_mat, None)
    if not a:
        return False
    for i in [i for i in ids if i not in d]:
        pts[i] = None
    track[tid] = rodrigues_and_translation_to_view_mat3x4(b, c)
    for j in range(len(track)):
        if id == j or track[j] is None:
            continue
        new_points, ids, median_cos = triangulate_correspondences(build_correspondences(corners[tid], corners[j]),
                                                                  track[tid], track[j],
                                                                  intrinsic_mat, params)
        new_ids = list(filter(lambda el: pts[el[1]] is None, zip(new_points, ids)))
        if len(new_ids) > 0:
            print(f' {len(new_ids)} new points')
        for point, i in new_ids:
            pts[i] = point
    return True


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
