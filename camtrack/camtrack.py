#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import time
from typing import List, Optional, Tuple

import numpy as np
import cv2
from ba import run_bundle_adjustment
from corners import *
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    eye3x4,
    remove_correspondences_with_ids,
    TriangulationParameters, triangulate_correspondences, pose_to_view_mat3x4, build_correspondences,
    rodrigues_and_translation_to_view_mat3x4)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corners: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    prms = TriangulationParameters(max_reprojection_error=2.2, min_triangulation_angle_deg=1., min_depth=0.1)
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = get_views(corners, intrinsic_mat, prms)

    track = [None for _ in range(len(corners))]
    points = [None for _ in range(corners.max_corner_id() + 1)]
    track[known_view_1[0]], track[known_view_2[0]] = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(
        known_view_2[1])

    corres = build_correspondences(corners[known_view_1[0]], corners[known_view_2[0]])
    new_ids_size = 0
    if len(corres.points_1) != 0 and len(corres.points_2) != 0:
        new_points, ids, _ = triangulate_correspondences(
            corres,
            track[known_view_1[0]], track[known_view_2[0]],
            intrinsic_mat, prms)
        new_ids = list(filter(lambda el: points[el[1]] is None, zip(new_points, ids)))
        for point, i in new_ids:
            points[i] = point
        new_ids_size = len(new_ids)
    if new_ids_size > 0:
        print(f' {new_ids_size} new points')

    flag = True
    ba_step = 5
    while flag:
        flag = False
        for i in range(len(track)):
            if track[i] is None:
                flag = tracker(i, track, points, corners, prms, intrinsic_mat)
                if i > 0 and i % ba_step == 0:
                    track[i - ba_step + 1: i + 1], points = \
                        run_bundle_adjustment(intrinsic_mat,
                                              corners[i - ba_step + 1: i + 1],
                                              prms.max_reprojection_error,
                                              track[i - ba_step + 1: i + 1],
                                              points)
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


def get_poses(retval):
    R1, R2, t = cv2.decomposeEssentialMat(retval)
    return [Pose(R1.T, np.dot(R1.T, t)), Pose(R2.T, np.dot(R2.T, t)),
            Pose(R1.T, np.dot(R1.T, (-t))), Pose(R2.T, np.dot(R2.T, (-t)))]


def get_views(corners: CornerStorage,
              intrinsic_mat: np.ndarray,
              prms: TriangulationParameters) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    max_size, pose_for_max_size, index_for_max_size = -1, None, None
    init_ind = 0
    for i in range(1, len(corners)):
        corres = build_correspondences(corners[init_ind], corners[i])
        pose, cloud_size = None, -1
        if len(corres.points_1) >= 5:
            retval, mask = cv2.findEssentialMat(corres.points_1, corres.points_2,
                                                intrinsic_mat,
                                                method=cv2.RANSAC, prob=0.9999, threshold=1.)
            hretval, hmask = cv2.findHomography(corres.points_1,
                                                corres.points_2,
                                                ransacReprojThreshold=prms.max_reprojection_error,
                                                method=cv2.RANSAC, confidence=0.9999, )
            if not (retval is None) \
                    and retval.shape == (3, 3) \
                    and np.count_nonzero(mask) >= np.count_nonzero(hmask):
                corres = remove_correspondences_with_ids(corres, np.argwhere(mask == 0))
                poses = get_poses(retval)
                tr = [triangulate_correspondences(corres, eye3x4(), pose_to_view_mat3x4(pose),
                                                  intrinsic_mat, prms) for pose in poses]
                cloud_size, _best_ind = max((len(tr[i][0]), i) for i in range(4))
                pose = poses[_best_ind]
        if cloud_size > max_size:
            max_size, pose_for_max_size, index_for_max_size = cloud_size, pose, i
        print(f'cloud size = {cloud_size}, frames={init_ind}, {i}')
    return (0, view_mat3x4_to_pose(eye3x4())), (index_for_max_size, pose_for_max_size,)


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
        corres = build_correspondences(corners[tid], corners[j])
        if len(corres.points_1) == 0 or len(corres.points_2) == 0:
            continue
        new_points, ids, median_cos = triangulate_correspondences(corres,
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
