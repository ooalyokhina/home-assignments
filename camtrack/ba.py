from typing import List, Dict, Tuple

from collections import namedtuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import approx_fprime as derivative
from time import time

from corners import FrameCorners
from _camtrack import *

ProjectionError = namedtuple(
    'ProjectionError',
    ('frame_id', 'id_3d', 'id_2d')
)


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          point_positions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    print('Running bundle adjustment')
    orig_point_positions = point_positions
    def transf(p):
        if p is not None:
            return p
        return np.zeros(3)
    point_positions = np.array([transf(p) for p in point_positions])
    proj_mats = [intrinsic_mat @ view_mat for view_mat in view_mats]
    projection_errors = []
    used_3d_points_inds = set()

    for k, (proj_mat, corners) in enumerate(zip(proj_mats, list_of_corners)):
        indices = np.array([ind for ind in corners.ids if point_positions[ind] is not None], dtype=np.int32)
        indices_2d_local = np.array([k for k, ind in enumerate(corners.ids) if ind in indices], np.int32)
        indices_3d_local = indices[:, 0]
        inlier_indices = calc_inlier_indices(point_positions[indices_3d_local],
                                             corners.points[indices_2d_local],
                                             proj_mat,
                                             max_inlier_reprojection_error)
        for ind in inlier_indices:
            id_3d = indices_3d_local[ind]
            id_2d = indices_2d_local[ind]
            used_3d_points_inds.add(id_3d)
            projection_errors.append(ProjectionError(frame_id=k, id_3d=id_3d, id_2d=id_2d))

    used_3d_points_inds = list(sorted(used_3d_points_inds))
    point_ind_to_position = {}
    n_matrix_params = 6 * len(view_mats)
    p = np.concatenate([_view_mats3x4_to_rt(view_mats),
                        _points_to_flat_coordinates(point_positions[used_3d_points_inds])])
    for k, point_ind in enumerate(used_3d_points_inds):
        point_ind_to_position[point_ind] = n_matrix_params + 3 * k

    if _run_optimization(projection_errors, list_of_corners, point_ind_to_position, p, intrinsic_mat, 3):
        for k in range(len(view_mats)):
            r_vec = p[6 * k: 6 * k + 3].reshape(3, 1)
            t_vec = p[6 * k + 3: 6 * k + 6].reshape(3, 1)
            view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
            view_mats[k] = view_mat

        for k, ind in enumerate(used_3d_points_inds):
            orig_point_positions[ind] = p[n_matrix_params + 3 * k: n_matrix_params + 3 * k + 3]

    return view_mats, orig_point_positions


def _view_mats3x4_to_rt(view_mats: List[np.ndarray]) -> np.ndarray:
    result = np.zeros(6 * len(view_mats))
    for k, mat in enumerate(view_mats):
        pos = 6 * k
        r, t = view_mat3x4_to_rodrigues_and_translation(mat)
        result[pos: pos + 3] = r[:, 0]
        result[pos + 3: pos + 6] = t[:]
    return result


def _points_to_flat_coordinates(points: np.ndarray) -> np.ndarray:
    return points.reshape(-1)


def _vec_to_proj_mat(vec: np.ndarray, intrinsic_mat: np.ndarray) -> np.ndarray:
    r_vec = vec[0:3].reshape(3, 1)
    t_vec = vec[3:6].reshape(3, 1)
    view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
    return np.dot(intrinsic_mat, view_mat)


def _reprojection_error(vec: np.ndarray, point2d: np.ndarray, intrinsic_mat: np.ndarray) -> np.float32:
    point3d = vec[6:9]
    proj_mat = _vec_to_proj_mat(vec, intrinsic_mat)
    point3d_hom = np.hstack((point3d, 1))
    proj_point2d = np.dot(proj_mat, point3d_hom)
    proj_point2d = proj_point2d / proj_point2d[2]
    proj_point2d = proj_point2d.T[:2]
    proj_error = (point2d - proj_point2d).reshape(-1)
    return np.linalg.norm(proj_error)


def _reprojection_errors(projection_errors: List[ProjectionError],
                         list_of_corners: List[FrameCorners],
                         mapping: Dict[int, int],
                         p: np.ndarray,
                         intrinsic_mat: np.ndarray) -> np.ndarray:
    errors = np.zeros(len(projection_errors))
    for i, proj_err in enumerate(projection_errors):
        vec = np.zeros(9)

        mat_pos = 6 * proj_err.frame_id
        vec[:6] = p[mat_pos: mat_pos + 6]

        point_pos = mapping[proj_err.id_3d]
        vec[6:] = p[point_pos: point_pos + 3]

        point2d = list_of_corners[proj_err.frame_id].points[proj_err.id_2d]

        errors[i] = _reprojection_error(vec, point2d, intrinsic_mat)

    return errors


def _compute_jacobian(projection_errors: List[ProjectionError],
                      list_of_corners: List[FrameCorners],
                      mapping: Dict[int, int],
                      p: np.ndarray,
                      intrinsic_mat: np.ndarray) -> csr_matrix:
    start_time = time()
    print("Started Jacobian computation")
    # J = np.zeros((len(projection_errors), len(p)))
    rows = np.zeros(9 * len(projection_errors), dtype=np.int32)
    cols = np.zeros(9 * len(projection_errors), dtype=np.int32)
    vals = np.zeros(9 * len(projection_errors), dtype=np.float32)
    cur = 0
    for row, proj_err in enumerate(projection_errors):
        vec = np.zeros(9)

        mat_pos = 6 * proj_err.frame_id
        vec[:6] = p[mat_pos: mat_pos + 6]

        point_pos = mapping[proj_err.id_3d]
        vec[6:] = p[point_pos: point_pos + 3]

        point2d = list_of_corners[proj_err.frame_id].points[proj_err.id_2d]

        partial_derivatives = derivative(vec,
                                         lambda v: _reprojection_error(v, point2d, intrinsic_mat),
                                         np.full_like(vec, 1e-9))

        for i in range(6):
            rows[cur] = row
            cols[cur] = mat_pos + i
            vals[cur] = partial_derivatives[i]
            cur += 1

        for i in range(3):
            rows[cur] = row
            cols[cur] = point_pos + i
            vals[cur] = partial_derivatives[6 + i]
            cur += 1
    print(f"Finished in {time() - start_time} sec")
    return csr_matrix((vals, (rows, cols)), shape=(len(projection_errors), len(p)))


def _run_optimization(projection_errors: List[ProjectionError],
                      list_of_corners: List[FrameCorners],
                      mapping: Dict[int, int],
                      p: np.ndarray,
                      intrinsic_mat: np.ndarray,
                      n_steps: int) -> bool:
    n = 3 * len(list_of_corners)
    lmbd = 2.
    initial_error = _reprojection_errors(projection_errors, list_of_corners, mapping, p, intrinsic_mat).sum()
    print(f'Initial error: {initial_error}')
    if initial_error < 1.:
        print(f'Error is too small for BA')
        return False

    total_steps = 0
    successful_steps = 0
    hit_the_bottom = False
    lmbd_change = 4.
    while successful_steps < n_steps or total_steps < n_steps * 2:
        total_steps += 1
        J = _compute_jacobian(projection_errors, list_of_corners, mapping, p, intrinsic_mat)
        JJ = (J.T @ J).toarray()
        JJ += lmbd * np.diag(np.diag(JJ))
        U = JJ[:n, :n]
        W = JJ[:n, n:]
        V = JJ[n:, n:]
        try:
            V_inv = np.zeros_like(V)
            for i in range(0, len(V), 3):
                s = 3 * i
                t = 3 * i + 3
                V_inv[s:t, s:t] = np.linalg.inv(V[s:t, s:t])
        except np.linalg.LinAlgError:
            hit_the_bottom = True
            lmbd *= lmbd_change
            continue

        g = J.T @ _reprojection_errors(projection_errors, list_of_corners, mapping, p, intrinsic_mat)
        A = U - W @ V_inv @ W.T
        b = W @ V_inv @ g[n:] - g[:n]

        try:
            dc = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            hit_the_bottom = True
            lmbd *= lmbd_change
            continue

        dx = V_inv @ (-g[n:] - W.T @ dc)

        p[:n] += dc
        p[n:] += dx
        error = _reprojection_errors(projection_errors, list_of_corners, mapping, p, intrinsic_mat).sum()
        if error > initial_error:
            p[:n] -= dc
            p[n:] -= dx
            lmbd *= lmbd_change
            hit_the_bottom = True
            print(f"Lambda changed to {lmbd}")
        else:
            initial_error = error
            successful_steps += 1
            if not hit_the_bottom:
                lmbd /= lmbd_change
            print(f"Lambda changed to {lmbd}")
        print(f"Current error {error}")
    print(f'Final error: {initial_error}')
    return True
