#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli

maxCorners = 1000
minDistance = 6
max_diff = 0.2


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    frame_sequence = list(map(lambda t: (np.array(t) * 255.0).astype(np.uint8), frame_sequence))
    prev = frame_sequence[0]
    points = cv2.goodFeaturesToTrack(prev,
                                     maxCorners=maxCorners,
                                     qualityLevel=0.01,
                                     minDistance=minDistance,
                                     blockSize=5).squeeze(axis=1)
    ptr = len(points)
    ids = np.arange(ptr)
    sizes = np.full(ptr, 10)
    builder.set_corners_at_frame(0, FrameCorners(ids, points, sizes))
    idx = 0
    for cur in frame_sequence[1:]:
        idx += 1
        fwd = cv2.calcOpticalFlowPyrLK(prev, cur,
                                       points, None,
                                       winSize=(15, 15), maxLevel=2,
                                       criteria=(cv2.TERM_CRITERIA_EPS
                                                 | cv2.TERM_CRITERIA_COUNT, 10, 0.03))[0].squeeze()
        bwd = cv2.calcOpticalFlowPyrLK(cur, prev,
                                       fwd, None,
                                       winSize=(15, 15), maxLevel=2,
                                       criteria=(cv2.TERM_CRITERIA_EPS
                                                 | cv2.TERM_CRITERIA_COUNT, 10, 0.03))[0].squeeze()
        mask = np.abs(points - bwd).max(-1) < max_diff
        ids = ids[mask]
        points = fwd[mask]
        sizes = sizes[mask]
        pts = points
        if len(pts) < maxCorners:
            next_features = cv2.goodFeaturesToTrack(cur,
                                                    mask=get_mask(pts, cur),
                                                    maxCorners=maxCorners,
                                                    qualityLevel=0.05,
                                                    minDistance=minDistance,
                                                    blockSize=7)
            next_features = next_features.squeeze(axis=1) if next_features is not None else []
            for pnt in next_features[:maxCorners - len(pts)]:
                ids = np.concatenate([ids, [ptr]])
                points = np.concatenate([points, [pnt]])
                sizes = np.concatenate([sizes, [10]])
                ptr += 1

        builder.set_corners_at_frame(idx, FrameCorners(ids, points, sizes))
        prev = cur


def get_mask(points_, shape):
    mask = np.ones_like(shape, dtype=np.uint8)
    for x, y in points_:
        cv2.circle(mask, (x, y), minDistance, 0, -1)
    return mask * 255


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter