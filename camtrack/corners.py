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
    dist = 13
    params = dict(maxCorners=400,
                  qualityLevel=0.01,
                  minDistance=dist,
                  useHarrisDetector=False,
                  blockSize=dist)
    prev = (frame_sequence[0] * 255).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(image=prev, **params)
    ids = np.arange(len(pts))
    builder.set_corners_at_frame(0, FrameCorners(ids, pts, np.full(len(pts), dist)))
    for frame, cur in enumerate(frame_sequence[1:]):
        cur = (cur * 255).astype(np.uint8)
        pts, status, err = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(dist, dist))
        pts = pts[status.reshape(-1).astype(np.bool)]
        ids = ids[status.reshape(-1).astype(np.bool)]
        if len(pts) < 400:
            mask = get_mask(cur, pts, dist)
            params['maxCorners'] = 400 - len(pts)
            pts = np.append(pts, cv2.goodFeaturesToTrack(cur, mask=mask, **params)).reshape((-1, 1, 2))
            ids = np.arange(len(pts))
        builder.set_corners_at_frame(frame, FrameCorners(ids, pts, np.full(len(pts), dist)))
        prev = cur


def get_mask(cur, pts, dist):
    mask = np.full_like(cur, 255)
    for x, y in pts.reshape(-1, 2):
        cv2.circle(mask, (x, y), dist, 0, -1)
    return mask


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
