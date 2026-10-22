# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hysteresis for a binary gripper driven by a continuous policy output.

The pi0.5 xArm6 finetunes are trained on binary gripper targets but predict blurry values (often
0.4-0.6 while approaching). Thresholding each step at 0.5 closed the gripper in the first few
steps of an episode; the policy then saw a closed gripper in its state, took itself to be
carrying the cube, and headed for the box without ever reaching for it. This filter changes the
gripper state only after the prediction has clearly crossed a threshold for several consecutive
steps, and holds it otherwise.
"""


class GripperHysteresis:
    def __init__(self, close_at=0.7, open_at=0.3, consecutive=3, closed=False):
        assert open_at < close_at
        self.close_at, self.open_at, self.consecutive = close_at, open_at, consecutive
        self.closed, self._streak = closed, 0

    def reset(self, closed=False):
        self.closed, self._streak = closed, 0

    def __call__(self, value):
        """Feed one raw gripper prediction (0 = open, 1 = closed); returns the command, 0.0 or 1.0."""
        crossing = value <= self.open_at if self.closed else value >= self.close_at
        self._streak = self._streak + 1 if crossing else 0
        if self._streak >= self.consecutive:
            self.closed, self._streak = not self.closed, 0
        return 1.0 if self.closed else 0.0
