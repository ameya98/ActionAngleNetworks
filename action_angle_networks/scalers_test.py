# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for scalers."""

from typing import Sequence

import numpy as np

from absl.testing import absltest, parameterized

from action_angle_networks import scalers


class TrainTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            "arr": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            "arr_scaled": [[-1, -1, -1], [1, 1, 1]],
            "mean": [2.0, 3.0, 4.0],
            "std": [1.0, 1.0, 1.0],
        },
        {
            "arr": [[11.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            "arr_scaled": [[1, -1, -1], [-1, 1, 1]],
            "mean": [7.0, 3.0, 4.0],
            "std": [4.0, 1.0, 1.0],
        },
    )
    def test_standard_scaler(
        self,
        arr: Sequence[Sequence[float]],
        arr_scaled: Sequence[Sequence[float]],
        mean: Sequence[float],
        std: Sequence[float],
    ):
        arr = np.asarray(arr)
        arr_scaled = np.asarray(arr_scaled)

        scaler = scalers.StandardScaler()
        scaler = scaler.fit(arr)

        self.assertTrue(np.allclose(scaler.transform(arr), arr_scaled))
        self.assertTrue(np.allclose(scaler.inverse_transform(arr_scaled), arr))

        self.assertTrue(np.allclose(scaler.mean(), mean))
        self.assertTrue(np.allclose(scaler.std(), std))

    @parameterized.parameters(
        {
            "arr": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            "arr_scaled": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        },
        {
            "arr": [[11.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            "arr_scaled": [[11.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        },
    )
    def test_identity_scaler(
        self, arr: Sequence[Sequence[float]], arr_scaled: Sequence[Sequence[float]]
    ):
        arr = np.asarray(arr)
        arr_scaled = np.asarray(arr_scaled)

        scaler = scalers.IdentityScaler()
        scaler = scaler.fit(arr)

        self.assertTrue(np.allclose(scaler.transform(arr), arr_scaled))
        self.assertTrue(np.allclose(scaler.inverse_transform(arr_scaled), arr))


if __name__ == "__main__":
    absltest.main()
