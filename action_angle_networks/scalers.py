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

"""Implements scalers for data pre-processing in a JAX-compatible manner."""

import abc
from typing import Optional, Tuple

import chex
import flax
import jax.numpy as jnp


class Scaler(flax.struct.PyTreeNode, abc.ABC):
    """Abstract base class for scalers."""

    @abc.abstractmethod
    def fit(self, data: chex.Array) -> "Scaler":
        pass

    @abc.abstractmethod
    def transform(self, data: chex.Array) -> chex.Array:
        pass

    @abc.abstractmethod
    def inverse_transform(self, data: chex.Array) -> chex.Array:
        pass


class IdentityScaler(Scaler):
    """Implements the identity scaler."""

    def fit(self, data: chex.Array) -> "IdentityScaler":
        del data
        return IdentityScaler()

    def transform(self, data: chex.Array) -> chex.Array:
        return data

    def inverse_transform(self, data: chex.Array) -> chex.Array:
        return data


class StandardScaler(Scaler):
    """Implements sklearn.preprocessing.StandardScaler."""

    _mean: Optional[chex.Array] = None
    _std: Optional[chex.Array] = None

    def fit(self, data: chex.Array) -> "StandardScaler":
        mean = jnp.mean(data, axis=0, keepdims=True)
        std = jnp.std(data, axis=0, keepdims=True)
        return StandardScaler(mean, std)

    def transform(self, data: chex.Array) -> chex.Array:
        return (data - self.mean()) / self.std()

    def inverse_transform(self, data: chex.Array) -> chex.Array:
        return data * self.std() + self.mean()

    def mean(self) -> chex.Array:
        return self._mean

    def std(self) -> chex.Array:
        return self._std
