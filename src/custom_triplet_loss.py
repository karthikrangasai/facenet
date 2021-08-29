# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Implements triplet loss."""

import torch
import torch.nn.functional as F
from typeguard import typechecked
from typing import Optional, Union, Callable


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = torch.min(data, dim=dim, keepdim=True)
    masked_maximums = (
        torch.max(
            torch.multiply(data - axis_minimums, mask), dim=dim, keepdim=True
        )
        + axis_minimums
    )
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = torch.max(data, dim=dim, keepdim=True)
    masked_minimums = (
        torch.min(
            torch.multiply(data - axis_maximums, mask), dim=dim, keepdim=True
        )
        + axis_maximums
    )
    return masked_minimums

def triplet_focal_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    margin: torch.FloatTensor = 0.2,
    sigma: torch.FloatTensor = 0.3,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> torch.Tensor:
    """Computes the triplet focal loss with hard negative and hard positive mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      sigma: Float, sigma term in the loss definition.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_focal_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == torch.float16 or embeddings.dtype == torch.bfloat16
    )
    precise_embeddings = (
        embeddings.type(torch.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])

    elif distance_metric == "squared-L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])
        pdist_matrix = torch.square(pdist_matrix)

    elif distance_metric == "angular":
        pdist_matrix = torch.maximum(
            (1 - F.cosine_similarity(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])),
            0.0
        )

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, torch.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacency)

    adjacency_not = adjacency_not.type(torch.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = labels.size()

    adjacency = adjacency.type(torch.float32)

    mask_positives = adjacency.type(torch.float32) - torch.diag(torch.ones(batch_size))

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    p_hard = torch.exp(torch.divide(hard_positives, sigma))
    n_hard = torch.exp(torch.divide(hard_negatives, sigma))

    if soft:
        triplet_loss = torch.log1p(torch.exp(p_hard - n_hard))
    else:
        triplet_loss = torch.maximum(p_hard - n_hard + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    if convert_to_float32:
        return triplet_loss.type(embeddings.dtype)
    else:
        return triplet_loss



def triplet_batch_hard_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    margin: torch.FloatTensor = 1.0,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> torch.Tensor:
    """Computes the triplet loss with hard negative and hard positive mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      soft: Boolean, if set, use the soft margin version.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_batch_hard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    Returns:
      triplet_loss: float scalar with dtype of y_pred.
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == torch.float16 or embeddings.dtype == torch.bfloat16
    )
    precise_embeddings = (
        embeddings.type(torch.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])

    elif distance_metric == "squared-L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])
        pdist_matrix = torch.square(pdist_matrix)

    elif distance_metric == "angular":
        pdist_matrix = torch.maximum(
            (1 - F.cosine_similarity(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])),
            0.0
        )

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, torch.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacency)

    adjacency_not = adjacency_not.type(torch.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = labels.size()

    adjacency = adjacency.type(torch.float32)

    mask_positives = adjacency.type(torch.float32) - torch.diag(torch.ones(batch_size))

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    if soft:
        triplet_loss = torch.log1p(torch.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = torch.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    if convert_to_float32:
        return triplet_loss.type(embeddings.dtype)
    else:
        return triplet_loss

def assorted_triplet_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    margin: torch.FloatTensor = 1.0,
    focal: bool = False,
    sigma: torch.FloatTensor = 0.3,
    distance_metric: Union[str, Callable] = "L2",
) -> torch.Tensor:
    """Computes assorted triplet loss with hard negative and hard positive mining.
    See https://arxiv.org/pdf/2007.02200.pdf
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      focal: Boolean, if set, use triplet focal loss.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_batch_hard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    Returns:
      triplet_loss: float scalar with dtype of y_pred.
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == torch.float16 or embeddings.dtype == torch.bfloat16
    )
    precise_embeddings = (
        embeddings.type(torch.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])

    elif distance_metric == "squared-L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])
        pdist_matrix = torch.square(pdist_matrix)

    elif distance_metric == "angular":
        pdist_matrix = torch.maximum(
            (1 - F.cosine_similarity(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])),
            0.0
        )

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, torch.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacency)

    adjacency_not = adjacency_not.type(torch.float32)
    # hard negatives: smallest D_an.
    r = torch.FloatTensor(1).uniform_(0,1)
    if r < 0.5:
        hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)
    else:
        hard_negatives = _masked_maximum(pdist_matrix, adjacency_not)

    batch_size = labels.size()

    adjacency = adjacency.type(torch.float32)

    mask_positives = adjacency.type(torch.float32) - torch.diag(torch.ones(batch_size))

    # hard positives: largest D_ap.
    s = torch.FloatTensor(1).uniform_(0,1)
    if s < 0.5:
        hard_positives = _masked_minimum(pdist_matrix, mask_positives)
    else:
        hard_positives = _masked_maximum(pdist_matrix, mask_positives)
    
    if focal:
        hard_positives = torch.exp(torch.divide(hard_positives, sigma))
        hard_negatives = torch.exp(torch.divide(hard_negatives, sigma))

    triplet_loss = torch.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    if convert_to_float32:
        return triplet_loss.type(embeddings.dtype)
    else:
        return triplet_loss

def triplet_batch_hard_v2_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    margin1: torch.FloatTensor = -1.0,
    margin2: torch.FloatTensor = 0.01,
    beta: torch.FloatTensor = 0.002,
    distance_metric: Union[str, Callable] = "L2",
) -> torch.Tensor:
    """Computes the triplet loss with hard negative and hard positive mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin1: Float, margin term in the loss definition.
      margin2: Float, margin term in the loss definition.
      beta: Float, multiplier for intra-class constraint.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_batch_hard_v2_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )

      See https://ieeexplore.ieee.org/document/7780518
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == torch.float16 or embeddings.dtype == torch.bfloat16
    )
    precise_embeddings = (
        embeddings.type(torch.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])

    elif distance_metric == "squared-L2":
        pdist_matrix = F.pairwise_distance(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])
        pdist_matrix = torch.square(pdist_matrix)

    elif distance_metric == "angular":
        pdist_matrix = torch.maximum(
            (1 - F.cosine_similarity(precise_embeddings[:,:,None], precise_embeddings.transpose()[None,:,:])),
            0.0
        )

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, torch.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacency)

    adjacency_not = adjacency_not.type(torch.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = labels.size()

    adjacency = adjacency.type(torch.float32)

    mask_positives = adjacency.type(torch.float32) - torch.diag(torch.ones(batch_size))

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    triplet_loss = torch.maximum(hard_positives - hard_negatives, margin1) + torch.multiply(
                                              torch.maximum(hard_positives, margin2), beta)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    if convert_to_float32:
        return triplet_loss.type(embeddings.dtype)
    else:
        return triplet_loss


class TripletFocalLoss(torch.nn.Module):
    """Computes the triplet loss with hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8558553.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      sigma: Float, sigma term in the loss definition.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_semihard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self, 
        margin: torch.FloatTensor = 1.0, 
        sigma: torch.FloatTensor = 0.3,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None, **kwargs
    ):
        super(TripletFocalLoss, self).__init__(reduction='none')
        self.margin = margin
        self.sigma = sigma
        self.soft = soft
        self.distance_metric = distance_metric
        self.name = name
    
    def forward(self, y_true, y_pred):
        return triplet_focal_loss(y_true, y_pred, margin=self.margin, sigma=self.sigma, soft=self.soft, distance_metric=self.distance_metric)


class TripletBatchHardLoss(torch.nn.Module):
    """Computes the triplet loss with hard negative and hard positive mining.
    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://arxiv.org/pdf/1703.07737.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_semihard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    """

    @typechecked
    def __init__(
        self,
        margin: torch.FloatTensor = 1.0,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs
    ):
        super(TripletBatchHardLoss, self).__init__(reduction='none')
        self.name = name
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric
    
    def forward(self, y_true, y_pred):
        return triplet_batch_hard_loss(y_true, y_pred, margin=self.margin, soft=self.soft, distance_metric=self.distance_metric)


class TripletBatchHardV2Loss(torch.nn.Module):
    """Computes the triplet loss with hard negative and hard positive mining.
    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch. Intra-class variability is enforced through
    a second margin that places a constraint on the spread of the cluster.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://ieeexplore.ieee.org/document/7780518.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_semihard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    """

    @typechecked
    def __init__(
        self,
        margin1: torch.FloatTensor = -1.0,
        margin2: torch.FloatTensor = 0.01,
        beta: torch.FloatTensor = 0.002,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs
    ):
        super(TripletBatchHardV2Loss, self).__init__(reduction='none')
        self.name = name
        self.margin1 = margin1
        self.margin2 = margin2
        self.beta = beta
        self.distance_metric = distance_metric
    
    def forward(self, y_true, y_pred):
        return triplet_batch_hard_v2_loss(y_true, y_pred, margin1=self.margin1, margin2=self.margin2, beta=self.beta, distance_metric=self.distance_metric)


class AssortedTripletLoss(torch.nn.Module):
    """Computes assorted triplet loss with hard negative and hard positive mining.
    See https://arxiv.org/pdf/2007.02200.pdf
    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      focal: Boolean, if set, use triplet focal loss.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          assorted_triplet_loss(batch, labels,
                                                distance_metric=custom_distance
                                    )
      name: Optional name for the op.

    Returns:
      triplet_loss: float scalar with dtype of y_pred.
    """

    @typechecked
    def __init__(
        self, margin: torch.FloatTensor = 1.0, 
        sigma: torch.FloatTensor = 0.3,
        focal: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None, **kwargs
    ):
        super(AssortedTripletLoss, self).__init__(reduction='none')
        self.name = name
        self.margin = margin
        self.sigma = sigma
        self.focal = focal
        self.distance_metric = distance_metric
    
    def forward(self, y_true, y_pred):
        return assorted_triplet_loss(y_true, y_pred, margin=self.margin, focal=self.focal, sigma=self.sigma, distance_metric=self.distance_metric)

class ConstellationLoss(torch.nn.Module):
    '''Computes constellation loss.
    See https://arxiv.org/pdf/1905.10675.pdf for more details.
    Note that the batch is divided into groups of k, so the effective batch size
    for training should be batch_size * k. To make things simpler, we perform an
    internal divison of batch size by k to prevent issues.
    '''
    def __init__(self, k: int = 4, batch_size: int = 128, 
                 name: str = 'ConstellationLoss'):
        super(ConstellationLoss, self).__init__(reduction='none')
        self.k = k
        self.BATCH_SIZE = batch_size // k

    def _get_triplet_mask(self, labels: torch.Tensor):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """

        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size()[0]).to(dtype=torch.bool)
        indices_not_equal = torch.logical_not(indices_equal)
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

        # Combine the two masks
        mask = torch.logical_and(distinct_indices, valid_labels)

        return mask

    def forward(self, labels, embeddings):
        """Build the constellation loss over a batch of embeddings.
                Args:
                    labels: labels of the batch, of size (batch_size,)
                    embeddings: tensor of shape (batch_size, embed_dim)

                Returns:
                    ctl_loss: scalar tensor containing the constellation loss

                @TODO: Try to optimize the code wherever possible to speed up performance
                """

        labels_list = []
        embeddings_list = []
        for i in range(self.k):
            labels_list.append(labels[self.BATCH_SIZE * i:self.BATCH_SIZE * (i + 1)])
            embeddings_list.append(embeddings[self.BATCH_SIZE * i:self.BATCH_SIZE * (i + 1)])

        loss_list = []
        for i in range(len(embeddings_list)):
            # Get the dot product
            pairwise_dist = torch.matmul(embeddings_list[i], torch.transpose(embeddings_list[i]))

            # shape (batch_size, batch_size, 1)
            anchor_positive_dist = pairwise_dist.unsqueeze(2)
            assert anchor_positive_dist.size()[2] == 1, "{}".format(anchor_positive_dist.size())
            # shape (batch_size, 1, batch_size)
            anchor_negative_dist = pairwise_dist.unsqueeze(1)
            assert anchor_negative_dist.size()[1] == 1, "{}".format(anchor_negative_dist.size())

            ctl_loss = anchor_negative_dist - anchor_positive_dist

            # (where label(a) != label(p) or label(n) == label(a) or a == p)
            mask = self._get_triplet_mask(labels_list[i])
            mask = mask.to(dtype=torch.float32)
            ctl_loss = torch.multiply(mask, ctl_loss)

            loss_list.append(ctl_loss)

        ctl_loss = 1. + torch.exp(loss_list[0])
        for i in range(1, len(embeddings_list)):
            ctl_loss += torch.exp(loss_list[i])

        ctl_loss = torch.log(ctl_loss)

        # # Get final mean constellation loss and divide due to very large loss value
        ctl_loss = ctl_loss.sum() / 1000.

        return ctl_loss
