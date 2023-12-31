# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT


"""
Vector Quantization autograd functions
From https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""

import torch
from torch.autograd import Function


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)  # 128
            inputs_size = inputs.size() # [1, 2048, 128]
            inputs_flatten = inputs.view(-1, embedding_size)  # [2048, 128]

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            # euclidean_distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)  # size(2048, 8)
            # _, indices_flatten = torch.min(euclidean_distances, dim=1)  # size(2048)
            
            
            cosine_distances = inputs_flatten @ codebook.t()  # size(2048, 8)
            _, indices_flatten = torch.max(cosine_distances, dim=1)  # size(2048)

            indices = indices_flatten.view(*inputs_size[:-1]) # size(1, 2048)
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vector_quantization(inputs, codebook) # size(1, 2048)
        indices_flatten = indices.view(-1) # size(2048)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous().view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vector_quantization = VectorQuantization.apply
vector_quantization_st = VectorQuantizationStraightThrough.apply
__all__ = [vector_quantization, vector_quantization_st]
