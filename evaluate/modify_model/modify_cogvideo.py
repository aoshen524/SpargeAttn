import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models import CogVideoXTransformer3DModel
from spas_sage_attn.autotune import SparseAttentionMeansim, extract_sparse_attention_state_dict, load_sparse_attention_state_dict


class SageAttnCogVideoXAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, idx, ):
        self.idx = idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        sparse_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert attention_mask is None, "Attention mask is not supported"

        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        if self.use_kv_sparse and not self.evaluate_mode:
            # Pruning
            key, value = self.mask_kv(key, value, sparse_table)
        
        # 执行注意力计算
        outputs = attn.inner_attention(query, key, value, is_causal=False, return_sparse_table=self.evaluate_mode, kv_sparse_threshold=self.kv_sparse_threshold)
        hidden_states = outputs["o"]
        if self.use_kv_sparse and self.evaluate_mode:
            sparse_table = outputs["sparse_table"]

        # 调整形状
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        processor_outputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
        }
        if self.use_kv_sparse and self.evaluate_mode:
            processor_outputs["sparse_table"] = sparse_table
        return processor_outputs
    
    def set_sparse_properties(
        self,
        use_kv_sparse: bool = False,
        kv_sparse_threshold: float = 0.1,
        evaluate_mode: bool = False,
    ):
        """
        Set sparse properties for the attention processor.

        Args:
            use_kv_sparse (bool): Whether to use key-value sparsity.
            kv_sparse_threshold (float): Threshold for key-value sparsity.
            evaluate_mode (bool): Whether the model is in evaluation mode.
        """
        self.use_kv_sparse = use_kv_sparse
        self.kv_sparse_threshold = kv_sparse_threshold
        self.evaluate_mode = evaluate_mode

    def mask_kv(self, key: torch.Tensor, value: torch.Tensor, sparse_table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask the key and value tensors based on the sparse table.

        Args:
            key (torch.Tensor): Key tensor of shape [batch_size, num_heads, seq_len, head_size].
            value (torch.Tensor): Value tensor of shape [batch_size, num_heads, seq_len, head_size].
            sparse_table (torch.Tensor): Sparse table of shape [batch_size, seq_len], indicating which keys/values to keep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pruned key and value tensors.
        """
        batch_size, seq_len = sparse_table.shape
        batch_size_k, num_heads, seq_len_k, head_size = key.shape

        # Verify compatible dimensions
        assert batch_size == batch_size_k, "Batch sizes must match"
        assert seq_len == seq_len_k, "Sequence lengths must match"

        # Expand sparse_table to match key/value tensor shape: [batch_size, 1, seq_len, 1]
        expanded_sparse_table = sparse_table.unsqueeze(1).unsqueeze(-1)

        # Broadcast expanded_sparse_table across num_heads and head_size dimensions
        mask = expanded_sparse_table.expand(-1, num_heads, -1, head_size)

        # Apply mask: positions where sparse_table == 0 are set to zero
        masked_key = key * mask
        masked_value = value * mask

        return masked_key, masked_value
        
    def prune_kv(self, key: torch.Tensor, value: torch.Tensor, sparse_table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = sparse_table.shape
        batch_size_k, num_heads, seq_len_k, head_size = key.shape
        
        # Verify compatible dimensions
        assert batch_size == batch_size_k, "Batch sizes must match"
        assert seq_len == seq_len_k, "Sequence lengths must match"
        
        # Get indices where sparse_table is 1 (positions to keep)
        keep_mask = sparse_table.bool()  # [batch_size, seq_len]
        
        # Reshape key and value for gathering
        key_reshaped = key.reshape(batch_size, num_heads, seq_len, head_size)
        value_reshaped = value.reshape(batch_size, num_heads, seq_len, head_size)
        
        # For each batch, gather the positions we want to keep
        # We need to apply the mask separately for each batch
        key_pruned = []
        value_pruned = []
        for b in range(batch_size):
            batch_mask = keep_mask[b]  # [seq_len]
            key_batch = key_reshaped[b, :, :, :][:, batch_mask, :]  # [num_heads, new_seq_len, head_size]
            value_batch = value_reshaped[b, :, :, :][:, batch_mask, :]  # [num_heads, new_seq_len, head_size]
            key_pruned.append(key_batch)
            value_pruned.append(value_batch)
        
        # Stack the results back together
        key = torch.stack(key_pruned)  # [batch_size, num_heads, new_seq_len, head_size]
        value = torch.stack(value_pruned)  # [batch_size, num_heads, new_seq_len, head_size]

        return key, value

def set_spas_sage_attn_cogvideox(
    model: CogVideoXTransformer3DModel,
    verbose=False,
    l1=0.06,
    pv_l1=0.065
):
    for idx, block in enumerate(model.transformer_blocks):
        block.attn1.verbose = verbose
        block.attn1.inner_attention = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1, layer_id=idx)
        origin_processor = block.attn1.get_processor()
        processor = SageAttnCogVideoXAttnProcessor(idx, )
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor
