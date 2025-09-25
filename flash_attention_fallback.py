"""
Flash Attention CPU Fallback
============================

This module provides a CPU-compatible fallback for Flash Attention
when CUDA compilation is not available.

Usage:
    Replace flash_attn imports with this module for basic functionality.
"""

import torch
import torch.nn.functional as F
import warnings


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, 
                   window_size=(-1, -1), alibi_slopes=None, deterministic=False):
    """
    CPU fallback implementation of Flash Attention
    
    Args:
        q, k, v: Query, Key, Value tensors of shape (batch, seqlen, nheads, headdim)
        dropout_p: Dropout probability (ignored in this CPU implementation)
        softmax_scale: Scale factor for attention scores
        causal: Whether to apply causal masking
        window_size: Attention window size (ignored in this implementation)
        alibi_slopes: ALiBi slopes (not implemented)
        deterministic: Whether to use deterministic operations
        
    Returns:
        Attention output of same shape as q
    """
    warnings.warn(
        "Using CPU fallback for Flash Attention. Performance will be slower than CUDA implementation.",
        UserWarning
    )
    
    # Get dimensions
    batch, seqlen, nheads, headdim = q.shape
    
    # Compute scale
    if softmax_scale is None:
        softmax_scale = headdim ** -0.5
    
    # Reshape for batched matrix multiply: (batch * nheads, seqlen, headdim)
    q = q.transpose(1, 2).reshape(batch * nheads, seqlen, headdim)
    k = k.transpose(1, 2).reshape(batch * nheads, seqlen, headdim)
    v = v.transpose(1, 2).reshape(batch * nheads, seqlen, headdim)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    out = torch.matmul(attn_weights, v)
    
    # Reshape back to original format
    out = out.reshape(batch, nheads, seqlen, headdim).transpose(1, 2)
    
    return out


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                          dropout_p=0.0, softmax_scale=None, causal=False, 
                          window_size=(-1, -1), alibi_slopes=None, deterministic=False):
    """
    Variable length Flash Attention fallback (simplified implementation)
    
    Note: This is a basic implementation that doesn't handle variable lengths optimally.
    For production use, consider implementing proper variable-length handling.
    """
    warnings.warn(
        "Variable-length Flash Attention fallback not fully implemented. Using basic implementation.",
        UserWarning
    )
    
    # For now, just use the regular implementation
    # In a full implementation, you'd handle the variable sequence lengths properly
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, 
                          window_size, alibi_slopes, deterministic)


# Compatibility layer
class FlashAttention(torch.nn.Module):
    """PyTorch module wrapper for Flash Attention fallback"""
    
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        
    def forward(self, q, k, v, causal=False):
        return flash_attn_func(q, k, v, 
                              dropout_p=self.dropout_p, 
                              softmax_scale=self.softmax_scale, 
                              causal=causal)


# Print warning when module is imported
print("⚠️  Using Flash Attention CPU fallback. Install CUDA version for better performance.")
