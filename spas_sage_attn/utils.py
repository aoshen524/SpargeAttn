import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import triton
import triton.language as tl
from torch import Tensor


def precision_metric(quant_o, fa2_o, verbose=True, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}

def hyperparameter_check(hyper, H, device):
    if type(hyper) == float or type(hyper) == int:
        hyper = torch.full((H,), float(hyper), device=device)
    elif isinstance(hyper, Tensor):
        assert len(hyper.shape) <= 1, "Hyperparameter tensor must be 1D"
        if len(hyper.shape) == 0:
            hyper = torch.full((H,), hyper.item(), device=device)
        assert hyper.numel() == H, f"Hyperparameter tensor must have {H} elements, but has {hyper.numel()}"
        hyper = hyper.to(device)
    else:
        print(hyper)
        raise ValueError("Hyperparameter must be a float or a tensor")
    return hyper



@triton.jit
def triton_block_map_to_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, num_block_k):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    valid_block_num = 0

    map_ptr = map_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    lut_ptr = lut_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    valid_block_num_ptr = valid_block_num_ptr + b * H * Q + h * Q + q
    
    valid_block_num = 0
    prev_block = 0

    for i in range(num_block_k):
        cur_block = tl.load(map_ptr + i)
        if cur_block:
            tl.store(lut_ptr + valid_block_num, i - prev_block)
            valid_block_num += 1
            prev_block = i

    tl.store(valid_block_num_ptr, valid_block_num)

def block_map_lut_triton(block_map):
    assert block_map.dim() == 4
    assert block_map.is_contiguous()

    # block_map 的形状是 [2, 30, 139, 278]
    B, H, Q, K = block_map.shape
    lut = torch.zeros((B, H, Q, K), dtype=torch.int32, device=block_map.device)
    # 记录每个 q 块（[b, h, q]）有多少个 k 块需要计算
    valid_block_num = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)

    grid = (B, H, Q)
    triton_block_map_to_lut_kernel[grid](block_map, lut, valid_block_num, K)

    return lut, valid_block_num

@triton.jit
def triton_bmm_pool_sim_simmean_fuse_quant(
    x_ptr,
    xm_ptr,
    pool_ptr,
    sim_ptr,
    x_quant_ptr,
    scale_ptr,
    simthreshd1,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    fuse_mean: tl.constexpr
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    if fuse_mean:
        xm_ptrs = xm_ptr + b * H * D + h * D + tl.arange(0, D)
        x_mean = tl.load(xm_ptrs)
        x -= x_mean
        x = tl.where(xmask, x, 0)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)

    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)

    scale = tl.max(tl.abs(x_fp32)) / 127.
    scale += 0.0000001
    x_int8 = x_fp32 / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    x_quant_ptrs = x_quant_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    scale_ptrs = scale_ptr + b * H * NB + h * NB + nb
    tl.store(x_quant_ptrs, x_int8, mask = xmask)
    tl.store(scale_ptrs, scale)

@triton.jit
def triton_bmm_pool_sim_simmean(x_ptr, pool_ptr, sim_ptr, simthreshd1, N: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)
    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)
    
    
def get_pool_sim_triton_simmean(x, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    # Launch kernel
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks

# x_mean：x 的均值（可选），形状是 [B, H, 1, D]
def get_pool_sim_triton_simmean_fuse_quant(x, x_mean, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    # 记录每个块是否“重要”（布尔值，True 表示重要）
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    x_quant = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    x_scale = torch.empty((B, H, nblock), device=x.device, dtype=torch.float32)
    grid = (B, H, nblock)
    triton_bmm_pool_sim_simmean_fuse_quant[grid](x, x_mean, pool, sim_blocks, x_quant, x_scale, simthreshd1, N=N, D=D, BS=block_size, fuse_mean=(True if x_mean is not None else False))
    return pool, sim_blocks, x_quant, x_scale

@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)
    

def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map

@triton.jit
def triton_fill_causal_mask(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)

def fill_causal_mask_triton(mask, BqdivBk:float):
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


def get_block_map_meansim(q, k, is_causal=False, BLKQ=128, BLKK=64, simthreshd1=0.1, cdfthreshd=0.9, is_sparse=True, return_lut=False, attention_sink=False):
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
    cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
    num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    if not return_lut:
        return final_map
    else:
        lut, valid_block_num = block_map_lut_triton(final_map)
        return lut, valid_block_num

def get_block_map_meansim_fuse_quant(q, k, km=None, is_causal=False, BLKQ=128, BLKK=64, simthreshd1=0.1, cdfthreshd=0.9, is_sparse=True, return_lut=False, attention_sink=False):
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    # Q和K的分块大小是不一样的，沿着长度进行分块
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks, q_int8, q_scale = get_pool_sim_triton_simmean_fuse_quant(q, None, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks, k_int8, k_scale = get_pool_sim_triton_simmean_fuse_quant(k, km, BLKK, simthreshd1)

    # sim_kblocks：形状从 [B, H, nk] 扩展到 [B, H, nq, nk]，例如 [2, 30, 141, 282]。
    # unsqueeze(-2)：在倒数第二个维度插入一个维度，变成 [B, H, 1, nk]。
    # expand(-1, -1, nq, -1)：将第 2 个维度扩展到 nq

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    #     sim_qblocks：形状从 [B, H, nq] 扩展到 [B, H, nq, nk]，例如 [2, 30, 141, 282]。
    # unsqueeze(-1)：在最后一个维度插入一个维度，变成 [B, H, nq, 1]。
    # expand(-1, -1, -1, nk)：将最后一个维度扩展到 nk。
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    # 计算块级别的注意力分数
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    #  [B, H, nq, nk]
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
    cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
#     在 cdf 中查找 cdfthreshd_ts 的位置。
# 例如，cdfthreshd = 0.9，表示选择累积和达到 90% 的块。
# 返回的 num_to_select 形状是 [B, H, Q]，例如 [2, 30, 141]，表示每个 q 块需要保留的 k 块数量。
    num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    if not return_lut:
        return final_map, q_int8, q_scale, k_int8, k_scale
    else:
        lut, valid_block_num = block_map_lut_triton(final_map)
        return lut, valid_block_num, q_int8, q_scale, k_int8, k_scale


def block_map_to_mask(block_map, BLKQ=128, BLKK=64):
    B, H, x, y = block_map.shape

    expanded_mask = torch.zeros((B, H, x * BLKQ, y * BLKK), dtype=torch.bool, device=block_map.device)
    for i in range(x):
        for j in range(y):
            expanded_mask[..., i * BLKQ: (i + 1) * BLKQ, j * BLKK: (j + 1) * BLKK] = block_map[..., i:i+1, j:j+1]

    return expanded_mask

def block_map_lut(block_map):
    valid_entry_num = block_map.to(torch.int32).sum(dim=-1)

    B, H, x, y = block_map.shape

    one_matrix = torch.ones((B, H, x, y), dtype=torch.int32, device=block_map.device)
    cum_matrix = torch.cumsum(one_matrix, dim=-1)
    masked_cum_matrix = cum_matrix * block_map.to(torch.int32)
    filled_matrix = masked_cum_matrix.clone()
    filled_matrix[~block_map] = 10000000
    lut = torch.sort(filled_matrix, dim=-1)[0] - 1 # make index start from 0
    lut[:, :, :, 1:] = lut[:, :, :, 1:] - lut[:, :, :, :-1]

    return lut.to(torch.int32), valid_entry_num.to(torch.int32)


def get_attn_qk_sparse_table_int8_quant(Q, K, scale, kv_sparse_threshold):
    batchsize, num_heads, seq_length, head_size = Q.shape
    device = Q.device
    dtype = torch.float32

    # Initialize sparse_table with float32 to store intermediate results
    sparse_table = torch.zeros(batchsize, num_heads, seq_length, device=device, dtype=dtype)

    grid = (batchsize, num_heads, seq_length)
    BLOCK_SIZE_HEAD = triton.next_power_of_2(head_size)

    attn_qk_int8_kernel[grid](
        Q, K, sparse_table,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        sparse_table.stride(0), sparse_table.stride(1),
        seq_length, head_size,
        scale,
        BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD
    )
    
    # Take mean across heads
    sparse_table = sparse_table.mean(dim=1)
    
    # 初始化二值表，默认所有位置为 1（代表非零）
    sparse_binary_table = torch.ones(batchsize, seq_length, device=device, dtype=torch.bfloat16)
    
    # 对每个样本对 seq_length 上的值进行升序排序，sorted_indices 保存排序后的索引
    _, sorted_indices = torch.sort(sparse_table, dim=1, descending=False)
    sparsity = 0.01
    num_zero = max(1, int(seq_length * sparsity))
    # 利用高级索引，将每个样本中 sorted_indices 前 num_zero 的位置置 0
    row_indices = torch.arange(batchsize, device=device).unsqueeze(1).expand(-1, num_zero)
    sparse_binary_table[row_indices, sorted_indices[:, :num_zero]] = 0
    
    return sparse_binary_table


@triton.jit
def attn_qk_int8_kernel(
    Q_ptr, K_ptr, sparse_table_ptr,
    batch_stride_q, head_stride_q, seq_stride_q,
    batch_stride_k, head_stride_k, seq_stride_k,
    sparse_batch_stride, sparse_head_stride,
    seq_length, head_size,
    scale,
    BLOCK_SIZE_HEAD: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_id = tl.program_id(2)

    head_offsets = tl.arange(0, BLOCK_SIZE_HEAD)
    q_offset = batch_id * batch_stride_q + head_id * head_stride_q + seq_id * seq_stride_q + head_offsets

    q_int8 = tl.load(Q_ptr + q_offset, mask=head_offsets < head_size, other=0)
    q = q_int8.to(tl.float32) * scale

    attn_max = float('-inf')

    for i in range(seq_length):
        k_offset = batch_id * batch_stride_k + head_id * head_stride_k + i * seq_stride_k + head_offsets
        k_int8 = tl.load(K_ptr + k_offset, mask=head_offsets < head_size, other=0)
        k = k_int8.to(tl.float32) * scale

        attn_score = tl.sum(q * k)
        attn_max = tl.maximum(attn_max, attn_score)

    # recompute for numerical stability softmax
    exp_sum = 0.0
    for i in range(seq_length):
        k_offset = batch_id * batch_stride_k + head_id * head_stride_k + i * seq_stride_k + head_offsets
        k_int8 = tl.load(K_ptr + k_offset, mask=head_offsets < head_size, other=0)
        k = k_int8.to(tl.float32) * scale

        attn_score = tl.sum(q * k)
        exp_sum += tl.exp(attn_score - attn_max)

    # Store the intermediate result
    sparse_offset = batch_id * sparse_batch_stride + head_id * sparse_head_stride + seq_id
    tl.store(sparse_table_ptr + sparse_offset, exp_sum)