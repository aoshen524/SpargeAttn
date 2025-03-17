/*
 * Copyright (c) 2025 by SpargAttn team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "../cp_async.cuh"
#include "../mma.cuh"
#include "../permuted_smem.cuh"
#include "../math.cuh"
#include "../dispatch_utils.h"
#include "../reduction_utils.cuh"

#include "attn_utils.cuh"

// 这些定义可能与硬件加速（如Tensor Core）的内存对齐和计算单元打包有关，决定了数据如何被高效加载和处理
#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 16  // fp8
#define PACK_SIZE_O 8   // fp16
// 定义了Q和K之间矩阵乘法的维度（基于Tensor Core的int8计算）：
// MMA_QK_M: 输出矩阵的行数（16）。

// MMA_QK_N: 输出矩阵的列数（16）。

// MMA_QK_K: 共享维度（32），即Q的列数和K的行数。

// 作用: 这指定了Tensor Core在计算Q和K的点积时的块大小。

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp8 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32
// mask_mode: 掩码模式（如因果掩码，用于自回归模型），默认无掩码。

// fuse_v_scale: 是否将V的缩放操作融合到计算中，以减少访存
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, PVThresholdMode pv_threashold_mode, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool fuse_v_scale=false, bool return_pv_count = false>
__global__ void qk_int_sv_f8_block_sparse_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, DTypeOut *__restrict__ O, int32_t *__restrict__ PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
                      // Q、K、V的量化缩放因子，用于将int8/fp8转换回浮点数。
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q, 
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4, "DTypeQK must be int8 or int4");
  // Q_GRAN和K_GRAN是Q和K量化的“粒度”，只能是“每个线程块（block）”、“每个线程束（warp）”或者“每个线程（thread）”之一。
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
//   意思：DTypeSVAccum是计算S和V（Softmax后的权重和值）时的累加器类型，必须是float（32位浮点数），half（16位浮点数）还在开发中（WIP = Work In Progress）。  
// 为什么？float精度更高，避免累加时的误差，half虽然省内存但还没调好。
  static_assert(std::is_same<DTypeSVAccum, float>::value, "DTypeSVAccum must be float, half is WIP");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(CTA_K % 64 == 0);
//   第一步：CTA_Q 和 CTA_K 是什么？
// 在 CUDA 中，"CTA" 表示协作线程数组（Cooperative Thread Array），也就是一个线程块。这里：
// CTA_Q：线程块处理的查询（Query，简称 Q）元素数量。

// CTA_K：线程块处理的关键（Key，简称 K）元素数量
// 注意力计算
// 这个内核实现的是块稀疏注意力机制：
// 查询-关键点积：使用 int8 整数运算（借助 Tensor Core）计算查询 (Q) 和关键 (K) 的分数。
// cpp

// compute_int_qk<...>(smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

// RS 保存了一块 Q 和 K 的结果分数。

// 因果掩码：如果启用了因果性，应用掩码阻止关注未来的关键。

// Softmax 和值计算：归一化分数并用值 (V) 计算输出。

// 数据按瓦片（tile）处理：
// num_tiles_q = WARP_Q / MMA_QK_M = 32 / 16 = 2（每个线程束的查询瓦片数）。

// num_tiles_k = WARP_K / MMA_QK_N = 64 / 16 = 4（每个线程束的关键瓦片数）。

// 每个线程块有：
// num_warps_q = CTA_Q / WARP_Q = 128 / 32 = 4 个线程束处理查询。

// num_warps_k = CTA_K / WARP_K = 64 / 64 = 1 个线程束处理关键。

// 总线程束数 = num_warps_q * num_warps_k = 4。

// 因果掩码的效率
// 在因果注意力中，位置 i 的查询只能关注位置 j <= i 的关键。内核一次处理 CTA_Q 个查询和 CTA_K 个关键，掩码按块应用。索引计算如下：
// cpp

// uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<...>() * WARP_Q + lane_id / 4;
// uint32_t K_idx_lane_base = get_warp_idx_k<...>() * WARP_K + 2 * (lane_id % 4);

// bx 是查询维度的块索引。

// Q_idx_lane_base 和 K_idx_lane_base 确定查询和关键块的起始位置。

// 对于块索引 bx，查询范围是 bx * CTA_Q 到 (bx + 1) * CTA_Q - 1（如 bx = 0 时为 0–127）。关键由 Lut 定义的稀疏集合提供，通常跨 CTA_K 个元素（第一次迭代可能是 0–63）。
// 在因果场景中：
// 早期的查询（0–63）只能看到它们位置之前的键。

// 后期的查询（64–127）可以看到更多键，但不超过自己的位置。

// 如果 CTA_Q 相对于 CTA_K 太大，块内的查询可能需要关注序列中更早的关键，而当前 CTA_K 并未覆盖这些关键。这可能导致低效的内存访问或需要更多迭代来获取早期关键，尤其是在稀疏注意力中，Lut 决定了哪些关键块是相关的。
// 为什么是 2？
// 当 CTA_Q / CTA_K = 2：
// 查询 0–127 与关键 0–63 配对。

// 对于因果性，查询 64–127 不应看到超过 127 的关键，但可以关注它们自己的位置。

// 如果 CTA_Q / CTA_K > 2（如 CTA_Q = 192，CTA_K = 64，比例 = 3）：
// 查询 0–191 与关键 0–63 配对。

// 查询 128–191 需要关键直到 191，但只加载了 0–63，可能错过 64–191，除非 Lut 通过更多迭代补偿。

// 比例 ≤ 2 确保查询块不会比关键块超前太多，使因果掩码应用保持高效。它平衡了工作负载，使得：
// 线程块的关键范围与查询的因果需求合理对齐。

// 共享内存使用（smem_Q 和 smem_K）和线程束并行性保持优化。

// 第四步：效率影响
// 共享内存：内核为 Q、K 和 V 使用共享内存：
// cpp

// size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));

// 更大的 CTA_Q 增加内存压力，但保持 ≤ 2 * CTA_K 确保可控。

// 线程束利用率：4 个线程束（128 / 32）处理查询，每个 32 个；1 个线程束处理 64 个关键，比例支持 Tensor Core 的高效 MMA 操作。

// 因果掩码开销：小的比例减少了被掩码掉的无效计算，因为关键块与查询的因果需求紧密匹配。
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation
// WARP_Q 是每个线程束处理的查询元素数
/// TODO: 这里是在做warp specialization吗？不是的，只是说一个warp要完成多少的Q的任务，多少的K的任务，然后排列组合一下，得到总的warp数量
  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
// 每个线程束要处理的小矩阵乘法，再分成小块（tile），交给Tensor Core算矩阵乘法，看看要多少次tensor core的使用
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  // 如果是int4：瓦片数 = head_dim ÷ 2 ÷ 32（因为int4比int8数据量少一半），目前理解是tensorcore有块存储，int4更小，所以存的更多，需要更少的瓦片数
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == DataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;
// 直白说：这些是内存布局的“跨度”，告诉GPU怎么读写共享内存
// QK_SMEM_STRIDE: Q和K在共享内存中的步幅（一行多长）
  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == DataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  //                       for fp16: head_dim
  constexpr uint32_t V_SMEM_STRIDE = CTA_K;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  float pv_threshold;
  int pv_count = 0;
  // PV不是肯定要稀疏的吗，然后一个head定一个threshold
  if constexpr (pv_threashold_mode != PVThresholdMode::kNone)
  {
    pv_threshold = PV_Threshold[head_id];
  }

  // RS holds the fragment of S
//   最后的 [8] 和 [2]：
// 这是 Tensor Core 的输出分片大小，8 表示每次 MMA（矩阵乘法累加）操作输出 8 个元素（因为用的是 16×16×32 的矩阵块）。
// 2 是因为每个瓦片分成 2 个子块处理（与线程束的 32 个线程分配有关）
  // 存Softmax前的注意力分数（Q·K^T的片段），维度是[Q瓦片数][K瓦片数][8]，用int32_t存。
  int32_t RS[num_tiles_q][num_tiles_k][8];
  // 存输出O的中间结果（S·V的片段），维度是[Q瓦片数][V瓦片数][8]，类型是DTypeSVAccum（默认float）
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d初始化RO（输出中间结果）：
// 如果DTypeSVAccum是float：每个元素设为0.0。

// 如果是half：用int32_t清零（因为half占16位，两个half塞进一个int32_t，所以循环到4）。

// 初始化m和d（Softmax辅助数组）：
// m设为一个很小的值（-5000000.0），表示当前最大值初始为“负无穷”。

// d设为1.0，表示分母初始为1
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {        
          RO[fq][fv][k] = 0.0f;
        }
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((int32_t*)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }
  // 共享内存是一个线程块内共用的“工作台”，Q、K、V的数据要分开放。

  // Q从头开始（偏移0），K紧跟在Q后面，V再跟在K后面，靠偏移量分开。
  
  // 直白说：就像在桌子上划区域，Q放左边，K放中间，V放右边，CTA_Q和CTA_K是每块区域的宽度
  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  //  通过打乱数据存放顺序（比如按32字节、64字节、128字节为单位重新排列），减少线程间的“银行冲突”（bank conflict）
  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  //                                             for fp16: 32
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);

  // smem_O：输出的共享内存，从smem开头（可能后面覆盖Q）。
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  // global_to_shared_line_lanes_*：每行数据分给几个线程（lane）去加载。

  // global_to_shared_copy_lines_per_warp_*：每个线程束（warp）负责加载几行。
  
  // 根据步幅大小调整：
  // QK：步幅32时2线程/行，16行/warp；64时4线程/行，8行/warp；其他8线程/行，4行/warp。
  
  // V：步幅64时4线程/行，8行/warp；其他8线程/行，4行/warp。
  
  // O：步幅32时4线程/行，8行/warp；其他8线程/行，4行/warp。
  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  //                                                         for fp16: 32
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 64) ? 4 : 8;
  //                                                                  for fp16: 32
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  // 计算加载Q、K、V、O到共享内存时，行和列各需要几次迭代：
  // QK_smem_iters_row：Q和K每行迭代次数 = 步幅 ÷ (线程数 × 打包大小)。
  
  // Q_smem_iters_col：Q的列迭代次数 = Q总行数 ÷ (线程束数 × 每束行数)。
  
  // K_smem_iters_col：K的列迭代次数 = K总行数 ÷ (线程束数 × 每束行数)。
  
  // V_smem_iters_row和V_smem_iters_col：类似，基于V的步幅和head_dim。
  
  // O_smem_iters_row和O_smem_iters_col：基于O的步幅和CTA_Q。
  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  //                          for fp16: CTA_K
  constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  //                                                                for fp16: CTA_K / num_warps * warp_id * stride_seq_v + lane_id / global_to_shared_line_lanes_V * stride_seq_v
  int8_t *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + head_dim / num_warps * warp_id * stride_d_v + lane_id / global_to_shared_line_lanes_V * stride_d_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  // 告诉搬运工把货放共享内存的哪个位置，位置是“打乱”过的（Swizzle），避免挤一块
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);
// 计算Q、K、V在共享内存中给矩阵乘法（MMA，Tensor Core用）的偏移，基于线程束和线程编号。
  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  // for fp 16:
  // uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K / PACK_SIZE_V + (lane_id / 8) % 2);

  // for causal masking
  // 记录Q和K的“门牌号”，方便后面判断哪些不能算（因果掩码），以及从哪开始搬数据。
//   使用场景
// 在因果模式（mask_mode == kCausal）下：
// cpp

// apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);

// 如果 Q_idx_lane_base > K_idx_lane_base，说明关键在查询之前，保留分数；否则，设为负无穷（掩码掉）。


  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;

  // read from Valid_Block_Num to get how much iterations we need to do
//   计算当前线程块需要迭代多少次来处理稀疏的关键块。
// Valid_Block_Num 是一个数组，记录每个查询块的有效关键块数量（稀疏注意力只处理部分 K）
// num_iterations：从Valid_Block_Num取当前块需要迭代几次（稀疏性控制）。

// 如果迭代次数是0，直接退出。

// Lut（查找表）调整到当前批次、头和块的起始位置。

// 直白说：查表看这块要干几次活，没活就下班；调整“导航图”（LUT）到当前任务的位置。
  const uint32_t num_block_q = gridDim.x;
  const uint32_t num_iterations = Valid_Block_Num[batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx];

  if (num_iterations == 0)
  {
    return;
  }

  // move Lut to the correct place
  const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
  Lut += batch_id * num_qo_heads * num_block_q * num_block_k + head_id * num_block_q * num_block_k + bx * num_block_k;

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t KV_block_increm = 0;

  // load K with predicate
  // 查导航图找到第一个K的位置
  KV_block_increm = Lut[0];
  K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
  K_load_idx_lane_base += KV_block_increm * CTA_K;
  K_idx_lane_base += KV_block_increm * CTA_K;
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  k_scale_idx += KV_block_increm * k_scale_advance_offset;
  float dequant_scale = q_scale * K_scale[k_scale_idx];

  sm_scale = original_sm_scale * dequant_scale;

  // load V
  // ! we assume that V is padded. If not, there might be illegal memory access or nan issue.
  // for fp16: 
  // load_global_to_share                stride_seq_v
  V_lane_base_ptr += KV_block_increm * CTA_K;
  load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
  cp_async::commit_group();
  // 为什么从1开始，到倒数第二结束？：
  // 前一次迭代（iter=0）已经在循环外加载并处理了（之前的代码）。
  
  // 最后一次迭代（iter=num_iterations-1）可能有特殊处理（比如写回结果），所以留到循环外
#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
    smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    
    float RS_f32[num_tiles_q][num_tiles_k][8];

    // RS是量化后的整数，Softmax需要浮点数。
    // dequant_scale（Q和K的缩放因子乘积）把量化值还原成真实分数
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    if constexpr (pv_threashold_mode != PVThresholdMode::kNone) // use pv_threshold to skip unnecessary computation
    {
      __syncthreads();

      // first issue the load for K
      KV_block_increm = Lut[iter];
      K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
      K_load_idx_lane_base += KV_block_increm * CTA_K;
      K_idx_lane_base += KV_block_increm * CTA_K;
      load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
      cp_async::commit_group();
  
      float local_max_diff = update_mo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, sm_scale);

      // reduce max diff in a warp
      local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x4));
      local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x8));
      local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x10));

      if constexpr (pv_threashold_mode == PVThresholdMode::kPerBlock)
      {
        // reduce max diff in a block
        static __shared__ float reduced_buffer[num_warps * 32];
        reduced_buffer[lane_id + warp_id * 32] = local_max_diff;
        __syncthreads();

        if constexpr (num_warps == 4)
        {
          local_max_diff = reduced_buffer[(lane_id % 4) * 32 + lane_id];
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x1));
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x2));
        }
        else if constexpr (num_warps == 8)
        {
          local_max_diff = reduced_buffer[(lane_id % 8) * 32 + lane_id];
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x1));
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x2));
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x4));
        }
        else
        {
          static_assert(num_warps == 4 || num_warps == 8, "num_warps must be 4 or 8");
        }
      }

      // ensure V is ready
      cp_async::wait_group<1>();
      __syncthreads();

      // skip the computation on warp level
      if (local_max_diff + pv_threshold > 0)
      {
        if constexpr (return_pv_count)
        {
          pv_count++;
        }

        exponentiate_r<num_tiles_q, num_tiles_k, true>(RS_f32, m, sm_scale);
        
        if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
        {
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
        }
    
        uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
        RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
    
        if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
        {
          accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
        }

        if constexpr (!use_inst_buffer)
        {
          compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
            smem_V, RS_f8, RO, d);
        }
        else
        {
          compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
            smem_V, RS_f8, RO, d);   
        }
      }
    }
    else // if we don't use pv_threshold, we just do the computation
    {

      if constexpr (return_pv_count)
      {
        pv_count++;
      }

      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, sm_scale);
  
      if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
      {
        accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
      }
  
      uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
      RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
  
      if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
      {
        accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
      }
  
      __syncthreads();
  
      // load K
      KV_block_increm = Lut[iter];
      K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
      K_load_idx_lane_base += KV_block_increm * CTA_K;
      K_idx_lane_base += KV_block_increm * CTA_K;
      // 做double buffer，提前加载下一轮的K
      load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
      cp_async::commit_group();
  
      // ensure V is ready
      cp_async::wait_group<1>();
      __syncthreads();
  
      // for fp16:
      // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
      //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
      if constexpr (!use_inst_buffer)
      {
        compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);
      }
      else
      {
        compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);   
      }
    }

    __syncthreads();
    // load V
    // for fp16: 
    // load_global_to_share                stride_seq_v
    V_lane_base_ptr += KV_block_increm * CTA_K;
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();
  
    k_scale_idx += KV_block_increm * k_scale_advance_offset;
    dequant_scale = q_scale * K_scale[k_scale_idx];
    sm_scale = original_sm_scale * dequant_scale;
  }

  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
    smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    __syncthreads();

    if constexpr (return_pv_count)
    {
      pv_count++;
    }

    // load K with predicate
    KV_block_increm = Lut[num_iterations - 1];
    K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
    K_load_idx_lane_base += KV_block_increm * CTA_K;
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    K_idx_lane_base += KV_block_increm * CTA_K;

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, original_sm_scale);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    __syncthreads();

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }

    __syncthreads();
    // load V
    // for fp16: 
    // load_global_to_share                stride_seq_v
    V_lane_base_ptr += KV_block_increm * CTA_K;
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();

    k_scale_idx += KV_block_increm * k_scale_advance_offset;
    dequant_scale = q_scale * K_scale[k_scale_idx];
    sm_scale = original_sm_scale * dequant_scale;

  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
    smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (return_pv_count)
    {
      pv_count++;
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, original_sm_scale);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }

    __syncthreads();

  }

  if constexpr (return_pv_count)
  {

    if (lane_id == 0)
    {
      PV_Count[batch_id * num_qo_heads * num_block_q * num_warps_q + head_id * num_block_q * num_warps_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>()] = pv_count;
    }

    __syncthreads();
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  // ! here we just implement the case for fp32 acumulation
  if constexpr (fuse_v_scale)
  {
    float v_scale[4];
    float *V_scale_base_ptr = V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
      ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        RO[fq][fv][0] *= v_scale[0];
        RO[fq][fv][1] *= v_scale[1];
        RO[fq][fv][2] *= v_scale[0];
        RO[fq][fv][3] *= v_scale[1];
        RO[fq][fv][4] *= v_scale[2];
        RO[fq][fv][5] *= v_scale[3];
        RO[fq][fv][6] *= v_scale[2];
        RO[fq][fv][7] *= v_scale[3];
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          if constexpr (std::is_same<DTypeOut, half>::value)
          {
            ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
          }
          else
          {
            ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
          }
        }

        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

        offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      { 
        // TODO: not implement
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }
}

void qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor lut,
                    torch::Tensor valid_block_num,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(lut);
  CHECK_CUDA(valid_block_num);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value); // ensure value is contiguous to prevent troubles in the kernel
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(lut);
  CHECK_CONTIGUOUS(valid_block_num);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  // TODO: how to check fp8 data type?
  // CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(lut, torch::kInt32);
  CHECK_DTYPE(valid_block_num, torch::kInt32);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);
  CHECK_DTYPE(value_scale, torch::kFloat32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(lut, 4);
  CHECK_DIMS(valid_block_num, 3);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 3);

  const int batch_size = query.size(0);
  const int head_dim = query.size(3);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k, stride_h_v, stride_d_v, stride_seq_o, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);

    stride_seq_q = query.stride(1);
    stride_h_q = query.stride(2);
    stride_seq_k = key.stride(1);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_d_v = value.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_o = output.stride(2);

    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    assert(value.size(1) == head_dim);
    assert(value.size(2) == num_kv_heads);
  }
  else
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);

    stride_seq_q = query.stride(2);
    stride_h_q = query.stride(1);
    stride_seq_k = key.stride(2);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_d_v = value.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_o = output.stride(1);

    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    assert(value.size(2) == head_dim);
    assert(value.size(1) == num_kv_heads);
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            
          constexpr int CTA_Q = 128;
          constexpr int CTA_K = 64;
          constexpr int WARP_Q = 32;
          constexpr int WARP_K = 64;

          assert(value.size(0) == batch_size);
          assert(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K);

          constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

          if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerBlock))
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
          }
          else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
          }
          else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);    
          }
          else
          {
            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerBlock) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
          }

          CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

          CHECK_SHAPE(lut, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q), div_ceil(kv_len, CTA_K));
          CHECK_SHAPE(valid_block_num, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));

          //                                     smem_Q                                     smem_K                            smem_V                     smem_O
          size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
          
          auto kernel_func = qk_int_sv_f8_block_sparse_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                      float, true, PVThresholdMode::kNone, DTypeOut, ComputeUnit::kCudaCore, mask_mode, true, false>;

          cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

          dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
          dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

          kernel_func<<<grid, block, smem_max>>>(
            query.data_ptr<int8_t>(), 
            key.data_ptr<int8_t>(),
            reinterpret_cast<int8_t*>(value.data_ptr()),
            reinterpret_cast<DTypeOut*>(output.data_ptr()),
            nullptr,
            reinterpret_cast<int*>(lut.data_ptr()),
            reinterpret_cast<int*>(valid_block_num.data_ptr()),
            nullptr,
            reinterpret_cast<float*>(query_scale.data_ptr()),
            reinterpret_cast<float*>(key_scale.data_ptr()),
            reinterpret_cast<float*>(value_scale.data_ptr()),
            qo_len,
            kv_len,
            num_kv_groups,
            stride_bz_q, stride_seq_q, stride_h_q,
            stride_bz_k, stride_seq_k, stride_h_k,
            stride_bz_v, stride_h_v, stride_d_v,
            stride_bz_o, stride_seq_o, stride_h_o,
            sm_scale);
        });
      });
    });
  });
}

torch::Tensor qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor lut,
                    torch::Tensor valid_block_num,
                    torch::Tensor pv_threshold,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_pv_count)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(lut);
  CHECK_CUDA(valid_block_num);
  CHECK_CUDA(pv_threshold);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value); // ensure value is contiguous to prevent troubles in the kernel
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(lut);
  CHECK_CONTIGUOUS(valid_block_num);
  CHECK_CONTIGUOUS(pv_threshold);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  // TODO: how to check fp8 data type?
  // CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(lut, torch::kInt32);
  CHECK_DTYPE(valid_block_num, torch::kInt32);
  CHECK_DTYPE(pv_threshold, torch::kFloat32);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);
  CHECK_DTYPE(value_scale, torch::kFloat32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(lut, 4);
  CHECK_DIMS(valid_block_num, 3);
  CHECK_DIMS(pv_threshold, 1);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 3);

  const int batch_size = query.size(0);
  const int head_dim = query.size(3);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k, stride_h_v, stride_d_v, stride_seq_o, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);

    stride_seq_q = query.stride(1);
    stride_h_q = query.stride(2);
    stride_seq_k = key.stride(1);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_d_v = value.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_o = output.stride(2);

    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    assert(value.size(1) == head_dim);
    assert(value.size(2) == num_kv_heads);
  }
  else
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);

    stride_seq_q = query.stride(2);
    stride_h_q = query.stride(1);
    stride_seq_k = key.stride(2);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_d_v = value.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_o = output.stride(1);

    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    assert(value.size(2) == head_dim);
    assert(value.size(1) == num_kv_heads);
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  torch::Tensor pv_count = torch::empty({0});

  auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_RETURN_PV_COUNT(return_pv_count, RETURN_PV_COUNT, {
        DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
              
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            if constexpr (RETURN_PV_COUNT)
            {
              pv_count = torch::empty({batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q)}, query.options().dtype(torch::kInt32));
            }

            assert(value.size(0) == batch_size);
            assert(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K);

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerBlock))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);    
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerBlock) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);
            
            CHECK_SHAPE(lut, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q), div_ceil(kv_len, CTA_K));
            CHECK_SHAPE(valid_block_num, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
            CHECK_SHAPE(pv_threshold, num_qo_heads);

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func = qk_int_sv_f8_block_sparse_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                        float, true, PVThresholdMode::kNone, DTypeOut, ComputeUnit::kCudaCore, mask_mode, true, false>;

            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            kernel_func<<<grid, block, smem_max>>>(
              query.data_ptr<int8_t>(), 
              key.data_ptr<int8_t>(),
              reinterpret_cast<int8_t*>(value.data_ptr()),
              reinterpret_cast<DTypeOut*>(output.data_ptr()),
              (RETURN_PV_COUNT) ? reinterpret_cast<int*>(pv_count.data_ptr()) : nullptr,
              reinterpret_cast<int*>(lut.data_ptr()),
              reinterpret_cast<int*>(valid_block_num.data_ptr()),
              reinterpret_cast<float*>(pv_threshold.data_ptr()),
              reinterpret_cast<float*>(query_scale.data_ptr()),
              reinterpret_cast<float*>(key_scale.data_ptr()),
              reinterpret_cast<float*>(value_scale.data_ptr()),
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_h_v, stride_d_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });

  return pv_count;
}