# Agibot 数据清洗与过滤指南 (Data Cleaning Guide)

**最后更新日期**: 2026-05-09

## 为什么需要数据清洗？

对于像 OpenPI (π0.5) 这样的 VLA（视觉-语言-动作）模型，数据的质量直接决定了模型泛化和执行的上限。正如 DROID 数据集训练指南中所提到的：“Like any diverse real-robot dataset, the dataset isn't perfectly 'clean' and we have found data filtering to significantly improve policy performance.”

在真实的机器人遥操作（Teleoperation）或数据采集中，通常会包含大量的 **Idle 帧（静止帧）**。例如：
- 操作员在开始抓取前，犹豫了几秒钟。
- 机械臂到达目标位置后，停顿了一段时间才闭合夹爪。
- 任务结束后，操作员没有立刻停止录制。

**如果把这些 Idle 帧喂给模型，会导致极其严重的后果：**
1. **模型变得“迟钝”或“卡死”**：由于模型在预测 Action Chunk 时发现大量标签都是“原地不动”，它在推理时会倾向于输出极小的位移甚至静止指令，导致机械臂在半空中停滞不前。
2. **浪费计算资源**：静止帧不包含任何有用的物理意图，却要消耗昂贵的 VLM 视觉编码算力。

## 核心清洗标准：Idle 帧过滤 (Idle Frame Filtering)

参考 OpenPI 官方在 DROID 数据集上的最佳实践，我们为 Agibot 数据集制定了以下清洗规则：

1. **阈值判定**：如果相邻两帧之间的关节/末端位置变化小于设定的微小阈值（如 `1e-3`），则认为该帧是 Idle 帧。
2. **最大连续静止允许 (min_idle_len)**：如果连续静止的帧数超过阈值（如 7 帧），则将这部分完全剔除。因为 `action_horizon` 通常是 10，过滤长静止段能防止模型输出全是静止的 action chunk。
3. **最小有效动作段 (min_non_idle_len)**：如果一段连续的运动非常短（比如不到 16 帧，约 1.5 秒），这可能是操作员的手抖，也应作为噪音过滤。
4. **截断末端 (filter_last_n_in_ranges)**：通常运动结束时的最后几帧都伴随着静止和减速，可以适当剔除以保持动作的干脆利落。

---

## 实施方案：集成至 LeRobot 转换脚本

为了最大化效率并保持原始数据的完整性，我们不需要去删除 `/agi_bot/data` 里的 `.npy` 和 `.mp4` 文件。
相反，我们直接在 `convert_agibot_data_to_lerobot.py` 转换脚本中引入**动态过滤逻辑**。在读取每一帧之前，先计算其是否属于有效的运动范围。只有在有效范围内的帧，才会被 `dataset.add_frame()` 写入最终的 `.parquet` 文件中。

### 脚本修改详情 (`convert_agibot_data_to_lerobot.py`)

我们在脚本中加入了 `compute_valid_ranges` 函数，它会在处理每个 episode 前，先读取 `actions.npy` 计算出应该保留的帧索引范围：

```python
def compute_valid_ranges(actions, min_idle_len=7, min_non_idle_len=16, threshold=1e-3):
    """
    计算 episode 中有效（非静止）的动作帧范围。
    actions: 形状为 (N, action_dim) 的 numpy 数组。
    """
    # 计算相邻两帧的动作差的绝对值
    action_diffs = np.abs(actions[1:] - actions[:-1])
    
    # 我们只关注位置的移动，假设前3维是 x, y, z
    # 如果平移的变动小于阈值，认为是静止
    pos_diffs = action_diffs[:, :3]
    is_idle = np.all(pos_diffs < threshold, axis=1)
    
    # 补齐长度使其与原 action 长度一致，默认第一帧不静止
    is_idle = np.concatenate([[False], is_idle])
    
    # 寻找连续的 False (运动) 或 True (静止) 区间
    # 此处省略具体区间切分逻辑，详见转换脚本源码...
    
    return valid_indices
```

通过这一步清洗，您的 LeRobot 数据集将变得极其紧凑和纯粹，模型学习到的将是**“看到目标 -> 果断出击”**的高效映射！
