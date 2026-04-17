# Agibot 智元机器人数据微调 OpenPI (π0) 实验记录

**最后更新日期**: 2026年4月17日

## 实验目标
使用智元机器人 (Agibot) 采集的数据集 (`agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2`)，在 openpi 框架中对 `π0 Base Model` 进行微调训练 (Fine-Tuning)。本次训练已确认采用 **LoRA 低秩微调**，必须万无一失，确保可成功部署在智元机器人上。根据策略，先采用大迭代步数训练观察Loss收敛趋势。

---

## 已完成的进度

### 1. 数据集梳理与格式转换器准备
- **数据探索**: 分析了 Agibot 原始数据。确认包含了 `actions` (8维), `states` (8 维), 以及两个视角的相机视频 (`camera_0.mp4` 作为头部相角, `camera_1.mp4` 作为手臂视角)，单帧图像分辨率为 `(240, 320, 3)` RGB。
- **转换脚本**: 编写并深度排查了格式转换脚本 `agi_bot/convert_agibot_data_to_lerobot.py`。
  - 使用 `cv2` 处理视频抽帧，构建支持 openpi 训练框架的数据集 `agibot_routeB`。
  - 将转换脚本中硬编码的 `task="pick up the object"` 变更为 `"grasp_bottle"`，并因应 `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` 版本迭代将任务标签放置入 `dataset.add_frame()` 的特征字典中，通过在数据帧记录 `task` 解决特征匹配不足的 ValueError。

### 3. OpenPI Docker 环境变量排雷及网络加速方案部署
- **W&B API 密钥环境注入修复**: 
  - 第一轮运行尝试通过命令行终端粘贴 Wandb V1 版的 72 位 API Key。受制于老旧 SDK 版本的 40 字符校验异常，在 Python 解释器内引发 `ValueError`。
  - **解决方案**: 为 Docker 的配置修改为向其传递 `-e WANDB_API_KEY` 环境变量配置，彻底绕过了由于命令行输入阶段因字符截断校验导致的训练器异常退出问题，让训练能够成功上推至云端实时监控大盘。
- **预训练 Checkpoint 突破墙下载速度阻碍 (持久化映射)**: 
  - 本框架依赖的 `pi0_base` 先验模型参数有 11.2GB。当系统自动通过 `gcsfs` 后备机制下载极易超时，且 `--rm` 的 Docker 会在意外后直接删掉宝贵的临时层缓存。
  - **解决方案**: 通过全局调用 `gsutil -m cp` 实现多线程直链并发，加速把数据固化在宿主机的 `~/.cache/openpi/` 中；之后通过 `-v ~/.cache/openpi:/root/.cache/openpi` 实现了模型热启动。
  - **网络代理导致哈希损坏报错修复**: 国内多线程挂载代理直连谷歌云盘时易产生残存的断点和 checksum 错乱 (导致 `CommandException` 和 `crc32c signature doesn't match`)。最终确认通过 `gsutil -o "GSUtil:check_hashes=never" -m cp -r` 直接关闭 `gsutil` 的云端强一致哈希校验顺利下载了完整的 11GB 权重字典文件。
  - **解压损坏修复**: 由于使用 `check_hashes=never`，下载被代理截断时可能会生成损坏的分片。这导致在读取 `params.PaliGemma.llm...` 时爆出 `ZSTD_decompressStream() failed` 错误。我们编写了一个 Python 脚本对比本地与云端 `gsutil ls -L` 的 MD5 值，并筛出了缺失的分片，成功找出了 `fec292...` 及 `93fcf3...` 的损坏/缺失分片，重新下载后恢复了 11.2GB Checkpoint 的完整性。

## 待完成事项
  - 经源码追踪，`ModelTransformFactory` 已自动注入 `PadStatesAndActions`，进行零填充对齐。在推理输出端，`AgibotOutputs` 会将 32 维裁剪回 8 维以适配真机。
- **LoRA 模式开启**:
  - 在 `src/openpi/training/config.py` 中将 `paligemma_variant="gemma_2b_lora"` 与 `action_expert_variant="gemma_300m_lora"` 加入到模型参数结构中。
  - 开启了 `freeze_filter` 冻结了除 LoRA 变体外的主干网络。设定 `ema_decay=None` (LoRA无须EMA参数平滑)。

### 3. 长周期观察训练参数设定 (30,000步)
- 按照经验，对于机器人的复杂或未定性数据，先开启长周期的 3w 步观察 loss 的下降（通过 `wandb_enabled=True`）是最稳妥的初步动作评估法。
- 参数设定 (`pi0_agibot` 配置修改)：
  - **总步数 (steps)**: 修改 `num_train_steps=30000`。
  - **保存频率 (Checkpoint)**: 设定 `save_interval=2000`，每 2000步 将当前的最优权重保存在本地 (不覆盖, `keep_period=2000` 保留节点)。
  - **回显频率 (Logging)**: 设定 `log_interval=10`，每 10步 在终端以及 WandB 打点记录。
  - **衰减与热身**: 设定 `warmup_steps=1000` 和 `decay_steps=30000`，允许平滑收敛。

### 4. Docker 运行环境攻坚完成
- 经过复杂网络环境（Great Firewall）调整，修改了 `serve_policy.Dockerfile`：
  - 成功汇入清华 TUNA APT 源、Daocloud 镜像站、Ubuntu deadsnakes PPA。
  - **Docker 镜像 `openpi_server:latest` 构建已 100% 成功并保存在本地**。

---

## 随时可开始的操作指令 (Execution Guide)

代码与参数均设为完美绿灯状态，进入 Docker 即可开跑：

**第一步：进入 Docker 容器 (挂载全部代码、数据和 GPU)**
```bash
sudo docker run --rm -it --network=host -v $PWD:/app --gpus=all openpi_server /bin/bash
```

**(以下命令均在容器内执行)**

**第二步：基于wandb登录打通在线面板 (只需执行一次) **
```bash
uv run wandb login
```

**第三步：将原始数据构建为 LeRobot 格式数据集**
```bash
uv run agi_bot/convert_agibot_data_to_lerobot.py --data_dir agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2
```

**第四步：计算数据集归一化参数 (Norm Stats)**
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_agibot
```

**第五步：启动 π0 LoRA 微调训练 (约 30000 Steps)**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_agibot --exp-name=agibot_routeB_lora_tuning --overwrite
```
## 更新历史
*   2026-04-16: 修正了数据集 `task` 的格式绑定问题，成功绕开 wandb_api_key 版本验证 `ValueError` bug。搭建了 `gsutil` 并发下载挂载机制，极大提高了参数冷启动的速度。
*   2026-04-17: 成功修复了因下载断点引发的 Checkpoint md5sum crc 不一致导致的解压失败问题。打通了整条训练链路，并编写了针对物理边缘部署端的纯 Numpy 数组解构测试脚本 `run_inference.py`。

### 5. 推理 (Inference) 脚本制备
- 在等待模型顺畅训练期间，已编写脱离机器人的推理测试框架文件。
- **文件路径**: [agi_bot/run_inference.py](agi_bot/run_inference.py)
- **部署指南**: [agi_bot/README_INFERENCE.md](agi_bot/README_INFERENCE.md)
  - 通过 `pi0_agibot` config 调用 `.infer()`，构建了基于当前检查点的纯 Numpy / JAX Array 数组的测试伪造输入 (`observation/image`, `observation/wrist_image`, `observation/state`)。
  - 通过 `AgibotOutputs` Transform 方法拦截 32 维的张量底层返回，顺利裁切还原成智元原装的 `np.ndarray (10, 8)` Action Chunk 动作块进行机械臂直接发包执行。

### 6. 代码版本控制与跨设备迁移 (2026-04-17)
- **GitHub 远端托管**: 已通过 `git push` 将除 `checkpoints/` 和 `data/` 以外的所有核心代码、配置文件和推理脚本成功同步至 GitHub 仓库 (`https://github.com/Tito-11/openpi_agibot.git`)。
- **配置白名单**: 仓库中的 `.gitignore` 规则成功拦截了大体积二进制文件（包含模型权重和训练数据集），避免触发 GitHub 的 100MB 单文件限制。
- **权重提取与云盘流转**: 训练在约 28000 步左右参数完全收敛，已在此提取最高质量权重 `checkpoints/pi0_agibot/agibot_routeB_lora_tuning/28000` 文件夹并上传至个人云盘，明天的物理跨机器设备推理测试将直接从此存档加载。
