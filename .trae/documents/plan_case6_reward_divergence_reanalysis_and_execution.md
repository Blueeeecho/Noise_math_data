# Case\_6 奖励背离重分析与执行计划

## Summary

本计划基于真实日志 [277377-MTMT-A100.out](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out)、当前奖励实现 [reward\_case\_6.py](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py) 与训练入口 [run\_exp1.sh](file:///root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/run_exp1.sh) 制定，目标分为两层：

1. 重新给出一份**不混淆旧日志与当前代码状态**的分析文档。
2. 在执行阶段完成一轮**可验证、可推送**的代码修正，并在修正后推送远程。

这次计划要明确区分两个事实：

* `277377` 这次训练日志反映的是**旧版 case\_6 聚合逻辑**下的行为。

* 仓库当前代码已经包含一轮后续修正，但仍存在**默认值不一致**与**可观测性不足**的问题，导致下一轮训练即使变好，也很难直接从日志判断“到底是哪一项 reward 在主导”。

本计划的执行目标不是再空泛地讨论“reward 和 accuracy 为什么背离”，而是把后续实施收敛为三件具体工作：

* 统一 `case_6` 奖励默认值，避免不同入口仍落回旧权重。

* 把 reward 的各组成贡献显式暴露到验证输出，便于下一轮直接诊断。

* 修正完成后做最小必要验证，并按用户要求推送到远程。

## Current State Analysis

### 1. `277377` 的背离现象来自旧版聚合逻辑，不是当前仓库状态的直接镜像

从 [277377-MTMT-A100.out:L850-L876](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out#L850-L876) 可以看到首轮：

* `gsm8k-test_acc = 0.2085`

* `gsm8k-test_reward = 0.3013`

* `r_acc/mean@1 = 0.2234`

* `r_fmt/mean@1 = 0.6969`

* `step_process_score/mean@1 = 0.0323`

如果按旧公式：

* `0.7 * r_acc`

* `+ 0.4 * good_ratio`

* `- 0.3 * bad_ratio`

* `+ 0.2 * format_term`

去估算，这些数值与日志能对上；而当前仓库里的 [reward\_case\_6.py:L1328-L1361](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py#L1328-L1361) 已经是带 `process_gate` / `format_gate` 的新聚合。因此：

* `277377` 说明的是“旧 reward 为什么会把 reward 顶高”

* 它不能直接当作“当前代码仍然完全如此”的证据

### 2. 旧日志确认了 reward 背离的根因：稀疏答案项被稠密结构项压过

从日志中几轮关键点：

* [277377-MTMT-A100.out:L1538-L1664](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out#L1538-L1664)

* [277377-MTMT-A100.out:L2370-L2496](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out#L2370-L2496)

* [277377-MTMT-A100.out:L3200-L3327](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out#L3200-L3327)

* [277377-MTMT-A100.out:L13347-L13360](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out#L13347-L13360)

可以确认：

* `gsm8k-test_acc` 先降后缓慢回升，但长期低于初期高点

* `gsm8k-test_reward` 基本单调上涨

* `r_fmt` 从 `0.6969` 快速升到接近 `1.0`

* `step_process_score` 从 `0.0323` 升到接近 `1.0`

* `bad_count` 持续很低，惩罚不足

因此旧版问题已经足够明确：

* `r_acc` 是稀疏二值信号

* `good_ratio` 与 `format_term` 是稠密高频信号

* 名义权重大不等于平均贡献大

### 3. 当前仓库已经有第一轮修正，但存在“代码状态与文档状态不一致”

当前代码中：

* [reward\_case\_6.py:L1328-L1361](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py#L1328-L1361) 已经改成：

  * `process_gate = 0.4 + 0.6 * r_acc`

  * `format_gate = 0.2 + 0.8 * r_acc`

  * 总分为 `acc + gated_process + gated_format`

* [run\_exp1.sh:L297-L315](file:///root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/run_exp1.sh#L297-L315) 中 `case_6` 权重也已经改成：

  * `step_acc_weight = 1.0`

  * `step_good_weight = 0.25`

  * `step_bad_weight = 0.2`

  * `step_fmt_weight = 0.05`

但仍有两个现实问题：

#### 3.1 `reward_case_6.py` 的函数默认值仍停留在旧权重

在 [reward\_case\_6.py:L1245-L1248](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py#L1245-L1248) 里，`compute_reward()` 的默认参数还是：

* `step_acc_weight=0.7`

* `step_good_weight=0.4`

* `step_bad_weight=0.3`

* `step_fmt_weight=0.2`

这与 [run\_exp1.sh:L301-L304](file:///root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/run_exp1.sh#L301-L304) 中当前 `case_6` 实际训练入口不一致。只要有其他入口、脚本、离线分析或直接函数调用没有覆盖这些参数，就可能重新落回旧行为。

这不是文档问题，而是实际的**默认值漂移风险**。

#### 3.2 现有验证记录看不到 reward 贡献拆解

在 [ray\_trainer.py:L958-L984](file:///root/autodl-tmp/Reasoning360/verl/trainer/ppo/ray_trainer.py#L958-L984) 里，验证 JSONL 只记录：

* `r_acc`

* `r_fmt`

* `step_process_score`

* `step_good_count`

* `step_bad_count`

* 等基础字段

但当前 reward 的关键新字段并没有进入记录：

* `process_gate`

* `format_gate`

* `acc_contrib`

* `process_contrib`

* `format_contrib`

* `raw_process_term`

这意味着：

* 即使下一轮训练使用了新聚合

* 日志和 JSONL 仍然不够直接回答“reward 现在到底是不是由答案项主导”

### 4. 指标聚合链路支持新增数值字段，因此补可观测性是低风险修正

从 [metric\_utils.py:L416-L475](file:///root/autodl-tmp/Reasoning360/verl/trainer/ppo/metric_utils.py#L416-L475) 可见，验证阶段会对 `reward_extra_infos_dict` 中的数值字段自动做 `mean@k` 聚合。因此后续只要在 reward 返回中增加数值字段，并在 [ray\_trainer.py:L958-L984](file:///root/autodl-tmp/Reasoning360/verl/trainer/ppo/ray_trainer.py#L958-L984) 补到导出记录里，就能同时获得：

* 控制台 / wandb 级别的聚合曲线

* JSONL / CSV 级别的样本级排查线索

## Proposed Changes

### 1. 新建并固定这份文档，作为后续执行唯一依据

执行阶段不再沿用旧的 [plan\_case6\_reward\_accuracy\_divergence\_and\_fix.md](file:///root/autodl-tmp/Reasoning360/.trae/documents/plan_case6_reward_accuracy_divergence_and_fix.md) 作为唯一依据，因为它描述的是“准备去修改”的状态，而当前仓库已经部分修改过。

后续执行以本文件为准，明确区分：

* 旧日志解释

* 当前代码真实状态

* 仍需补齐的修正项

### 2. 修改 `reward_case_6.py`，统一默认值并显式返回贡献拆解

目标文件：

* [reward\_case\_6.py](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py)

执行阶段将做两类修改：

#### 2.1 对齐默认权重，避免函数入口回退到旧版本

将 `compute_reward()` 的默认值从旧版：

* `0.7 / 0.4 / 0.3 / 0.2`

改为与当前 `case_6` 训练入口一致的：

* `step_acc_weight = 1.0`

* `step_good_weight = 0.25`

* `step_bad_weight = 0.2`

* `step_fmt_weight = 0.05`

修改原因：

* 让函数默认行为与训练脚本一致

* 降低“某些入口仍使用旧参数”的风险

* 让离线调试和训练入口保持同一基线

#### 2.2 保持现有门控思路，但把关键贡献项显式返回

在不推翻当前门控设计的前提下，新增并返回以下数值字段：

* `raw_good_term = step_good_weight * good_ratio`

* `raw_bad_term = step_bad_weight * bad_ratio`

* `raw_process_term = raw_good_term - raw_bad_term`

* `acc_contrib = step_acc_weight * r_acc`

* `process_contrib = process_gate * raw_process_term`

* `format_contrib = step_fmt_weight * format_gate * format_term`

保留并继续返回：

* `process_gate`

* `format_gate`

* `r_acc`

* `r_fmt`

* `step_process_score`

这样后续运行时可以直接回答：

* reward 里答案项平均贡献是多少

* 过程项被 gate 后还剩多少

* 格式项是否仍然过强

### 3. 修改 `ray_trainer.py`，把新增 reward 字段写入验证导出

目标文件：

* [ray\_trainer.py](file:///root/autodl-tmp/Reasoning360/verl/trainer/ppo/ray_trainer.py)

执行阶段在验证记录 `record` 中新增读取并保存：

* `process_gate`

* `format_gate`

* `raw_good_term`

* `raw_bad_term`

* `raw_process_term`

* `acc_contrib`

* `process_contrib`

* `format_contrib`

修改原因：

* 当前 JSONL 样本记录看不到新 reward 的真实分解

* 只看 `r_acc` / `r_fmt` / `step_process_score` 不足以判断“总 reward 是如何被拼起来的”

* 用户后续还要继续整理训练结果，这些字段必须在导出里稳定存在

### 4. 保持 `run_exp1.sh` 的当前权重基线，但在执行阶段核对是否需要同步注释说明

目标文件：

* [run\_exp1.sh](file:///root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/run_exp1.sh)

当前 `case_6` 权重已经与“答案优先、过程辅助、格式弱 shaping”一致，因此执行阶段不计划再次改动数值，除非读到文件时发现被其他本地修改覆盖。

执行阶段只做两件事：

* 核对 `case_6` 分支仍保持 `1.0 / 0.25 / 0.2 / 0.05`

* 如有必要，补一条简短注释说明这组权重的设计目标，避免后续再次误改回旧版

### 5. 执行完成后推送远程

用户已经明确要求：

* 修正之后帮忙推送到远程

因此执行阶段的收尾步骤必须包含：

* 本地检查通过

* 查看改动范围，避免误带无关文件

* 提交本次修正

* 推送到远程 `origin`

## Assumptions & Decisions

* 当前计划以 [277377-MTMT-A100.out](file:///root/autodl-tmp/Reasoning360/logs/277377-MTMT-A100.out) 为“旧 reward 背离现象”的唯一分析依据。

* 当前仓库中的 [reward\_case\_6.py:L1328-L1361](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py#L1328-L1361) 与 [run\_exp1.sh:L297-L315](file:///root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/run_exp1.sh#L297-L315) 视为已存在的第一轮修正，不在执行阶段盲目重复推翻。

* 本次执行优先补齐：

  * 默认值一致性

  * reward 贡献拆解

  * 验证导出可观测性

* 本次不重新大改 `strict` / `natural` step classifier，因为从当前问题看，首要缺口不在分类规则，而在“新聚合是否真的生效、能否被清楚观测”。

* 推送远程属于本次执行范围。

## Verification Steps

1. 复核 [reward\_case\_6.py:L1245-L1248](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py#L1245-L1248) 与 [run\_exp1.sh:L301-L304](file:///root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/run_exp1.sh#L301-L304) 的默认值是否一致。
2. 修改后检查 [reward\_case\_6.py](file:///root/autodl-tmp/Reasoning360/examples/noise_math/reward_case_6.py) 返回结果中是否新增：

   * `raw_good_term`

   * `raw_bad_term`

   * `raw_process_term`

   * `acc_contrib`

   * `process_contrib`

   * `format_contrib`

   * `process_gate`

   * `format_gate`
3. 修改后检查 [ray\_trainer.py:L958-L984](file:///root/autodl-tmp/Reasoning360/verl/trainer/ppo/ray_trainer.py#L958-L984) 的验证记录是否包含上述字段。
4. 对修改后的 Python 文件运行最小语法检查，并对最近编辑文件跑诊断，确认没有新报错。
5. 如需最小行为验证，构造一个正确答案样本与一个错误答案样本，确认：

   * 正确样本的 `acc_contrib` 显著高于错误样本

   * 错误样本的 `process_contrib` / `format_contrib` 不会再单独掩盖 `acc_contrib = 0` 的事实
6. 查看提交范围，只包含本次计划涉及文件。
7. 提交并推送远程，向用户回报：

   * 改了哪些文件

   * 为什么这些改动是对当前缺口的直接修复

   * 推送到哪个 commit
