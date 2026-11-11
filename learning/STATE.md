# 学习状态

## 环境
- commit：单元 1 结束于 `38b6682` 之后（本文件所在的 commit）。
- 本机只编辑。远端 `519.server:~/mini-nccl` 有 24 核，没有 GPU，装的是 torch 2.15.0+cpu。
- 学习者可用的 GPU：2×A40、2×4090、2×5090。拓扑还不清楚（哪些卡在同一台机器上）。没有 IB/RDMA/NVLink。学习者可以自己 ssh。

## 掌握状态
| 主题 | 状态 | 证据 |
|---|---|---|
| all-reduce 语义 | 可独立 | 诊断题直接答对 |
| 阻塞 send 双向死锁 | 可独立 | 诊断题答对原因。补充过：阈值是自己的发送缓冲区加对方的接收缓冲区，小消息不会触发 |
| reduce-scatter / all-gather 语义 | 可独立 | 概念由导师讲解。之后 W=4、N=8 的题目值和归属全对，但误以为输出是全长向量（已纠正）。换成 N=6 后，独立答出项目会抛 `ValueError`，并独立提出 padding 方案 |
| ring all-reduce 数据流 | 未学 | |

## 单元 1：reduce-scatter / all-gather 语义（已完成）
- 已核查的源码（`mini_nccl/collectives.py`）：
  - `reduce_scatter` 在第 484 行。第 493-494 行要求长度能被 W 整除，函数返回 `blocks[r].clone()`，不修改输入。
  - `all_gather` 在第 459 行，返回一个按 rank 排列的列表。
  - padding 在三处都用到：`_ring_on_channel` 第 224-230 行（补齐）和第 286-287 行（截断），`_halving_doubling` 第 302-306 行，`fsdp.py` 第 94-97 行。
  - `_split`（第 96 行）用的是另一种切法：余数分给前面的块，不补零。这种切法用于按 channel 切分。
- 实验：`learning/exp01_reduce_scatter.py`，W=4，N=8，与主进程中算出的参考结果一致。
- 未验证：非整除长度在 ring all-reduce 中的实际表现（只读了代码，没有运行）。
- 尚存的误区：无。待确认：是否清楚 padding 的值不影响 max/prod 的正确性，以及所有 rank 必须使用同一个切分规则。

## 下一步任务
单元 2：ring all-reduce 的数据流。W=3，根据 `_ring_on_channel` 第 249-251 行的下标，填写 reduce-scatter 阶段的逐轮状态表。

## 闭卷复测（下次开头）
1. W=3，N=7，用 padding 方案做 reduce-scatter。每个 rank 拿到几个元素？padding 落在哪个 rank 的哪个位置？
2. 如果 op=max，把 padding 设成 0 会不会让结果出错？为什么？
