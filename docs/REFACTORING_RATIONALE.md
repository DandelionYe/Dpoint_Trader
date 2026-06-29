# Refactoring Rationale

Dpoint_Trader 选择从零重构，而不是直接继承某一个旧仓库，主要原因如下：

## Architectural Difference

两个核心参考版本的架构差异较大：

- Ashare_DpointTrader_deeplearning_Ver2.0 偏向单股搜索、单股回测、概率校准、特征解释、Regime 检测和 A 股执行约束。
- DpointTrader_deeplearning_Ver1.0 偏向面板数据、横截面排名、组合回测、序列构建、日期粒度切分、实验契约系统和 Rank IC 评价。

直接合并两者会导致主干代码复杂、职责混乱、可维护性下降。

## Selected Design

Dpoint_Trader 采用以下设计：

| 决策 | 选择 | 理由 |
|---|---|---|
| 基础版本 | 从零重构 | 两者架构差异大，单股 vs 面板，从零搭建最干净 |
| 共存方式 | dpoint single / dpoint basket 子命令 | 代码复用度最高，同时保持业务边界清晰 |
| 搜索目标 | 可配置 PnL / Rank IC | 单股用 PnL 更直观，篮子用 Rank IC 更符合量化范式 |
| 回测层 | 保留两套引擎，统一 BacktestResult | 单股全仓 vs 多股组合是根本差异，不宜强行统一 |

## Result

最终设计目标是：

- 保留 Ver2.0 的搜索引擎和分析工具优势；
- 保留 Ver1.0 的面板架构和组合回测优势；
- 用新的 CLI、模块边界和结果契约统一整体项目；
- 避免旧项目中的重复、备份、长文件和历史包袱。
