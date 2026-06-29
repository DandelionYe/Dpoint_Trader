# Dpoint_Trader Lineage

Dpoint_Trader 是从工作区中 10 个项目文件夹经过全面分析后，从零重构而来的新项目。

它不是任何一个旧 Git 仓库的直接延续，也不是通过 merge 旧仓库形成的项目。

## Directly Referenced Projects

直接参考的两个项目：

| 来源项目 | 评分 | 贡献模块 |
|---|---:|---|
| Ashare_DpointTrader_deeplearning_Ver2.0 | 8.5/10 | 单股回测引擎、分轮搜索引擎、Top-K 池、exploit-explore 搜索、概率校准、特征解释、Regime 检测、A 股执行约束 |
| DpointTrader_deeplearning_Ver1.0 | 8.0/10 | 面板数据架构、横截面排名特征、组合回测引擎、序列构建器、日期粒度切分、实验契约系统、Rank IC 指标 |

## Reconstruction Summary

Dpoint_Trader = Ver2.0 的搜索引擎和分析工具 + Ver1.0 的面板架构和组合回测 + 全新的统一 CLI 和模块化设计。

## Git History Policy

main 分支只表示 Dpoint_Trader 新项目自己的真实演进历史。

旧项目历史如需保留，应作为 legacy/* branches 或 legacy-source/* annotated tags 归档，而不是 merge 到 main。
