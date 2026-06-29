# Legacy Projects

Dpoint_Trader 重构前共分析了 10 个项目文件夹。

## Directly Referenced Projects

| 项目 | 评分 | 处理方式 | 主要贡献 |
|---|---:|---|---|
| Ashare_DpointTrader_deeplearning_Ver2.0 | 8.5/10 | 直接参考 | 单股回测引擎、分轮搜索引擎、Top-K 池、exploit-explore 搜索、概率校准、特征解释、Regime 检测、A 股执行约束 |
| DpointTrader_deeplearning_Ver1.0 | 8.0/10 | 直接参考 | 面板数据架构、横截面排名特征、组合回测引擎、序列构建器、日期粒度切分、实验契约系统、Rank IC 指标 |

## Deprecated Projects

| 项目 | 评分 | 废弃原因 |
|---|---:|---|
| Ashare_DpointTrader 2.0 | 7.5/10 | 纯 ML 版本，功能被 Ver2.0 完全覆盖 |
| Ashare_DpointTrader_deeplearning | 7.5/10 | Ver2.0 的前身 |
| Ashare_DpointTrader_deeplearning_Ver3.0 | 8.0/10 | 模块合并后可维护性下降，trainer.py 2800+ 行 |
| Ver3.0 备份 codex 未改 | 8.0/10 | 备份快照，无独特价值 |
| DpointTrader_deeplearning_Ver1.0 I盘ver | 7.5/10 | 主版本的不完整副本 |
| DpointTrader_deeplearning_Ver1.0_c | 7.5/10 | 精简版，有已知 Bug |
| 交易机器学习 | 6.5/10 | 179 个文件，80% 冗余，无 Git 无测试 |
| 交易机器学习_备份 | 3.0/10 | 逐字节一致，无价值 |

## Policy

废弃项目不应被 merge 到 main。
它们只用于历史参考、对照分析或 legacy archive。

如果需要保留旧 Git 历史，应导入到 legacy/* branches。
如果只需要标记来源点，应创建 legacy-source/* annotated tags。
