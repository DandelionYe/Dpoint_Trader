# Dpoint_Trader

A股深度学习量化交易研究框架 — 支持单股策略和多股因子/组合策略。

## 特性

- **双模式支持**：`dpoint single`（单股策略）和 `dpoint basket`（篮子/组合策略）
- **8种模型**：LogReg / SGD / XGBoost / MLP / LSTM / GRU / CNN1D / Transformer
- **完整验证体系**：Walk-Forward / Embargo WF / Nested WF / Final Holdout
- **A股约束建模**：涨跌停 / T+1 / 100股手数 / 佣金印花税 / 滑点
- **可配置搜索目标**：PnL（回测净值）或 Rank IC（排名信息系数）
- **实验可复现性**：种子管理 / manifest / 数据哈希 / 实验契约

## 安装

```bash
# 使用已有的 ashare_dpoint conda 环境
conda activate ashare_dpoint

# 安装项目（开发模式）
cd Dpoint_Trader
pip install -e .
```

## 使用

```bash
# 单股模式
dpoint single --data_path data/600698.xlsx --model lstm --runs 100 --metric pnl

# 篮子模式
dpoint basket --basket_path data/basket_1/ --model transformer --top_k 5 --metric rank_ic
```

## 项目结构

```
src/dpoint/
├── cli/           # CLI 入口（single / basket 子命令）
├── core/          # 常量、配置、任务类型、实验契约、工具
├── data/          # 数据加载（Excel/CSV/面板/篮子）
├── features/      # 特征工程（时序/横截面/标签/序列）
├── models/        # 模型工厂（sklearn + PyTorch）
├── search/        # 搜索引擎（可配置目标函数）
├── splits/        # 样本划分（WF/Embargo/Nested/Holdout）
├── backtester/    # 回测引擎（单股 + 组合）
├── reports/       # 报告生成（Excel + HTML）
├── analysis/      # 概率校准、特征解释、Regime 分析
└── tools/         # 工具（对比、更新、K线可视化）
```

## 开发状态

- [x] Phase 1: 基础框架（core/data/splits）
- [x] Phase 2: 特征与模型（features/models）
- [x] Phase 3: 搜索引擎 + 回测引擎（search/backtester）
- [x] Phase 4: 报告生成（reports/excel + html + ranking_metrics）
- [x] Phase 5: CLI 端到端流程（single/basket 完整 pipeline）
- [x] Phase 6: 分析工具 + CI/CD（calibration/explainer/regime/compare_runs/CI）
- **全部 64 个测试通过** ✅

### 可选依赖

以下库未安装不影响核心功能（代码中有 `try/except` 优雅降级）：

- `shap` — 特征解释（SHAP 值），`pip install shap`
- `plotly` — 交互式 HTML 报告，`pip install plotly`
- `xgboost` — XGBoost 模型，`pip install xgboost`（不可用时自动回退到 GradientBoosting）
