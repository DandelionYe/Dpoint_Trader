# Dpoint_Trader

A股深度学习量化交易研究框架 — 支持单股策略和多股因子/组合策略。

## 特性

- **四命令支持**：`dpoint single`（单股策略）、`dpoint basket`（篮子/组合策略）、`dpoint resume`（迭代搜索）、`dpoint fetch`（数据获取）
- **8种模型**：LogReg / SGD / XGBoost / MLP / LSTM / GRU / CNN1D / Transformer
- **完整验证体系**：Walk-Forward / Embargo WF / Nested WF / Final Holdout
- **A股约束建模**：涨跌停 / T+1 / 100股手数 / 佣金印花税 / 滑点
- **可配置搜索目标**：PnL（回测净值）或 Rank IC（排名信息系数）
- **实验可复现性**：种子管理 / manifest / 数据哈希 / 实验契约
- **多维度篮子筛选**：支持按 4 级行业 / 省份 / 城市 / 所有权共 7 个维度组合筛选

## 安装

```bash
# 使用已有的 ashare_dpoint conda 环境
conda activate ashare_dpoint

# 安装项目（开发模式）
cd Dpoint_Trader
pip install -e .
```

## 使用

### 数据获取（需要 XtMiniQMT 运行）

```bash
# 获取单只股票数据
dpoint fetch single --code 000001 --start 20200101

# 按行业获取篮子数据
dpoint fetch basket --ind4 C27

# 按省份获取
dpoint fetch basket --province 广东省

# 多维度组合筛选（行业 + 省份 + 所有权）
dpoint fetch basket --ind4 C27 --province 广东省 --ownership 私营企业

# 向后兼容：--industry 等同于 --ind4
dpoint fetch basket --industry C27

# 查看可选分类值
dpoint fetch list-industries --level 4   # 列出四级行业
dpoint fetch list-provinces              # 列出省份
dpoint fetch list-ownership              # 列出所有权类型
dpoint fetch list-cities --province 广东省  # 列出某省城市
```

### 策略运行

```bash
# 单股模式
dpoint single --data_path data/600698.xlsx --model lstm --runs 100 --metric pnl

# 篮子模式
dpoint basket --basket_path data/basket_1/ --model transformer --top_k 5 --metric rank_ic

# 迭代搜索（从上次结果继续优化）
dpoint resume output/single_003 --runs 50 --metric pnl
dpoint resume output/single_003 --runs 50 --metric pnl --seed 123  # 使用新种子
```

## 项目结构

```
Dpoint_Trader/
├── data/                      # 数据文件
│   ├── TRD_Co.csv             # 国泰安行业分类原始数据
│   └── csmar_industry.sqlite  # 构建的行业分类 SQLite（7维度筛选）
├── scripts/
│   └── build_industry_db.py   # 从 CSV 构建 SQLite 的脚本
└── src/dpoint/
    ├── cli/           # CLI 入口（single / basket / resume / fetch 子命令）
    ├── core/          # 常量、配置、任务类型、实验契约、工具
    ├── data/
    │   ├── fetch/     # 数据获取（QMT 客户端、行业分类、格式转换）
    │   ├── csv_loader.py
    │   ├── excel_loader.py
    │   └── basket_loader.py
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
- [x] Phase 7: 迭代搜索（resume 精确续跑，search_state 序列化）
- [x] Phase 8: 数据获取（fetch single/basket，多维度行业分类筛选）
- **101 个测试通过** ✅（4 个 QMT 客户端测试需要 XtMiniQMT 运行）

### 数据获取说明

篮子模式支持 7 个维度的分类筛选，数据来源为国泰安 CSMAR：

| 维度 | 参数 | 示例 | 分类数量 |
|------|------|------|---------|
| 一级行业 | `--ind1` | `--ind1 金融` | 6 |
| 二级行业 | `--ind2` | `--ind2 银行业` | 72 |
| 三级行业 | `--ind3` | `--ind3 货币金融服务` | 82 |
| 四级行业（中信） | `--ind4` | `--ind4 C27` | 83 |
| 省份 | `--province` | `--province 广东省` | 34 |
| 城市 | `--city` | `--city 深圳市` | 434 |
| 所有权 | `--ownership` | `--ownership 私营企业` | 8 |

所有参数可选，可组合取交集。`--industry` 作为 `--ind4` 的别名保持向后兼容。

### 可选依赖

以下库未安装不影响核心功能（代码中有 `try/except` 优雅降级）：

- `shap` — 特征解释（SHAP 值），`pip install shap`
- `plotly` — 交互式 HTML 报告，`pip install plotly`
- `xgboost` — XGBoost 模型，`pip install xgboost`（不可用时自动回退到 GradientBoosting）
