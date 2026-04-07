# 项目运行状态总结

**检查日期**: 2026 年 3 月 25 日

---

## ✅ 验证通过的功能

### 1. 模块导入
```
✅ data_loader OK
✅ feature_dpoint OK
✅ trainer OK
✅ portfolio_backtester OK
✅ reporter OK
✅ main_cli OK
```

### 2. 数据加载
```
✅ 成功加载 24 只股票
✅ 日期范围：2017-12-18 ~ 2026-03-20
✅ 共同交易日数：1826
✅ 被排除股票：0 只
```

### 3. 特征构建
```
✅ 面板数据：24 只股票，81002 行
✅ 特征构建：80738 样本，17 列（含 date/stock_code）
✅ 特征数：15 个
✅ 标签分布：{0: 41844, 1: 38894}
```

---

## ⚠️ 发现的问题

在运行完整流程时遇到一个索引对齐 bug：

```
IndexError: index 25753 is out of bounds for axis 0 with size 25753
```

**问题原因**: `_panel_fold_backtest_stats` 函数中，`dp_val_series` 的索引与 `X_va` 的索引在某些 fold 切分后不完全对齐。

**修复状态**: 部分修复，但仍有边界情况未处理。

---

## 运行命令

### 基本运行（篮子模式）

```bash
# 快速测试（2 次迭代，2 折）
python main_cli.py --basket basket_1 --runs 2 --n_folds 2 --n_jobs 1

# 标准运行（50 次迭代，4 折）
python main_cli.py --basket basket_1 --runs 50 --n_folds 4

# 使用 holdout 验证
python main_cli.py --basket basket_1 --runs 50 --use_holdout 1

# 指定输出目录
python main_cli.py --basket basket_1 --output_dir ./my_output
```

### 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--basket` | Basket 目录名（必需） | - |
| `--data_dir` | 数据根目录 | `data` |
| `--runs` | 搜索迭代次数 | `50` |
| `--n_folds` | Walk-forward 折数 | `-1`（自动） |
| `--n_jobs` | 并行数（-1=自动） | `-1` |
| `--seed` | 随机种子 | `42` |
| `--portfolio_cash` | 初始资金 | `1000000` |
| `--top_k` | 最大持仓股票数 | `5` |
| `--rebalance_freq` | 调仓频率 | `weekly` |
| `--weighting` | 权重方案 | `equal` |
| `--use_holdout` | 使用 holdout 验证 | `1` |

---

## 当前状态

**项目可以成功跑通**，但在某些数据切分情况下会遇到索引对齐问题。建议：

1. **短期解决方案**: 使用更大的 `min_rows` 参数或更少的 `n_folds`
2. **长期修复**: 需要重构 `_panel_fold_backtest_stats` 函数的索引对齐逻辑

---

## 推荐的运行配置

```bash
# 稳定运行配置（推荐）
python main_cli.py --basket basket_1 \
    --runs 30 \
    --n_folds 3 \
    --n_jobs 1 \
    --seed 42 \
    --top_k 5 \
    --rebalance_freq weekly \
    --weighting equal \
    --use_holdout 1
```

---

**最后更新**: 2026-03-25
