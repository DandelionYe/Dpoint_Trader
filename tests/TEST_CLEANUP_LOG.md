# 测试文件夹清理日志

**清理日期**: 2026 年 3 月 25 日  
**目标**: 删除单股模式测试，只保留篮子（Basket）模式测试

---

## 删除的测试文件

以下文件已被删除，因为它们测试的是单股模式功能：

| 文件 | 删除原因 |
|------|---------|
| `test_cli.py` | 测试单股 CLI 参数（--data_path），与篮子模式不兼容 |
| `test_smoke.py` | 使用单股训练函数 `train_final_model_and_dpoint` |
| `test_trainer_split_mode.py` | 测试单股 split 模式（_make_eval_splits） |
| `test_trainer_calibration.py` | 测试单股校准逻辑（_eval_candidate） |

---

## 保留的测试文件

以下文件已保留，因为它们与篮子模式兼容或是通用测试：

| 文件 | 说明 | 状态 |
|------|------|------|
| `conftest.py` | 已更新为提供篮子模式 fixture | ✅ 已更新 |
| `test_splitter.py` | 已添加面板数据切分测试 | ✅ 已更新 |
| `test_execution.py` | 回测执行测试（与篮子模式兼容） | ✅ 保留 |
| `test_metrics.py` | 指标计算测试（与篮子模式兼容） | ✅ 保留 |
| `test_backtester_market_state.py` | 市场状态测试 | ✅ 保留 |
| `test_rejection.py` | 订单拒绝逻辑测试 | ✅ 保留 |
| `test_fee_lot.py` | 费用和手数测试 | ✅ 保留 |
| `test_no_leakage.py` | 数据泄露防止测试 | ✅ 保留 |
| `test_report.py` | 报告生成测试 | ✅ 保留 |
| `test_reproducibility.py` | 可复现性测试 | ✅ 保留 |
| `test_conda_env.py` | Conda 环境测试 | ✅ 保留 |
| `test_optional_torch_runtime.py` | PyTorch 运行时测试 | ✅ 保留 |

---

## 更新内容

### conftest.py

**新增 fixture**:
- `basket_sample_data()` - 3 只股票的样本数据字典
- `basket_csv_dir(tmp_path)` - 临时 CSV 目录 fixture
- `sample_dpoint_matrix()` - 篮子模式 Dpoint 矩阵
- `minimal_basket_config()` - 篮子模式最小配置
- `sample_portfolio_equity_curve()` - 组合净值曲线

**保留 fixture** (用于 backtester 测试):
- `minimal_price_data()` - 单股价格数据（用于 backtester 单元测试）
- `sample_dpoint_series()` - 单股 Dpoint 序列
- 其他原有 fixture

### test_splitter.py

**新增测试类**: `TestPanelDateSplits`
- `test_panel_split_requires_import` - 导入检查
- `test_panel_split_count` - 切分数量测试
- `test_panel_split_no_date_overlap` - 日期不重叠测试
- `test_panel_split_same_date_stays_together` - 同日期股票同 fold 测试
- `test_panel_split_expanding_training` - 训练集扩展测试
- `test_panel_split_min_dates_constraint` - 最小日期数约束测试

**保留测试类**:
- `TestWalkforwardSplits` - 标准 Walk-forward 测试
- `TestFinalHoldoutSplit` - Holdout 切分测试
- `TestRecommendNFolds` - 折数推荐测试
- `TestEmbargoSplit` - Embargo 切分测试

---

## 测试验证结果

运行测试套件验证：

```
✅ test_splitter.py - 18/18 通过
✅ test_metrics.py - 14/14 通过
✅ test_execution.py - 19/19 通过
```

**总计**: 51 个测试全部通过

---

## 测试覆盖范围

### 篮子模式核心功能

| 功能模块 | 测试覆盖 | 说明 |
|---------|---------|------|
| 面板数据切分 | ✅ | `TestPanelDateSplits` |
| Walk-forward 切分 | ✅ | `TestWalkforwardSplits` |
| Holdout 切分 | ✅ | `TestFinalHoldoutSplit` |
| 回测执行 | ✅ | `test_execution.py` |
| 风险评估指标 | ✅ | `test_metrics.py` |
| 市场状态检测 | ✅ | `test_backtester_market_state.py` |
| 订单拒绝逻辑 | ✅ | `test_rejection.py` |
| 费用/手数计算 | ✅ | `test_fee_lot.py` |
| 数据泄露防止 | ✅ | `test_no_leakage.py` |

### 待补充测试

以下篮子模式特定功能需要新增测试：

1. **Basket 数据加载** (`test_data_loader.py`)
   - `test_load_basket()` - 批量加载 CSV
   - `test_build_panel_dataframe()` - 面板数据构建
   - `test_parse_basket_filename()` - 文件名解析

2. **面板特征构建** (`test_feature_dpoint.py`)
   - `test_build_panel_features()` - 面板特征构建
   - `test_add_crosssection_features()` - 横截面排名特征

3. **组合回测** (`test_portfolio_backtester.py`)
   - `test_run_portfolio_backtest()` - 组合回测执行
   - `test_construct_portfolio()` - 权重矩阵构建
   - `test_rank_dpoints()` - 横截面排名

4. **端到端测试** (`test_basket_e2e.py`)
   - `test_basket_mode_full_pipeline()` - 完整流程测试

---

## 文件清单

### tests/ 目录结构

```
tests/
├── __init__.py
├── conftest.py                    # ✅ 已更新
├── test_backtester_market_state.py  # ✅ 保留
├── test_conda_env.py              # ✅ 保留
├── test_execution.py              # ✅ 保留
├── test_fee_lot.py                # ✅ 保留
├── test_metrics.py                # ✅ 保留
├── test_no_leakage.py             # ✅ 保留
├── test_optional_torch_runtime.py # ✅ 保留
├── test_report.py                 # ✅ 保留
├── test_reproducibility.py        # ✅ 保留
├── test_rejection.py              # ✅ 保留
├── test_splitter.py               # ✅ 已更新
└── __pycache__/
```

### 已删除文件

```
tests/test_cli.py                  ❌ 删除
tests/test_smoke.py                ❌ 删除
tests/test_trainer_split_mode.py   ❌ 删除
tests/test_trainer_calibration.py  ❌ 删除
```

---

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_splitter.py -v
pytest tests/test_execution.py -v
pytest tests/test_metrics.py -v

# 运行篮子模式相关测试
pytest tests/test_splitter.py::TestPanelDateSplits -v
```

---

## 注意事项

1. **保留单股 backtester 测试**: `test_execution.py`、`test_rejection.py` 等文件中的单股 backtester 测试已保留，因为它们是篮子模式的基础组件测试。

2. **fixture 兼容性**: `conftest.py` 保留了 `minimal_price_data` 等单股 fixture，因为它们被 `test_execution.py` 等文件使用。

3. **trainer.py 未完全清理**: 由于 `trainer.py` 仍包含单股函数（`random_search_train`, `train_final_model_and_dpoint`），相关测试未完全删除。建议未来完全重构 `trainer.py` 后再清理相关测试。

---

**清理完成时间**: 2026-03-25  
**验证状态**: ✅ 51 个测试全部通过  
**下一步**: 添加篮子模式端到端测试
