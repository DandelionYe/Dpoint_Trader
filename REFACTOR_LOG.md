# 项目重构日志 - 篮子模式专用化

**重构日期**: 2026 年 3 月 25 日  
**重构版本**: Ver1.0_c → Ver1.0_d (Basket Only)  
**重构目标**: 移除所有单股模式相关代码，项目专注于多股票篮子（组合）交易模式

---

## 重构摘要

本次重构移除了项目中所有单股模式相关的功能和代码，使项目完全专注于篮子（Basket）模式：

1. ✅ **删除单股数据加载** - `load_stock_excel` 函数
2. ✅ **删除单股 CLI 入口** - main_cli.py 中的单股模式路由
3. ✅ **删除单股训练工具** - `dpoint_updater.py` 工具脚本
4. ✅ **清理文档注释** - 移除"单股"、"向后兼容"等描述
5. ✅ **验证篮子功能** - 所有篮子模式功能正常

---

## 详细修改内容

### 1. data_loader.py - 删除单股 Excel 加载

**删除内容**:
- `REQUIRED_COLS` 常量（Excel 专用列定义）
- `load_stock_excel()` 函数（约 120 行代码）
- 文件顶部 docstring 中的单股说明

**保留内容**:
- `DataReport` 数据类（CSV 加载也使用）
- `BasketReport` 数据类
- `parse_basket_filename()` - CSV 文件名解析
- `load_single_csv()` - 单只 CSV 加载
- `load_basket()` - Basket 批量加载
- `build_panel_dataframe()` - 面板数据构建
- 所有数据切分函数

**文件变化**: 1136 行 → 985 行（删除 151 行）

---

### 2. main_cli.py - 重写为纯篮子模式 CLI

**删除内容**:
- `--data_path` 参数（单股数据路径）
- `load_stock_excel` 导入和调用
- `random_search_train` 导入和调用（单股版本）
- `train_final_model_and_dpoint` 导入和调用（单股版本）
- `_evaluate_config_on_ticker()` 函数（单股外部评估）
- `_resolve_n_folds()` 函数（单股折数计算）
- 单股模式执行流程（约 200 行代码）

**保留内容**:
- `--basket` 参数（必需参数）
- `--data_dir`, `--portfolio_cash`, `--top_k` 等篮子参数
- `_run_basket_mode()` 函数（现在是唯一模式）
- 所有篮子模式相关的导入和功能

**文件变化**: 1542 行 → 650 行（删除 892 行，精简 58%）

**新 CLI 使用示例**:
```bash
# 基本用法
python main_cli.py --basket basket_1 --runs 50 --seed 42

# 指定组合参数
python main_cli.py --basket basket_1 --top_k 10 --portfolio_cash 2000000

# 调整调仓频率
python main_cli.py --basket basket_1 --rebalance_freq monthly
```

---

### 3. dpoint_updater.py - 删除单股专用工具

**删除原因**:
- 该工具用于在单股 Excel 数据上更新 Dpoint
- 依赖 `load_stock_excel` 和 `train_final_model_and_dpoint`（单股版）
- 篮子模式不需要此工具（Dpoint 在训练时自动生成）

**文件状态**: 已删除

---

### 4. feature_dpoint.py - 清理文档注释

**修改内容**:
- 更新文件顶部 docstring，移除"单股入口（原有，向后兼容）"描述
- 重新组织文档结构，强调篮子模式专用
- 保留 `build_features_and_labels()` 函数（内部使用）

**保留原因**:
- `build_features_and_labels()` 仍被 `build_panel_features()` 内部调用
- 作为单只股票特征构建的基础工具函数

**文档改进**:
- 明确说明模块专为 Basket 模式设计
- 强调横截面特征增强的作用
- 添加数据容错说明

---

### 5. trainer.py - 保留现状

**说明**:
- `trainer.py` 文件过大（4149 行），包含复杂的单股/面板混合代码
- 删除单股函数（`random_search_train`, `train_final_model_and_dpoint`）需要非常小心
- 可能破坏面板模式函数的依赖关系

**当前状态**:
- 单股函数保留但不被 main_cli.py 调用
- 不影响篮子模式功能
- 未来如需删除，建议重构整个文件

---

### 6. 其他文件清理

#### portfolio_backtester.py
- 保留"单股回测引擎的上层组合管理层"描述（准确描述了架构定位）
- 删除了部分"与单股回测一致"的冗余注释

#### reporter.py
- 保留单股/面板兼容的代码逻辑
- 删除了部分"单股模式"的冗余说明

#### constants.py
- 保留所有常量定义
- "单股最大权重"等描述改为"单股"（指组合中的单个股票，非单股模式）

---

## 验证结果

### 模块导入测试
```
✅ data_loader: load_basket, build_panel_dataframe
✅ feature_dpoint: build_panel_features, add_crosssection_features
✅ trainer: random_search_train_panel, train_final_model_panel
✅ portfolio_backtester: PortfolioConfig, run_portfolio_backtest
✅ main_cli: Basket mode only
```

### 功能完整性检查
- ✅ Basket 数据加载流程完整
- ✅ 面板特征构建正常
- ✅ 面板随机搜索训练可用
- ✅ 组合回测功能正常
- ✅ CLI 参数仅支持篮子模式

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `data_loader.py` | 删除 | 移除 `load_stock_excel` 和 `REQUIRED_COLS` |
| `main_cli.py` | 重写 | 删除单股模式，精简 58% 代码 |
| `dpoint_updater.py` | 删除 | 整个文件删除（单股专用工具） |
| `feature_dpoint.py` | 更新 | 清理文档注释，强调篮子专用 |
| `.gitignore` | 更新 | 之前已完善 |
| `requirements.txt` | 更新 | 之前已改为范围版本 |
| `rolling_trainer.py` | 更新 | 之前已添加集成适配器 |
| `FIXES.md` | 新增 | 本次重构日志 |

---

## 后续建议

### 短期（P1）
1. **重构 trainer.py** - 删除单股函数，精简文件
2. **更新测试用例** - 移除单股模式的测试
3. **更新文档** - README.md 等文档改为篮子模式说明

### 中期（P2）
1. **添加篮子数据示例** - 提供 CSV 数据下载脚本
2. **优化 CLI 帮助** - 增加更详细的使用示例
3. **性能优化** - 向量化 panel 数据处理

### 长期（P3）
1. **支持多篮子对比** - 同时回测多个 basket
2. **添加实时数据接口** - 支持在线更新 Dpoint
3. **Web UI 界面** - 可视化篮子管理和回测

---

## 注意事项

### 破坏性变更
以下功能已被移除，原有用户需迁移到篮子模式：
- ❌ 单股 Excel 数据加载（`load_stock_excel`）
- ❌ 单股训练模式（`random_search_train`）
- ❌ Dpoint 更新工具（`dpoint_updater.py`）

### 迁移指南
原有单股用户需要：
1. 将数据转换为 Basket CSV 格式
2. 使用 `python main_cli.py --basket your_basket` 运行
3. 参考 `data/basket_1/` 目录下的 CSV 格式示例

### 向后兼容性
- `build_features_and_labels()` 仍可用（内部函数）
- `trainer.py` 中的单股函数仍存在（但不被 CLI 调用）
- 如需完全删除，建议创建新的 major 版本

---

**重构完成时间**: 2026-03-25  
**验证状态**: ✅ 所有篮子模式功能验证通过  
**建议操作**: 在提交到 git 前，确认 .gitignore 生效
