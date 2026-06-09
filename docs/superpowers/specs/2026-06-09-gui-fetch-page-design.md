# GUI 数据获取页面设计

**日期**: 2026-06-09
**状态**: 已批准
**范围**: 为 GUI 新增数据获取页面，支持单股和篮子数据获取

## 背景

CLI 已支持 `dpoint fetch single/basket` 命令，包括 7 维度篮子筛选。但 GUI 中没有数据获取页面，用户必须通过命令行操作。需要在 NiceGUI 界面中提供同等功能。

## 页面结构

**路由**: `/fetch`
**布局**: 单页面 + Tab 切换（获取单股 / 获取篮子）

### Tab 1: 获取单股

- 股票代码输入框（6 位代码）
- 起始日期（默认 6 年前）
- 结束日期（默认今天）
- 输出路径（默认 `data/`）
- 格式选择（xlsx/csv）
- 「获取数据」按钮
- 日志面板（实时显示进度）

### Tab 2: 获取篮子

**筛选条件区**（7 个下拉框）：
- 一级行业、二级行业、三级行业、四级行业（中信）
- 省份、城市、所有权
- 每个下拉第一项为「全部」（不筛选该维度）

**预览区**：
- 显示「共 XXX 只股票」
- 显示前 5 只股票代码

**其他参数**：
- 起始日期、结束日期
- 输出目录
- 格式选择

**操作按钮**：
- 「获取数据」按钮
- 日志面板

## 数据流

### 列表查询（直接 API 调用）

页面加载时直接调用 `IndustryDB` API 填充下拉框：
```python
from dpoint.data.fetch.industry import IndustryDB

with IndustryDB() as db:
    values = db.list_values("ind4")  # 返回 list[DimensionValue]
```

每个下拉选项格式：`"代码 名称 (N只)"`，第一项为「全部」。

用户切换筛选条件时，实时调用 `db.query_stocks(**filters)` 更新预览数量。防抖 300ms。

### 数据获取（子进程调用）

构建 CLI 参数，通过 `run_experiment_subprocess()` 执行：
```
dpoint fetch single --code 000001 --start 20200101 --end 20260609 --output data/ --format xlsx
dpoint fetch basket --ind4 C27 --province 广东省 --start 20200101 --output data/ --format csv
```

日志面板实时显示获取进度。

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `gui/pages/fetch.py` | 新建 | 数据获取页面 |
| `gui/app.py` | 修改 | 导入 fetch 模块 |
| `gui/components/layout.py` | 修改 | 侧边栏添加导航链接 |
| `gui/pages/dashboard.py` | 修改 | 快捷操作添加按钮（可选） |

## UI 组件细节

### 下拉框填充逻辑

```python
def _build_select_options(values: list[DimensionValue]) -> list[str]:
    """将 DimensionValue 列表转为下拉选项。"""
    options = ["全部"]
    for v in values:
        options.append(f"{v.code} {v.name} ({v.count}只)")
    return options
```

### 筛选条件解析

从下拉选项文本中提取代码：
```python
def _parse_dimension_code(selection: str) -> str | None:
    """从 'C27 医药制造业 (349只)' 中提取 'C27'。"""
    if selection == "全部":
        return None
    return selection.split()[0]
```

### 预览更新

```python
async def update_preview(db, **filters):
    """更新预览区的股票数量。"""
    codes = db.query_stocks(**filters)
    preview_label.text = f"共 {len(codes)} 只股票"
    if codes:
        preview_label.text += f"\n前 5 只: {', '.join(codes[:5])}"
```

## 设计约束

- **不阻塞 GUI 线程**：列表查询在 `on_connect` 回调中执行，数据获取通过子进程
- **IndustryDB 生命周期**：页面加载时创建，通过 `ui.context.client.on_disconnect` 关闭
- **与现有模式一致**：数据获取复用 `run_experiment_subprocess()` 模式
- **错误处理**：数据库不存在时显示友好提示，引导用户运行构建脚本
