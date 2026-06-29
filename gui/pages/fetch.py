"""数据获取页面。

提供单股和篮子数据获取功能，支持 7 维度筛选。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.log_panel import create_log_panel, stream_subprocess_output
from gui.state import app_state


# ── 辅助函数 ──────────────────────────────────────────────────


def _build_select_options(values: list) -> list[str]:
    """将 DimensionValue 列表转为下拉选项。"""
    options = ["全部"]
    for v in values:
        options.append(f"{v.code} {v.name} ({v.count}只)")
    return options


def _parse_dimension_code(selection: str) -> str | None:
    """从 'C27 医药制造业 (349只)' 中提取 'C27'。返回 None 表示全部。"""
    if not selection or selection == "全部":
        return None
    return selection.split()[0]


def _default_date_range() -> tuple[str, str]:
    """返回默认日期范围（6 年前到今天），格式 YYYYMMDD。"""
    today = datetime.now()
    six_years_ago = today - timedelta(days=365 * 6)
    return six_years_ago.strftime("%Y%m%d"), today.strftime("%Y%m%d")


# ── 页面 ─────────────────────────────────────────────────────


@ui.page("/fetch")
def fetch_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("数据获取").classes("text-h4")
        ui.label(
            "通过 QMT 获取股票历史数据。需要 XtMiniQMT 运行。"
        ).classes("text-grey-6")

        with ui.tabs() as tabs:
            tab_single = ui.tab("获取单股", icon="person")
            tab_basket = ui.tab("获取篮子", icon="group")

        with ui.tab_panels(tabs, value=tab_single).classes("w-full"):

            # ── Tab 1: 获取单股 ──
            with ui.tab_panel(tab_single):
                _create_single_tab()

            # ── Tab 2: 获取篮子 ──
            with ui.tab_panel(tab_basket):
                _create_basket_tab()


# ── 获取单股 Tab ──────────────────────────────────────────────


def _create_single_tab():
    """创建获取单股的表单。"""
    with ui.card().classes("w-full"):
        ui.label("获取单只股票数据").classes("text-h6")
        ui.separator()

        with ui.row().classes("gap-4 w-full"):
            code_input = ui.input(
                label="股票代码",
                placeholder="如: 000001",
            ).tooltip("6 位股票代码，无需后缀")

            format_select = ui.select(
                ["xlsx", "csv"],
                label="输出格式",
                value="xlsx",
            )

        # 日期范围
        default_start, default_end = _default_date_range()

        with ui.row().classes("gap-4 w-full"):
            start_input = ui.input(
                label="起始日期",
                value=default_start,
                placeholder="YYYYMMDD",
            )
            end_input = ui.input(
                label="结束日期",
                value=default_end,
                placeholder="YYYYMMDD",
            )
            output_input = ui.input(
                label="输出路径",
                value="data",
                placeholder="留空则自动存入 data/",
            )

    # 运行按钮
    run_button = ui.button(
        "▶ 获取数据", color="blue", icon="download"
    ).classes("text-white")
    is_running = {"value": False}

    async def on_run_single():
        # 【修复 #1】is_running 守卫：防止双击竞态
        if is_running["value"]:
            return
        code = (code_input.value or "").strip()
        if not code:
            ui.notify("请填写股票代码", type="warning")
            return
        # 补零到 6 位
        code = code.zfill(6)

        is_running["value"] = True
        run_button.disable()
        try:
            cmd = [
                sys.executable, "-m", "dpoint.cli.main",
                "fetch", "single",
                "--code", code,
                "--start", (start_input.value or "").strip(),
                "--end", (end_input.value or "").strip(),
                "--output", (output_input.value or "data").strip(),
                "--format", format_select.value or "xlsx",
            ]

            log, status_label, progress = create_log_panel(f"获取 {code}")
            ui.notify(f"开始获取 {code}...", type="info")

            returncode = await stream_subprocess_output(cmd, log, status_label, progress)

            if returncode == 0:
                ui.notify("获取完成！", type="positive")
            else:
                ui.notify(f"获取失败，退出码: {returncode}", type="negative")
        finally:
            is_running["value"] = False
            run_button.enable()

    run_button.on_click(on_run_single)


# ── 获取篮子 Tab ──────────────────────────────────────────────


def _create_basket_tab():
    """创建获取篮子的表单（7 维度筛选）。"""
    try:
        from dpoint.data.fetch.industry import IndustryDB
        db = IndustryDB()
    except FileNotFoundError:
        ui.label("⚠ 行业分类数据库不存在").classes("text-h6 text-red")
        ui.label("请先运行: python scripts/build_industry_db.py").classes("text-grey-6")
        return

    # 不在此处关闭 db —— UI 回调需要持续使用它。
    # NiceGUI 页面销毁时 Python GC 会回收连接。
    _build_basket_form(db)


def _build_basket_form(db):
    """构建篮子获取表单（从 _create_basket_tab 拆分，便于管理生命周期）。"""
    with ui.card().classes("w-full"):
        ui.label("筛选条件").classes("text-h6")
        ui.separator()

        # 加载各维度可选值
        ind1_values = db.list_values("ind1")
        ind2_values = db.list_values("ind2")
        ind3_values = db.list_values("ind3")
        ind4_values = db.list_values("ind4")
        province_values = db.list_values("province")
        city_values = db.list_values("city")
        ownership_values = db.list_values("ownership")

        with ui.row().classes("gap-4 w-full"):
            ind1_select = ui.select(
                _build_select_options(ind1_values),
                label="一级行业",
                value="全部",
            )
            ind2_select = ui.select(
                _build_select_options(ind2_values),
                label="二级行业",
                value="全部",
            )
            ind3_select = ui.select(
                _build_select_options(ind3_values),
                label="三级行业",
                value="全部",
            )
            ind4_select = ui.select(
                _build_select_options(ind4_values),
                label="四级行业（中信）",
                value="全部",
            )

        with ui.row().classes("gap-4 w-full"):
            province_select = ui.select(
                _build_select_options(province_values),
                label="省份",
                value="全部",
            )
            city_select = ui.select(
                _build_select_options(city_values),
                label="城市",
                value="全部",
            )
            ownership_select = ui.select(
                _build_select_options(ownership_values),
                label="所有权",
                value="全部",
            )

    # 预览区
    with ui.card().classes("w-full"):
        preview_label = ui.label("请选择筛选条件").classes("text-body1")

        # 初始预览
        _update_preview(preview_label, db)

        # 筛选条件变化时更新预览
        def on_filter_change():
            _update_preview(
                preview_label, db,
                ind1=_parse_dimension_code(ind1_select.value),
                ind2=_parse_dimension_code(ind2_select.value),
                ind3=_parse_dimension_code(ind3_select.value),
                ind4=_parse_dimension_code(ind4_select.value),
                province=_parse_dimension_code(province_select.value),
                city=_parse_dimension_code(city_select.value),
                ownership=_parse_dimension_code(ownership_select.value),
            )

        for select in [ind1_select, ind2_select, ind3_select, ind4_select,
                        province_select, city_select, ownership_select]:
            select.on_value_change(on_filter_change)

    # 日期和输出
    # 【修复 #8】使用共享的 _default_date_range() 函数
    default_start, default_end = _default_date_range()

    with ui.card().classes("w-full"):
        with ui.row().classes("gap-4 w-full"):
            start_input = ui.input(
                label="起始日期",
                value=default_start,
                placeholder="YYYYMMDD",
            )
            end_input = ui.input(
                label="结束日期",
                value=default_end,
                placeholder="YYYYMMDD",
            )
            output_input = ui.input(
                label="输出目录",
                value="",
                placeholder="留空自动生成 data/basket_{筛选条件}",
            )
            format_select = ui.select(
                ["csv", "xlsx"],
                label="输出格式",
                value="csv",
            )

    # 运行按钮
    run_button = ui.button(
        "▶ 获取数据", color="green", icon="download"
    ).classes("text-white")
    is_running = {"value": False}

    async def on_run_basket():
        # 【修复 #2】is_running 守卫：防止双击竞态
        if is_running["value"]:
            return
        # 构建筛选参数
        args = []
        ind1 = _parse_dimension_code(ind1_select.value)
        ind2 = _parse_dimension_code(ind2_select.value)
        ind3 = _parse_dimension_code(ind3_select.value)
        ind4 = _parse_dimension_code(ind4_select.value)
        province = _parse_dimension_code(province_select.value)
        city = _parse_dimension_code(city_select.value)
        ownership = _parse_dimension_code(ownership_select.value)

        if ind1:
            args += ["--ind1", ind1]
        if ind2:
            args += ["--ind2", ind2]
        if ind3:
            args += ["--ind3", ind3]
        if ind4:
            args += ["--ind4", ind4]
        if province:
            args += ["--province", province]
        if city:
            args += ["--city", city]
        if ownership:
            args += ["--ownership", ownership]

        if not any([ind1, ind2, ind3, ind4, province, city, ownership]):
            ui.notify("请至少选择一个筛选条件", type="warning")
            return

        is_running["value"] = True
        run_button.disable()
        try:
            cmd = [
                sys.executable, "-m", "dpoint.cli.main",
                "fetch", "basket",
                *args,
                "--start", (start_input.value or "").strip(),
                "--end", (end_input.value or "").strip(),
                "--format", format_select.value or "csv",
            ]
            # 仅在用户指定了自定义输出目录时传递 --output
            custom_output = (output_input.value or "").strip()
            if custom_output:
                cmd += ["--output", custom_output]

            log, status_label, progress = create_log_panel("获取篮子")
            ui.notify("开始获取篮子数据...", type="info")

            returncode = await stream_subprocess_output(cmd, log, status_label, progress)

            if returncode == 0:
                ui.notify("获取完成！", type="positive")
            else:
                ui.notify(f"获取失败，退出码: {returncode}", type="negative")
        finally:
            is_running["value"] = False
            run_button.enable()

    run_button.on_click(on_run_basket)


def _update_preview(preview_label, db, **filters):
    """更新预览区的股票数量。"""
    # 过滤掉 None 值
    active_filters = {k: v for k, v in filters.items() if v is not None}
    try:
        codes = db.query_stocks(**active_filters)
        if active_filters:
            preview_label.text = f"共 {len(codes)} 只股票"
        else:
            preview_label.text = f"全部 {len(codes)} 只股票（未设置筛选条件）"
        if codes:
            preview_label.text += f"  |  前 5 只: {', '.join(codes[:5])}"
    except Exception as e:
        # 【修复 #5】记录错误日志，便于调试
        import logging
        logging.getLogger(__name__).warning("预览查询失败: %s", e)
        preview_label.text = f"查询出错: {e}"
