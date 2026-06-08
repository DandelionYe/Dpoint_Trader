"""ECharts 图表组件工厂。

提供权益曲线、回撤、IC 等常用图表。
"""

from __future__ import annotations

from nicegui import ui


def _base_options(
    dates: list[str],
    title: str,
    y_axis_name: str,
    *,
    grid_bottom: str = "15%",
    tooltip_formatter: str | None = None,
    y_axis_inverse: bool = False,
    has_data_zoom: bool = True,
    legend: bool = False,
) -> dict:
    """构建 ECharts 通用选项（标题、坐标轴、网格、缩放）。"""
    tooltip: dict = {"trigger": "axis"}
    if tooltip_formatter:
        tooltip["formatter"] = tooltip_formatter

    y_axis: dict = {"type": "value", "name": y_axis_name}
    if y_axis_inverse:
        y_axis["inverse"] = True

    options: dict = {
        "title": {"text": title, "left": "center"},
        "tooltip": tooltip,
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLabel": {"rotate": 45},
        },
        "yAxis": y_axis,
        "grid": {"left": "10%", "right": "5%", "bottom": grid_bottom},
    }

    if legend:
        options["legend"] = {"bottom": 0}

    if has_data_zoom:
        options["dataZoom"] = [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100},
        ]

    return options


def equity_curve_chart(dates: list[str], equity: list[float], title: str = "权益曲线") -> ui.echart:
    """绘制权益曲线折线图。"""
    options = _base_options(dates, title, "净值")
    options["series"] = [
        {
            "type": "line",
            "data": equity,
            "smooth": True,
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.15},
            "itemStyle": {"color": "#1976D2"},
        }
    ]
    return ui.echart(options).classes("w-full h-96")


def drawdown_chart(dates: list[str], drawdown: list[float], title: str = "回撤曲线") -> ui.echart:
    """绘制回撤曲线。"""
    options = _base_options(
        dates, title, "回撤 (%)",
        tooltip_formatter="{b}<br/>回撤: {c}%",
        y_axis_inverse=True,
    )
    options["series"] = [
        {
            "type": "line",
            "data": drawdown,
            "smooth": True,
            "lineStyle": {"width": 2, "color": "#C10015"},
            "areaStyle": {"opacity": 0.3, "color": "#C10015"},
            "itemStyle": {"color": "#C10015"},
        }
    ]
    return ui.echart(options).classes("w-full h-72")


def ic_bar_chart(dates: list[str], ic_values: list[float], title: str = "IC 序列") -> ui.echart:
    """绘制 IC 柱状图。"""
    options = _base_options(dates, title, "IC")
    options["series"] = [
        {
            "type": "bar",
            "data": ic_values,
            "itemStyle": {
                "color": {
                    "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": "#21BA45"},
                        {"offset": 1, "color": "#C10015"},
                    ],
                }
            },
        }
    ]
    return ui.echart(options).classes("w-full h-72")


def layered_return_chart(
    layers: list[str], returns: list[float], title: str = "分层收益"
) -> ui.echart:
    """绘制分层收益柱状图。"""
    options = _base_options(
        layers, title, "年化收益 (%)",
        tooltip_formatter="{b}<br/>收益: {c}%",
        grid_bottom="10%",
        has_data_zoom=False,
    )
    options["series"] = [
        {
            "type": "bar",
            "data": returns,
            "itemStyle": {"color": "#1976D2"},
        }
    ]
    return ui.echart(options).classes("w-full h-72")


def comparison_equity_chart(
    dates: list[str],
    curves: dict[str, list[float]],
    title: str = "权益曲线对比",
) -> ui.echart:
    """绘制多实验叠加权益曲线。"""
    colors = ["#1976D2", "#C10015", "#21BA45", "#FF9800", "#9C27B0", "#00BCD4"]
    series = [
        {
            "type": "line",
            "name": name,
            "data": data,
            "smooth": True,
            "lineStyle": {"width": 2},
            "itemStyle": {"color": colors[i % len(colors)]},
        }
        for i, (name, data) in enumerate(curves.items())
    ]

    options = _base_options(dates, title, "净值", legend=True)
    options["series"] = series
    return ui.echart(options).classes("w-full h-96")
