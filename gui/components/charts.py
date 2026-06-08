"""ECharts 图表组件工厂。

提供权益曲线、回撤、IC 等常用图表。
"""

from __future__ import annotations

from nicegui import ui


def equity_curve_chart(dates: list[str], equity: list[float], title: str = "权益曲线") -> ui.echart:
    """绘制权益曲线折线图。"""
    options = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLabel": {"rotate": 45},
        },
        "yAxis": {"type": "value", "name": "净值"},
        "series": [
            {
                "type": "line",
                "data": equity,
                "smooth": True,
                "lineStyle": {"width": 2},
                "areaStyle": {"opacity": 0.15},
                "itemStyle": {"color": "#1976D2"},
            }
        ],
        "grid": {"left": "10%", "right": "5%", "bottom": "15%"},
        "dataZoom": [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100},
        ],
    }
    return ui.echart(options).classes("w-full h-96")


def drawdown_chart(dates: list[str], drawdown: list[float], title: str = "回撤曲线") -> ui.echart:
    """绘制回撤曲线。"""
    options = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis", "formatter": "{b}<br/>回撤: {c}%"},
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLabel": {"rotate": 45},
        },
        "yAxis": {"type": "value", "name": "回撤 (%)", "inverse": True},
        "series": [
            {
                "type": "line",
                "data": drawdown,
                "smooth": True,
                "lineStyle": {"width": 2, "color": "#C10015"},
                "areaStyle": {"opacity": 0.3, "color": "#C10015"},
                "itemStyle": {"color": "#C10015"},
            }
        ],
        "grid": {"left": "10%", "right": "5%", "bottom": "15%"},
        "dataZoom": [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100},
        ],
    }
    return ui.echart(options).classes("w-full h-72")


def ic_bar_chart(dates: list[str], ic_values: list[float], title: str = "IC 序列") -> ui.echart:
    """绘制 IC 柱状图。"""
    colors = ["#21BA45" if v >= 0 else "#C10015" for v in ic_values]
    options = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLabel": {"rotate": 45},
        },
        "yAxis": {"type": "value", "name": "IC"},
        "series": [
            {
                "type": "bar",
                "data": ic_values,
                "itemStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                        "colorStops": [
                                            {"offset": 0, "color": "#21BA45"},
                                            {"offset": 1, "color": "#C10015"},
                                        ]}},
            }
        ],
        "grid": {"left": "10%", "right": "5%", "bottom": "15%"},
    }
    return ui.echart(options).classes("w-full h-72")


def layered_return_chart(
    layers: list[str], returns: list[float], title: str = "分层收益"
) -> ui.echart:
    """绘制分层收益柱状图。"""
    colors = ["#21BA45" if v >= 0 else "#C10015" for v in returns]
    options = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis", "formatter": "{b}<br/>收益: {c}%"},
        "xAxis": {"type": "category", "data": layers},
        "yAxis": {"type": "value", "name": "年化收益 (%)"},
        "series": [
            {
                "type": "bar",
                "data": returns,
                "itemStyle": {"color": "#1976D2"},
            }
        ],
        "grid": {"left": "10%", "right": "5%", "bottom": "10%"},
    }
    return ui.echart(options).classes("w-full h-72")


def comparison_equity_chart(
    dates: list[str],
    curves: dict[str, list[float]],
    title: str = "权益曲线对比",
) -> ui.echart:
    """绘制多实验叠加权益曲线。"""
    colors = ["#1976D2", "#C10015", "#21BA45", "#FF9800", "#9C27B0", "#00BCD4"]
    series = []
    for i, (name, data) in enumerate(curves.items()):
        series.append(
            {
                "type": "line",
                "name": name,
                "data": data,
                "smooth": True,
                "lineStyle": {"width": 2},
                "itemStyle": {"color": colors[i % len(colors)]},
            }
        )

    options = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLabel": {"rotate": 45},
        },
        "yAxis": {"type": "value", "name": "净值"},
        "series": series,
        "grid": {"left": "10%", "right": "5%", "bottom": "15%"},
        "dataZoom": [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100},
        ],
    }
    return ui.echart(options).classes("w-full h-96")
