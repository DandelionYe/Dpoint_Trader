"""通用配置表单组件。

为 FeatureConfig、ModelConfig、TradeConfig、SearchConfig、
SplitConfig、PortfolioConfig 提供可视化表单。
"""

from __future__ import annotations

from nicegui import ui


def feature_config_form(defaults: dict | None = None) -> dict:
    """特征工程配置表单。返回字段值字典。"""
    d = defaults or {}

    with ui.card().classes("w-full"):
        ui.label("特征工程配置").classes("text-h6")
        ui.separator()

        with ui.row().classes("gap-4 w-full"):
            windows = ui.input(
                label="滚动窗口大小",
                value=str(d.get("windows", [5, 10, 20])),
                placeholder="如: 5, 10, 20",
            ).tooltip("逗号分隔的整数列表，用于计算滚动特征")

            vol_metric = ui.select(
                ["std", "mad"],
                label="波动率度量",
                value=d.get("vol_metric", "std"),
            ).tooltip("std=标准差, mad=中位绝对偏差")

            liq_transform = ui.select(
                ["ratio", "zscore"],
                label="流动性变换",
                value=d.get("liq_transform", "ratio"),
            ).tooltip("ratio=比率, zscore=标准化")

        with ui.row().classes("gap-4 w-full"):
            use_momentum = ui.switch("动量特征", value=d.get("use_momentum", True))
            use_volatility = ui.switch("波动率特征", value=d.get("use_volatility", True))
            use_volume = ui.switch("成交量特征", value=d.get("use_volume", True))

        with ui.row().classes("gap-4 w-full"):
            use_candle = ui.switch("K线形态特征", value=d.get("use_candle", True))
            use_turnover = ui.switch("换手率特征", value=d.get("use_turnover", True))
            use_ta_indicators = ui.switch("技术指标(RSI/MACD等)", value=d.get("use_ta_indicators", True))

        include_cross_section = ui.switch(
            "截面排名特征（篮子模式）",
            value=d.get("include_cross_section", True),
        ).tooltip("仅篮子模式有效，计算跨股票的排名特征")

    return {
        "windows": windows,
        "use_momentum": use_momentum,
        "use_volatility": use_volatility,
        "use_volume": use_volume,
        "use_candle": use_candle,
        "use_turnover": use_turnover,
        "use_ta_indicators": use_ta_indicators,
        "include_cross_section": include_cross_section,
        "vol_metric": vol_metric,
        "liq_transform": liq_transform,
    }


def model_config_form(defaults: dict | None = None) -> dict:
    """模型配置表单。"""
    d = defaults or {}

    with ui.card().classes("w-full"):
        ui.label("模型配置").classes("text-h6")
        ui.separator()

        model_type = ui.select(
            ["logreg", "sgd", "xgb", "mlp", "lstm", "gru", "cnn", "transformer"],
            label="模型类型",
            value=d.get("model_type", "logreg"),
        ).tooltip("选择搜索使用的模型类型")

        # 深度学习参数
        with ui.expansion("深度学习参数", icon="psychology").classes("w-full"):
            with ui.row().classes("gap-4 w-full"):
                hidden_dim = ui.number(
                    "隐藏层维度", value=d.get("hidden_dim", 64), min=8, max=512, step=8
                )
                num_layers = ui.number(
                    "层数", value=d.get("num_layers", 2), min=1, max=8, step=1
                )
                dropout_rate = ui.number(
                    "Dropout 率", value=d.get("dropout_rate", 0.3), min=0.0, max=0.9, step=0.05,
                    format="%.2f",
                )

            with ui.row().classes("gap-4 w-full"):
                learning_rate = ui.number(
                    "学习率", value=d.get("learning_rate", 0.001), min=0.00001, max=0.1, step=0.0001,
                    format="%.5f",
                )
                batch_size = ui.number(
                    "批大小", value=d.get("batch_size", 256), min=16, max=2048, step=16
                )
                epochs = ui.number(
                    "最大轮数", value=d.get("epochs", 100), min=10, max=1000, step=10
                )
                patience = ui.number(
                    "早停轮数", value=d.get("patience", 10), min=3, max=100, step=1
                )

            with ui.row().classes("gap-4 w-full"):
                seq_len = ui.number(
                    "序列长度", value=d.get("seq_len", 10), min=2, max=100, step=1
                ).tooltip("LSTM/GRU/CNN/Transformer 的输入序列长度")
                bidirectional = ui.switch(
                    "双向 LSTM/GRU", value=d.get("bidirectional", False)
                )

        # XGBoost 参数
        with ui.expansion("XGBoost 参数", icon="forest").classes("w-full"):
            with ui.row().classes("gap-4 w-full"):
                n_estimators = ui.number(
                    "树数量", value=d.get("n_estimators", 200), min=10, max=2000, step=10
                )
                max_depth = ui.number(
                    "最大深度", value=d.get("max_depth", 3), min=1, max=20, step=1
                )

        # SKLearn 参数
        with ui.expansion("SKLearn 参数", icon="functions").classes("w-full"):
            with ui.row().classes("gap-4 w-full"):
                C = ui.number(
                    "正则化系数 C", value=d.get("C", 1.0), min=0.001, max=100.0, step=0.1,
                    format="%.3f",
                )
                penalty = ui.select(
                    ["l2", "l1", "elasticnet"],
                    label="正则化类型",
                    value=d.get("penalty", "l2"),
                )
                alpha = ui.number(
                    "SGD alpha", value=d.get("alpha", 0.0001), min=0.00001, max=1.0, step=0.0001,
                    format="%.5f",
                )

    return {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "seq_len": seq_len,
        "bidirectional": bidirectional,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "C": C,
        "penalty": penalty,
        "alpha": alpha,
    }


def trade_config_form(defaults: dict | None = None) -> dict:
    """交易配置表单。"""
    d = defaults or {}

    with ui.card().classes("w-full"):
        ui.label("交易参数配置").classes("text-h6")
        ui.separator()

        with ui.row().classes("gap-4 w-full"):
            buy_threshold = ui.number(
                "买入阈值", value=d.get("buy_threshold", 0.52), min=0.0, max=1.0,
                step=0.01, format="%.2f",
            ).tooltip("预测概率超过此值时触发买入信号")
            sell_threshold = ui.number(
                "卖出阈值", value=d.get("sell_threshold", 0.48), min=0.0, max=1.0,
                step=0.01, format="%.2f",
            ).tooltip("预测概率低于此值时触发卖出信号")
            confirm_days = ui.number(
                "确认天数", value=d.get("confirm_days", 1), min=1, max=10, step=1
            ).tooltip("连续 N 天触发信号后才执行")
            max_hold_days = ui.number(
                "最大持仓天数", value=d.get("max_hold_days", 20), min=1, max=200, step=1
            )

        with ui.row().classes("gap-4 w-full"):
            take_profit = ui.number(
                "止盈比例", value=d.get("take_profit", 0.0), min=0.0, max=1.0,
                step=0.01, format="%.2f",
            ).tooltip("0 表示不启用止盈")
            stop_loss = ui.number(
                "止损比例", value=d.get("stop_loss", 0.0), min=0.0, max=1.0,
                step=0.01, format="%.2f",
            ).tooltip("0 表示不启用止损")

        ui.label("交易成本").classes("text-subtitle2 q-mt-sm")
        with ui.row().classes("gap-4 w-full"):
            commission_rate_buy = ui.number(
                "买入佣金率", value=d.get("commission_rate_buy", 0.0003),
                min=0.0, max=0.01, step=0.0001, format="%.4f",
            )
            commission_rate_sell = ui.number(
                "卖出佣金率(含印花税)", value=d.get("commission_rate_sell", 0.0013),
                min=0.0, max=0.05, step=0.0001, format="%.4f",
            )
            slippage_bps = ui.number(
                "滑点(基点)", value=d.get("slippage_bps", 20.0),
                min=0.0, max=100.0, step=1.0, format="%.1f",
            )

    return {
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "confirm_days": confirm_days,
        "max_hold_days": max_hold_days,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "commission_rate_buy": commission_rate_buy,
        "commission_rate_sell": commission_rate_sell,
        "slippage_bps": slippage_bps,
    }


def search_config_form(defaults: dict | None = None) -> dict:
    """搜索配置表单。"""
    d = defaults or {}

    with ui.card().classes("w-full"):
        ui.label("搜索配置").classes("text-h6")
        ui.separator()

        with ui.row().classes("gap-4 w-full"):
            n_candidates = ui.number(
                "搜索候选数", value=d.get("n_candidates", 100), min=10, max=5000, step=10
            ).tooltip("总共搜索多少组超参数组合")
            n_rounds = ui.number(
                "搜索轮数", value=d.get("n_rounds", 4), min=1, max=20, step=1
            ).tooltip("分几轮搜索，每轮结束后可利用上轮最优结果")

        with ui.row().classes("gap-4 w-full"):
            metric = ui.select(
                ["pnl", "rank_ic"],
                label="搜索目标",
                value=d.get("metric", "pnl"),
            ).tooltip("pnl=收益率, rank_ic=排名信息系数")
            explore_ratio = ui.number(
                "探索比例", value=d.get("explore_ratio", 0.3), min=0.0, max=1.0,
                step=0.05, format="%.2f",
            ).tooltip("每轮中随机探索 vs 变异 Top-K 的比例")
            seed = ui.number(
                "随机种子", value=d.get("seed", 42), min=0, max=999999, step=1
            )

    return {
        "n_candidates": n_candidates,
        "n_rounds": n_rounds,
        "metric": metric,
        "explore_ratio": explore_ratio,
        "seed": seed,
    }


def split_config_form(defaults: dict | None = None) -> dict:
    """数据分割配置表单。"""
    d = defaults or {}

    with ui.card().classes("w-full"):
        ui.label("数据分割配置").classes("text-h6")
        ui.separator()

        with ui.row().classes("gap-4 w-full"):
            n_folds = ui.number(
                "Walk-Forward 折数", value=d.get("n_folds", 4), min=2, max=20, step=1
            )
            holdout_ratio = ui.number(
                "Holdout 比例", value=d.get("holdout_ratio", 0.15), min=0.0, max=0.5,
                step=0.05, format="%.2f",
            )
            train_start_ratio = ui.number(
                "初始训练集比例", value=d.get("train_start_ratio", 0.5), min=0.1, max=0.9,
                step=0.05, format="%.2f",
            )

        with ui.row().classes("gap-4 w-full"):
            min_rows = ui.number(
                "每折最小行数", value=d.get("min_rows", 80), min=10, max=500, step=10
            )
            embargo_days = ui.number(
                "隔离天数", value=d.get("embargo_days", 5), min=0, max=30, step=1
            ).tooltip("训练集与测试集之间的隔离期（天）")
            holdout_gap_days = ui.number(
                "Holdout 间隔天数", value=d.get("holdout_gap_days", 0), min=0, max=30, step=1
            )

    return {
        "n_folds": n_folds,
        "holdout_ratio": holdout_ratio,
        "train_start_ratio": train_start_ratio,
        "min_rows": min_rows,
        "embargo_days": embargo_days,
        "holdout_gap_days": holdout_gap_days,
    }


def portfolio_config_form(defaults: dict | None = None) -> dict:
    """组合配置表单（仅篮子模式）。"""
    d = defaults or {}

    with ui.card().classes("w-full"):
        ui.label("组合配置（篮子模式）").classes("text-h6")
        ui.separator()

        with ui.row().classes("gap-4 w-full"):
            top_k = ui.number(
                "Top-K 选股数", value=d.get("top_k", 5), min=1, max=50, step=1
            ).tooltip("每次调仓选取排名前 K 的股票")
            weighting = ui.select(
                ["equal", "score", "vol_inv"],
                label="加权方式",
                value=d.get("weighting", "equal"),
            ).tooltip("equal=等权, score=按分数加权, vol_inv=波动率倒数加权")
            rebalance_freq = ui.select(
                ["daily", "weekly", "monthly"],
                label="调仓频率",
                value=d.get("rebalance_freq", "daily"),
            )

        with ui.row().classes("gap-4 w-full"):
            max_weight = ui.number(
                "单股最大权重", value=d.get("max_weight", 0.20), min=0.01, max=1.0,
                step=0.01, format="%.2f",
            )
            cash_buffer = ui.number(
                "现金缓冲比例", value=d.get("cash_buffer", 0.05), min=0.0, max=0.5,
                step=0.01, format="%.2f",
            )

    return {
        "top_k": top_k,
        "weighting": weighting,
        "rebalance_freq": rebalance_freq,
        "max_weight": max_weight,
        "cash_buffer": cash_buffer,
    }


def collect_form_values(form_widgets: dict) -> dict:
    """从表单组件字典中提取实际值。

    Args:
        form_widgets: 由上述 xxx_config_form() 返回的字典

    Returns:
        字段名 → 实际值 的字典
    """
    result = {}
    for key, widget in form_widgets.items():
        if hasattr(widget, "value"):
            val = widget.value
            # 处理 windows 字段：字符串 → 整数列表
            if key == "windows" and isinstance(val, str):
                try:
                    val = [int(x.strip()) for x in val.strip("[]").split(",") if x.strip()]
                except ValueError:
                    val = [5, 10, 20]
            result[key] = val
    return result
