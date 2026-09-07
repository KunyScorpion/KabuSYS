import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class IndicatorCalculator:
    """
    全テクニカル指標のベクトル計算を行う計算エンジン。
    """
    
    # 指標メタデータ定義（メイン/サブ分類、パラメータ定義、表示色、基準線など）
    METADATA: Dict[str, Dict[str, Any]] = {
        # --- メインチャート指標（価格オーバーレイ） ---
        "SMA": {
            "name": "単純移動平均線 (SMA)",
            "type": "main",
            "params": [
                {"id": "period1", "name": "短期期間", "type": "int", "default": 5, "min": 1, "max": 200},
                {"id": "period2", "name": "中期期間", "type": "int", "default": 25, "min": 1, "max": 200},
                {"id": "period3", "name": "長期期間", "type": "int", "default": 75, "min": 1, "max": 300},
            ],
            "series": [
                {"key": "sma1", "name": "SMA (短期)", "color": "#2962FF", "width": 1.5},
                {"key": "sma2", "name": "SMA (中期)", "color": "#FF6D00", "width": 1.5},
                {"key": "sma3", "name": "SMA (長期)", "color": "#00C853", "width": 1.5},
            ]
        },
        "EMA": {
            "name": "指数平滑移動平均線 (EMA)",
            "type": "main",
            "params": [
                {"id": "period1", "name": "短期期間", "type": "int", "default": 9, "min": 1, "max": 200},
                {"id": "period2", "name": "中期期間", "type": "int", "default": 13, "min": 1, "max": 200},
                {"id": "period3", "name": "長期期間", "type": "int", "default": 26, "min": 1, "max": 300},
            ],
            "series": [
                {"key": "ema1", "name": "EMA (短期)", "color": "#00B0FF", "width": 1.5},
                {"key": "ema2", "name": "EMA (中期)", "color": "#AA00FF", "width": 1.5},
                {"key": "ema3", "name": "EMA (長期)", "color": "#FFD600", "width": 1.5},
            ]
        },
        "WMA": {
            "name": "加重移動平均線 (WMA)",
            "type": "main",
            "params": [
                {"id": "period1", "name": "短期期間", "type": "int", "default": 5, "min": 1, "max": 200},
                {"id": "period2", "name": "長期期間", "type": "int", "default": 20, "min": 1, "max": 200},
            ],
            "series": [
                {"key": "wma1", "name": "WMA (短期)", "color": "#00E5FF", "width": 1.5},
                {"key": "wma2", "name": "WMA (長期)", "color": "#FF4081", "width": 1.5},
            ]
        },
        "Bollinger_Bands": {
            "name": "ボリンジャーバンド (Bollinger Bands)",
            "type": "main",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 20, "min": 2, "max": 200},
                {"id": "nbdev", "name": "シグマ倍率", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.5},
            ],
            "series": [
                {"key": "bb_mid", "name": "BB ミドル (SMA)", "color": "#787B86", "width": 1},
                {"key": "bb_upper1", "name": "BB +1σ", "color": "#26A69A", "width": 1, "style": "dashed"},
                {"key": "bb_lower1", "name": "BB -1σ", "color": "#26A69A", "width": 1, "style": "dashed"},
                {"key": "bb_upper2", "name": "BB +2σ", "color": "#2196F3", "width": 1.5},
                {"key": "bb_lower2", "name": "BB -2σ", "color": "#2196F3", "width": 1.5},
                {"key": "bb_upper3", "name": "BB +3σ", "color": "#9C27B0", "width": 1},
                {"key": "bb_lower3", "name": "BB -3σ", "color": "#9C27B0", "width": 1},
            ]
        },
        "Envelope": {
            "name": "エンベロープ (Envelope)",
            "type": "main",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 20, "min": 2, "max": 200},
                {"id": "deviation1", "name": "乖離率 1 (%)", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                {"id": "deviation2", "name": "乖離率 2 (%)", "type": "float", "default": 2.0, "min": 0.1, "max": 15.0, "step": 0.1},
            ],
            "series": [
                {"key": "env_mid", "name": "Envelope 中心線", "color": "#787B86", "width": 1},
                {"key": "env_up1", "name": "Envelope 上限1", "color": "#FFA726", "width": 1},
                {"key": "env_low1", "name": "Envelope 下限1", "color": "#FFA726", "width": 1},
                {"key": "env_up2", "name": "Envelope 上限2", "color": "#FF7043", "width": 1.5},
                {"key": "env_low2", "name": "Envelope 下限2", "color": "#FF7043", "width": 1.5},
            ]
        },
        "Ichimoku": {
            "name": "一目均衡表 (Ichimoku Kinkouhyo)",
            "type": "main",
            "params": [
                {"id": "tenkan", "name": "転換線期間", "type": "int", "default": 9, "min": 1, "max": 100},
                {"id": "kijun", "name": "基準線期間", "type": "int", "default": 26, "min": 1, "max": 100},
                {"id": "senkou_b", "name": "先行スパン2期間", "type": "int", "default": 52, "min": 1, "max": 200},
            ],
            "series": [
                {"key": "tenkan_sen", "name": "転換線", "color": "#00897B", "width": 1.5},
                {"key": "kijun_sen", "name": "基準線", "color": "#D81B60", "width": 1.5},
                {"key": "senkou_span_a", "name": "先行スパン1", "color": "#43A047", "width": 1},
                {"key": "senkou_span_b", "name": "先行スパン2", "color": "#E53935", "width": 1},
                {"key": "chikou_span", "name": "遅行スパン", "color": "#8E24AA", "width": 1},
            ]
        },
        "HI_LOW_Bands": {
            "name": "ハイローバンド (HI-LOW Bands)",
            "type": "main",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 20, "min": 2, "max": 500},
            ],
            "series": [
                {"key": "high_band", "name": "High Band (高値ライン)", "color": "#ff7043", "width": 1.5},
                {"key": "low_band", "name": "Low Band (安値ライン)", "color": "#66bb6a", "width": 1.5},
                {"key": "center_line", "name": "Center Line (中心線)", "color": "#9E9E9E", "width": 1, "style": "dashed"},
            ]
        },
        "Keltner_Channel": {
            "name": "ケルトナーチャネル (Keltner Channel)",
            "type": "main",
            "params": [
                {"id": "ema_period", "name": "EMA期間", "type": "int", "default": 20, "min": 2, "max": 200},
                {"id": "atr_period", "name": "ATR期間", "type": "int", "default": 10, "min": 2, "max": 100},
                {"id": "multiplier", "name": "ATR倍率", "type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.5},
            ],
            "series": [
                {"key": "kc_mid", "name": "KC Center (EMA)", "color": "#3F51B5", "width": 1.5},
                {"key": "kc_upper", "name": "KC Upper", "color": "#009688", "width": 1.5},
                {"key": "kc_lower", "name": "KC Lower", "color": "#009688", "width": 1.5},
            ]
        },
        "Meander": {
            "name": "ミアンダーインジケーター (Meander)",
            "type": "main",
            "params": [
                {"id": "period", "name": "SMA期間", "type": "int", "default": 13, "min": 2, "max": 200},
                {"id": "multiplier", "name": "標準偏差倍率", "type": "float", "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1},
            ],
            "series": [
                {"key": "meander_mid", "name": "Meander SMA", "color": "#795548", "width": 1.5},
                {"key": "meander_upper", "name": "Meander Upper", "color": "#4CAF50", "width": 1.5},
                {"key": "meander_lower", "name": "Meander Lower", "color": "#F44336", "width": 1.5},
            ]
        },
        "PIVOT": {
            "name": "ピボットポイント (PIVOT)",
            "type": "main",
            "params": [],
            "series": [
                {"key": "pivot_p", "name": "Pivot (P)", "color": "#FFD600", "width": 1.5},
                {"key": "pivot_r1", "name": "R1", "color": "#00E676", "width": 1},
                {"key": "pivot_r2", "name": "R2", "color": "#00B0FF", "width": 1},
                {"key": "pivot_s1", "name": "S1", "color": "#FF9100", "width": 1},
                {"key": "pivot_s2", "name": "S2", "color": "#FF5252", "width": 1},
                {"key": "pivot_hbop", "name": "HBOP", "color": "#7C4DFF", "width": 1, "style": "dashed"},
                {"key": "pivot_lbop", "name": "LBOP", "color": "#FF4081", "width": 1, "style": "dashed"},
            ]
        },
        "Dynamic_SR": {
            "name": "動的サポート・レジスタンス (Dynamic S/R)",
            "type": "main",
            "is_custom_overlay": True,
            "params": [
                {"id": "strong_window", "name": "Strong左右足数", "type": "int", "default": 15, "min": 5, "max": 50},
                {"id": "medium_window", "name": "Medium左右足数", "type": "int", "default": 8, "min": 3, "max": 30},
                {"id": "weak_window", "name": "Weak左右足数", "type": "int", "default": 3, "min": 2, "max": 15},
                {"id": "merge_threshold", "name": "近接マージ値幅 (円)", "type": "float", "default": 20.0, "min": 5.0, "max": 200.0, "step": 5.0},
                {"id": "fade_medium_start", "name": "Medium減衰開始 (本)", "type": "int", "default": 40, "min": 10, "max": 200},
                {"id": "fade_medium_end", "name": "Medium消滅 (本)", "type": "int", "default": 60, "min": 20, "max": 300},
                {"id": "fade_weak_start", "name": "Weak減衰開始 (本)", "type": "int", "default": 15, "min": 5, "max": 100},
                {"id": "fade_weak_end", "name": "Weak消滅 (本)", "type": "int", "default": 25, "min": 10, "max": 150},
            ],
            "series": [
                {"key": "resistance_strong", "name": "抵抗線 (Strong: 実線2px/不透明1.0)", "color": "#ff4444", "width": 2.0},
                {"key": "resistance_medium", "name": "抵抗線 (Medium: 実線1.5px/フェード)", "color": "#ff6b6b", "width": 1.5},
                {"key": "resistance_weak", "name": "抵抗線 (Weak: 破線1px/フェード)", "color": "#ff8a80", "width": 1.0, "style": "dashed"},
                {"key": "support_strong", "name": "支持線 (Strong: 実線2px/不透明1.0)", "color": "#00e676", "width": 2.0},
                {"key": "support_medium", "name": "支持線 (Medium: 実線1.5px/フェード)", "color": "#4caf50", "width": 1.5},
                {"key": "support_weak", "name": "支持線 (Weak: 破線1px/フェード)", "color": "#81c784", "width": 1.0, "style": "dashed"},
            ]
        },

        # --- サブチャート指標（オシレーター等） ---
        "MACD": {
            "name": "MACD (移動平均収束拡散法)",
            "type": "sub",
            "params": [
                {"id": "fast_period", "name": "短期EMA期間", "type": "int", "default": 9, "min": 1, "max": 100},
                {"id": "slow_period", "name": "長期EMA期間", "type": "int", "default": 13, "min": 1, "max": 200},
                {"id": "signal_period", "name": "シグナル期間", "type": "int", "default": 3, "min": 1, "max": 50},
            ],
            "series": [
                {"key": "macd", "name": "MACD", "color": "#2962FF", "width": 1.5},
                {"key": "macd_signal", "name": "Signal", "color": "#FF6D00", "width": 1.5},
                {"key": "macd_hist", "name": "Histogram", "type": "histogram", "color": "#26A69A"},
            ],
            "baselines": [0.0]
        },
        "RSI": {
            "name": "RSI (相対力指数)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 9, "min": 2, "max": 100},
                {"id": "upper", "name": "買われすぎ水準", "type": "int", "default": 70, "min": 50, "max": 95},
                {"id": "lower", "name": "売られすぎ水準", "type": "int", "default": 30, "min": 5, "max": 50},
            ],
            "series": [
                {"key": "rsi", "name": "RSI", "color": "#7E57C2", "width": 1.5},
            ],
            "baselines": [30.0, 70.0],
            "range": [0, 100]
        },
        "Stochastics": {
            "name": "ストキャスティクス (Stochastics)",
            "type": "sub",
            "params": [
                {"id": "k_period", "name": "%K 期間", "type": "int", "default": 5, "min": 1, "max": 100},
                {"id": "d_period", "name": "%D 期間", "type": "int", "default": 3, "min": 1, "max": 50},
                {"id": "sd_period", "name": "Slow %D (SD) 期間", "type": "int", "default": 3, "min": 1, "max": 50},
            ],
            "series": [
                {"key": "stoch_k", "name": "%K", "color": "#29B6F6", "width": 1},
                {"key": "stoch_d", "name": "%D", "color": "#FF7043", "width": 1.5},
                {"key": "stoch_sd", "name": "SD", "color": "#AB47BC", "width": 1.5},
            ],
            "baselines": [20.0, 80.0],
            "range": [0, 100]
        },
        "Stochastics_RSI": {
            "name": "ストキャスティクスRSI (Stochastics RSI)",
            "type": "sub",
            "params": [
                {"id": "rsi_period", "name": "RSI 期間", "type": "int", "default": 9, "min": 2, "max": 100},
                {"id": "k_period", "name": "%K 期間", "type": "int", "default": 3, "min": 1, "max": 50},
                {"id": "d_period", "name": "%D 期間", "type": "int", "default": 3, "min": 1, "max": 50},
                {"id": "sd_period", "name": "SD 期間", "type": "int", "default": 3, "min": 1, "max": 50},
            ],
            "series": [
                {"key": "stoch_rsi_k", "name": "StochRSI %K", "color": "#00ACC1", "width": 1},
                {"key": "stoch_rsi_d", "name": "StochRSI %D", "color": "#FFA726", "width": 1.5},
                {"key": "stoch_rsi_sd", "name": "StochRSI SD", "color": "#EC407A", "width": 1.5},
            ],
            "baselines": [20.0, 80.0],
            "range": [0, 100]
        },
        "ATR": {
            "name": "ATR (アベレージ・トゥルー・レンジ)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 10, "min": 1, "max": 100},
            ],
            "series": [
                {"key": "atr", "name": "ATR", "color": "#AB47BC", "width": 1.5},
            ]
        },
        "DMI": {
            "name": "DMI / ADX (方向性指数)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 14, "min": 2, "max": 100},
            ],
            "series": [
                {"key": "pdi", "name": "+DI", "color": "#00E676", "width": 1.5},
                {"key": "mdi", "name": "-DI", "color": "#FF5252", "width": 1.5},
                {"key": "adx", "name": "ADX", "color": "#2979FF", "width": 2},
            ],
            "baselines": [20.0, 40.0],
            "range": [0, 100]
        },
        "RCI": {
            "name": "RCI (順位相関係数)",
            "type": "sub",
            "params": [
                {"id": "period1", "name": "短期期間", "type": "int", "default": 9, "min": 2, "max": 100},
                {"id": "period2", "name": "長期期間", "type": "int", "default": 26, "min": 2, "max": 200},
            ],
            "series": [
                {"key": "rci1", "name": "RCI (短期)", "color": "#FF1744", "width": 1.5},
                {"key": "rci2", "name": "RCI (長期)", "color": "#2979FF", "width": 1.5},
            ],
            "baselines": [-80.0, 0.0, 80.0],
            "range": [-100, 100]
        },
        "ROC": {
            "name": "ROC (Rate of Change / 変化率)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 7, "min": 1, "max": 100},
            ],
            "series": [
                {"key": "roc", "name": "ROC", "color": "#00E5FF", "width": 1.5},
            ],
            "baselines": [100.0]
        },
        "CMO": {
            "name": "CMO (Chande Momentum Oscillator)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 14, "min": 2, "max": 100},
            ],
            "series": [
                {"key": "cmo", "name": "CMO", "color": "#FF9100", "width": 1.5},
            ],
            "baselines": [-50.0, 0.0, 50.0],
            "range": [-100, 100]
        },
        "Psychological_Line": {
            "name": "サイコロジカルライン (Psychological Line)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 12, "min": 2, "max": 100},
            ],
            "series": [
                {"key": "psy", "name": "Psychological Line (%)", "color": "#76FF03", "width": 1.5},
            ],
            "baselines": [25.0, 50.0, 75.0],
            "range": [0, 100]
        },
        "Balance_of_Power": {
            "name": "Balance of Power (BOP)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 14, "min": 1, "max": 100},
            ],
            "series": [
                {"key": "bop", "name": "BOP", "color": "#FF4081", "width": 1.5},
            ],
            "baselines": [0.0],
            "range": [-1, 1]
        },
        "Sinohara_Ratio": {
            "name": "篠原レシオ (Sinohara Ratio)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 26, "min": 2, "max": 100},
            ],
            "series": [
                {"key": "a_ratio", "name": "Aレシオ (エネルギー)", "color": "#FF1744", "width": 1.5},
                {"key": "b_ratio", "name": "Bレシオ (人気)", "color": "#00E5FF", "width": 1.5},
            ],
            "baselines": [100.0]
        },
        "Volatility_Ratio": {
            "name": "ボラティリティレシオ (Volatility Ratio)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "EMA期間", "type": "int", "default": 9, "min": 2, "max": 100},
            ],
            "series": [
                {"key": "vol_ratio", "name": "Volatility Ratio", "color": "#651FFF", "width": 1.5},
            ],
            "baselines": [1.0, 1.95]
        },
        "R_Oscillator": {
            "name": "レンジオシレーター (R-Oscillator)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "期間", "type": "int", "default": 5, "min": 1, "max": 100},
            ],
            "series": [
                {"key": "r_osc", "name": "R-Oscillator", "color": "#00B0FF", "width": 1.5},
            ],
            "baselines": [20.0, 80.0],
            "range": [0, 100]
        },
        "Estrangement_SMA": {
            "name": "SMA乖離率 (Estrangement SMA)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "SMA期間", "type": "int", "default": 25, "min": 1, "max": 200},
            ],
            "series": [
                {"key": "est_sma", "name": "SMA乖離率 (%)", "color": "#FF9100", "width": 1.5},
            ],
            "baselines": [0.0]
        },
        "Estrangement_EMA": {
            "name": "EMA乖離率 (Estrangement EMA)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "EMA期間", "type": "int", "default": 25, "min": 1, "max": 200},
            ],
            "series": [
                {"key": "est_ema", "name": "EMA乖離率 (%)", "color": "#00E676", "width": 1.5},
            ],
            "baselines": [0.0]
        },
        "Estrangement_WMA": {
            "name": "WMA乖離率 (Estrangement WMA)",
            "type": "sub",
            "params": [
                {"id": "period", "name": "WMA期間", "type": "int", "default": 25, "min": 1, "max": 200},
            ],
            "series": [
                {"key": "est_wma", "name": "WMA乖離率 (%)", "color": "#E040FB", "width": 1.5},
            ],
            "baselines": [0.0]
        },
        "Binary_Wave": {
            "name": "バイナリウェーブ (Binary Wave)",
            "type": "sub",
            "params": [
                {"id": "macd_fast", "name": "MACD短期", "type": "int", "default": 9, "min": 1, "max": 50},
                {"id": "macd_slow", "name": "MACD長期", "type": "int", "default": 13, "min": 1, "max": 100},
                {"id": "macd_signal", "name": "MACDシグナル", "type": "int", "default": 3, "min": 1, "max": 30},
                {"id": "rsi_period", "name": "RSI期間", "type": "int", "default": 9, "min": 2, "max": 50},
                {"id": "sma_period", "name": "SMA期間", "type": "int", "default": 5, "min": 1, "max": 50},
            ],
            "series": [
                {"key": "bwave", "name": "Binary Wave Score", "type": "histogram", "color": "#2979FF"},
            ],
            "baselines": [0.0]
        },
        "Ultimate_Oscillator": {
            "name": "アルティメットオシレーター (Ultimate Oscillator)",
            "type": "sub",
            "params": [
                {"id": "period1", "name": "短期期間", "type": "int", "default": 7, "min": 1, "max": 50},
                {"id": "period2", "name": "中期期間", "type": "int", "default": 14, "min": 1, "max": 100},
                {"id": "period3", "name": "長期期間", "type": "int", "default": 28, "min": 1, "max": 200},
            ],
            "series": [
                {"key": "ultimate_osc", "name": "Ultimate Oscillator", "color": "#00E5FF", "width": 1.5},
            ],
            "baselines": [30.0, 70.0],
            "range": [0, 100]
        }
    }

    @staticmethod
    def calculate_wma(series: pd.Series, period: int) -> pd.Series:
        """加重移動平均 (WMA) を計算"""
        weights = np.arange(1, period + 1)
        w_sum = weights.sum()
        return series.rolling(period).apply(lambda s: np.dot(s, weights) / w_sum, raw=True)

    @staticmethod
    def calculate_true_range(df: pd.DataFrame) -> pd.Series:
        """True Range (TR) を計算"""
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @classmethod
    def compute(cls, indicator_id: str, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        指定された指標IDとパラメータに基づいて計算を行い、結果のDataFrameを返す。
        """
        res = pd.DataFrame(index=df.index)
        
        # --- メイン指標 ---
        if indicator_id == "SMA":
            p1 = int(params.get("period1", 5))
            p2 = int(params.get("period2", 25))
            p3 = int(params.get("period3", 75))
            res["sma1"] = df["close"].rolling(p1).mean()
            res["sma2"] = df["close"].rolling(p2).mean()
            res["sma3"] = df["close"].rolling(p3).mean()
            
        elif indicator_id == "EMA":
            p1 = int(params.get("period1", 9))
            p2 = int(params.get("period2", 13))
            p3 = int(params.get("period3", 26))
            res["ema1"] = df["close"].ewm(span=p1, adjust=False).mean()
            res["ema2"] = df["close"].ewm(span=p2, adjust=False).mean()
            res["ema3"] = df["close"].ewm(span=p3, adjust=False).mean()

        elif indicator_id == "WMA":
            p1 = int(params.get("period1", 5))
            p2 = int(params.get("period2", 20))
            res["wma1"] = cls.calculate_wma(df["close"], p1)
            res["wma2"] = cls.calculate_wma(df["close"], p2)

        elif indicator_id == "Bollinger_Bands":
            p = int(params.get("period", 20))
            nbdev = float(params.get("nbdev", 2.0))
            mid = df["close"].rolling(p).mean()
            std = df["close"].rolling(p).std(ddof=0)
            res["bb_mid"] = mid
            res["bb_upper1"] = mid + 1.0 * std
            res["bb_lower1"] = mid - 1.0 * std
            res["bb_upper2"] = mid + nbdev * std
            res["bb_lower2"] = mid - nbdev * std
            res["bb_upper3"] = mid + 3.0 * std
            res["bb_lower3"] = mid - 3.0 * std

        elif indicator_id == "Envelope":
            p = int(params.get("period", 20))
            dev1 = float(params.get("deviation1", 1.0)) / 100.0
            dev2 = float(params.get("deviation2", 2.0)) / 100.0
            mid = df["close"].rolling(p).mean()
            res["env_mid"] = mid
            res["env_up1"] = mid * (1.0 + dev1)
            res["env_low1"] = mid * (1.0 - dev1)
            res["env_up2"] = mid * (1.0 + dev2)
            res["env_low2"] = mid * (1.0 - dev2)

        elif indicator_id == "Ichimoku":
            tenkan_p = int(params.get("tenkan", 9))
            kijun_p = int(params.get("kijun", 26))
            senkou_b_p = int(params.get("senkou_b", 52))
            
            tenkan = (df["high"].rolling(tenkan_p).max() + df["low"].rolling(tenkan_p).min()) / 2.0
            kijun = (df["high"].rolling(kijun_p).max() + df["low"].rolling(kijun_p).min()) / 2.0
            senkou_a = (tenkan + kijun) / 2.0
            senkou_b = (df["high"].rolling(senkou_b_p).max() + df["low"].rolling(senkou_b_p).min()) / 2.0
            chikou = df["close"] # 描画時にシフトまたはそのまま
            
            res["tenkan_sen"] = tenkan
            res["kijun_sen"] = kijun
            res["senkou_span_a"] = senkou_a
            res["senkou_span_b"] = senkou_b
            res["chikou_span"] = chikou

        elif indicator_id == "HI_LOW_Bands":
            p = int(params.get("period", 13))
            hb = df["high"].rolling(p).max()
            lb = df["low"].rolling(p).min()
            res["high_band"] = hb
            res["low_band"] = lb
            res["center_line"] = (hb + lb) / 2.0

        elif indicator_id == "Keltner_Channel":
            ema_p = int(params.get("ema_period", 20))
            atr_p = int(params.get("atr_period", 10))
            mult = float(params.get("multiplier", 1.5))
            
            center = df["close"].ewm(span=ema_p, adjust=False).mean()
            tr = cls.calculate_true_range(df)
            atr = tr.rolling(atr_p).mean()
            
            res["kc_mid"] = center
            res["kc_upper"] = center + mult * atr
            res["kc_lower"] = center - mult * atr

        elif indicator_id == "Meander":
            p = int(params.get("period", 13))
            mult = float(params.get("multiplier", 1.0))
            mid = df["close"].rolling(p).mean()
            std = df["close"].rolling(p).std(ddof=0)
            res["meander_mid"] = mid
            res["meander_upper"] = mid + mult * std
            res["meander_lower"] = mid - mult * std

        elif indicator_id == "PIVOT":
            # 日足単位または直近高安値ベースのPivot計算
            # タイムフレーム内での高安終値から計算
            prev_high = df["high"].shift(1)
            prev_low = df["low"].shift(1)
            prev_close = df["close"].shift(1)
            
            p = (prev_high + prev_low + prev_close) / 3.0
            r1 = 2 * p - prev_low
            s1 = 2 * p - prev_high
            r2 = p + (prev_high - prev_low)
            s2 = p - (prev_high - prev_low)
            hbop = 2 * p - 2 * prev_low + prev_high
            lbop = 2 * p - 2 * prev_high + prev_low
            
            res["pivot_p"] = p
            res["pivot_r1"] = r1
            res["pivot_r2"] = r2
            res["pivot_s1"] = s1
            res["pivot_s2"] = s2
            res["pivot_hbop"] = hbop
            res["pivot_lbop"] = lbop

        elif indicator_id == "Dynamic_SR":
            from src.backend.support_resistance import DynamicLevelEngine, DynamicLevelConfig
            cfg = DynamicLevelConfig(
                strong_window=int(params.get("strong_window", 15)),
                medium_window=int(params.get("medium_window", 8)),
                weak_window=int(params.get("weak_window", 3)),
                merge_threshold_points=float(params.get("merge_threshold", 20.0)),
                fade_medium_start=int(params.get("fade_medium_start", 40)),
                fade_medium_end=int(params.get("fade_medium_end", 60)),
                fade_weak_start=int(params.get("fade_weak_start", 15)),
                fade_weak_end=int(params.get("fade_weak_end", 25)),
            )
            engine = DynamicLevelEngine(cfg)
            levels = engine.calculate_levels(df)
            res.attrs["dynamic_levels"] = levels
            res.attrs["dynamic_series"] = engine.to_highstock_series(levels)
            res["active_sr_levels"] = len(levels)

        # --- サブ指標 ---
        elif indicator_id == "MACD":
            fast = int(params.get("fast_period", 9))
            slow = int(params.get("slow_period", 13))
            sig = int(params.get("signal_period", 3))
            
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            hist = macd - signal
            
            res["macd"] = macd
            res["macd_signal"] = signal
            res["macd_hist"] = hist

        elif indicator_id == "RSI":
            p = int(params.get("period", 9))
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            res["rsi"] = rsi.fillna(50)

        elif indicator_id == "Stochastics":
            kp = int(params.get("k_period", 5))
            dp = int(params.get("d_period", 3))
            sdp = int(params.get("sd_period", 3))
            
            low_min = df["low"].rolling(kp).min()
            high_max = df["high"].rolling(kp).max()
            denom = high_max - low_min
            
            k = (df["close"] - low_min) / denom.replace(0, np.nan) * 100.0
            d = k.rolling(dp).mean()
            sd = d.rolling(sdp).mean()
            
            res["stoch_k"] = k.fillna(50)
            res["stoch_d"] = d.fillna(50)
            res["stoch_sd"] = sd.fillna(50)

        elif indicator_id == "Stochastics_RSI":
            rsi_p = int(params.get("rsi_period", 9))
            kp = int(params.get("k_period", 3))
            dp = int(params.get("d_period", 3))
            sdp = int(params.get("sd_period", 3))
            
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/rsi_p, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/rsi_p, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            rsi_min = rsi.rolling(kp).min()
            rsi_max = rsi.rolling(kp).max()
            denom = rsi_max - rsi_min
            
            stoch_rsi_k = (rsi - rsi_min) / denom.replace(0, np.nan) * 100.0
            stoch_rsi_d = stoch_rsi_k.rolling(dp).mean()
            stoch_rsi_sd = stoch_rsi_d.rolling(sdp).mean()
            
            res["stoch_rsi_k"] = stoch_rsi_k.fillna(50)
            res["stoch_rsi_d"] = stoch_rsi_d.fillna(50)
            res["stoch_rsi_sd"] = stoch_rsi_sd.fillna(50)

        elif indicator_id == "ATR":
            p = int(params.get("period", 10))
            tr = cls.calculate_true_range(df)
            res["atr"] = tr.rolling(p).mean()

        elif indicator_id == "DMI":
            p = int(params.get("period", 14))
            tr = cls.calculate_true_range(df)
            atr = tr.rolling(p).mean()
            
            up_move = df["high"] - df["high"].shift(1)
            down_move = df["low"].shift(1) - df["low"]
            
            pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            pdm_s = pd.Series(pdm, index=df.index).rolling(p).mean()
            mdm_s = pd.Series(mdm, index=df.index).rolling(p).mean()
            
            pdi = (pdm_s / atr.replace(0, np.nan)) * 100.0
            mdi = (mdm_s / atr.replace(0, np.nan)) * 100.0
            
            dx_denom = pdi + mdi
            dx = ((pdi - mdi).abs() / dx_denom.replace(0, np.nan)) * 100.0
            adx = dx.rolling(p).mean()
            
            res["pdi"] = pdi.fillna(0)
            res["mdi"] = mdi.fillna(0)
            res["adx"] = adx.fillna(0)

        elif indicator_id == "RCI":
            p1 = int(params.get("period1", 9))
            p2 = int(params.get("period2", 26))
            
            def calc_rci_series(series: pd.Series, period: int) -> pd.Series:
                ranks_time = np.arange(1, period + 1)
                rci_vals = []
                vals = series.values
                n = len(vals)
                for i in range(n):
                    if i < period - 1:
                        rci_vals.append(np.nan)
                    else:
                        window = vals[i - period + 1 : i + 1]
                        # 降順または昇順順位（Excel定義: 新しい日=1または過去日=period、価格順位）
                        # RCIの標準計算: 時間順位 (1..N), 価格順位 (降順 1..N)
                        order = np.argsort(-window)
                        ranks_price = np.empty_like(order, dtype=float)
                        ranks_price[order] = np.arange(1, period + 1)
                        # 同順位の平均化（簡易）
                        d = ranks_time - ranks_price
                        d2_sum = np.sum(d ** 2)
                        rci = (1.0 - (6.0 * d2_sum) / (period * (period**2 - 1))) * 100.0
                        rci_vals.append(rci)
                return pd.Series(rci_vals, index=series.index)

            res["rci1"] = calc_rci_series(df["close"], p1)
            res["rci2"] = calc_rci_series(df["close"], p2)

        elif indicator_id == "ROC":
            p = int(params.get("period", 7))
            res["roc"] = (df["close"] / df["close"].shift(p).replace(0, np.nan)) * 100.0

        elif indicator_id == "CMO":
            p = int(params.get("period", 14))
            delta = df["close"].diff()
            up = delta.clip(lower=0).rolling(p).sum()
            down = (-delta.clip(upper=0)).rolling(p).sum()
            res["cmo"] = ((up - down) / (up + down).replace(0, np.nan)) * 100.0

        elif indicator_id == "Psychological_Line":
            p = int(params.get("period", 12))
            is_up = (df["close"] > df["close"].shift(1)).astype(float)
            res["psy"] = (is_up.rolling(p).sum() / p) * 100.0

        elif indicator_id == "Balance_of_Power":
            p = int(params.get("period", 14))
            denom = df["high"] - df["low"]
            raw_bop = (df["close"] - df["open"]) / denom.replace(0, np.nan)
            res["bop"] = raw_bop.rolling(p).mean().fillna(0)

        elif indicator_id == "Sinohara_Ratio":
            p = int(params.get("period", 26))
            prev_close = df["close"].shift(1)
            
            a_num = (df["high"] - df["open"]).rolling(p).sum()
            a_den = (df["open"] - df["low"]).rolling(p).sum()
            res["a_ratio"] = (a_num / a_den.replace(0, np.nan)) * 100.0
            
            b_num = (df["high"] - prev_close).rolling(p).sum()
            b_den = (prev_close - df["low"]).rolling(p).sum()
            res["b_ratio"] = (b_num / b_den.replace(0, np.nan)) * 100.0

        elif indicator_id == "Volatility_Ratio":
            p = int(params.get("period", 9))
            tr = cls.calculate_true_range(df)
            ema_tr = tr.ewm(span=p, adjust=False).mean()
            res["vol_ratio"] = tr / ema_tr.replace(0, np.nan)

        elif indicator_id == "R_Oscillator":
            p = int(params.get("period", 5))
            low_min = df["low"].rolling(p).min()
            high_max = df["high"].rolling(p).max()
            res["r_osc"] = ((df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100.0

        elif indicator_id == "Estrangement_SMA":
            p = int(params.get("period", 25))
            sma = df["close"].rolling(p).mean()
            res["est_sma"] = ((df["close"] - sma) / sma.replace(0, np.nan)) * 100.0

        elif indicator_id == "Estrangement_EMA":
            p = int(params.get("period", 25))
            ema = df["close"].ewm(span=p, adjust=False).mean()
            res["est_ema"] = ((df["close"] - ema) / ema.replace(0, np.nan)) * 100.0

        elif indicator_id == "Estrangement_WMA":
            p = int(params.get("period", 25))
            wma = cls.calculate_wma(df["close"], p)
            res["est_wma"] = ((df["close"] - wma) / wma.replace(0, np.nan)) * 100.0

        elif indicator_id == "Binary_Wave":
            # 複数指標のシグナル合算波
            mf = int(params.get("macd_fast", 9))
            ms = int(params.get("macd_slow", 13))
            msig = int(params.get("macd_signal", 3))
            rp = int(params.get("rsi_period", 9))
            sp = int(params.get("sma_period", 5))
            
            # MACDシグナル
            macd = df["close"].ewm(span=mf, adjust=False).mean() - df["close"].ewm(span=ms, adjust=False).mean()
            sig = macd.ewm(span=msig, adjust=False).mean()
            s_macd = np.where(macd > sig, 1, np.where(macd < sig, -1, 0))
            
            # RSIシグナル
            delta = df["close"].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/rp, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/rp, adjust=False).mean()
            rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
            s_rsi = np.where(rsi > 50, 1, np.where(rsi < 50, -1, 0))
            
            # SMAクロスシグナル
            sma = df["close"].rolling(sp).mean()
            s_sma = np.where(df["close"] > sma, 1, np.where(df["close"] < sma, -1, 0))
            
            res["bwave"] = pd.Series(s_macd + s_rsi + s_sma, index=df.index)

        elif indicator_id == "Ultimate_Oscillator":
            p1 = int(params.get("period1", 7))
            p2 = int(params.get("period2", 14))
            p3 = int(params.get("period3", 28))
            
            prev_close = df["close"].shift(1)
            true_low = np.minimum(df["low"], prev_close)
            true_high = np.maximum(df["high"], prev_close)
            
            bp = df["close"] - true_low
            tr = true_high - true_low
            
            avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum().replace(0, np.nan)
            avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum().replace(0, np.nan)
            avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum().replace(0, np.nan)
            
            res["ultimate_osc"] = 100.0 * (4 * avg1 + 2 * avg2 + avg3) / 7.0
            
        return res
