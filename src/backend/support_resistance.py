"""
動的サポート・レジスタンス描画エンジン (Dynamic Support and Resistance Engine)

相場の重要な山・谷（スイングハイ/スイングロー）を検出し、
その重要度（Strong / Medium / Weak）に応じて寿命や透明度・線種を動的に変化させて
チャートへ描画するための指標計算コアロジック。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class LevelType(str, Enum):
    """ライン種別"""
    RESISTANCE = "RESISTANCE"  # 抵抗線（山 / スイングハイ）
    SUPPORT = "SUPPORT"        # 支持線（谷 / スイングロー）


class LevelStrength(int, Enum):
    """強度分類"""
    WEAK = 1    # 局所的な高安
    MEDIUM = 2  # 中規模な節目
    STRONG = 3  # 主要な山・谷（永続）


class LevelStatus(str, Enum):
    """ラインの状態"""
    ACTIVE = "ACTIVE"    # 有効（現在も生きているライン）
    BROKEN = "BROKEN"    # 終値でブレイクされ無効化
    EXPIRED = "EXPIRED"  # 寿命切れでフェードアウト消滅


@dataclass
class PriceLevel:
    """サポート・レジスタンスラインのデータモデル"""
    id: str
    type: LevelType
    price: float
    created_at_idx: int          # 山・谷が形成されたローソク足のインデックス
    confirmed_at_idx: int        # 右側N本が確定してラインが認識された足のインデックス
    created_time: pd.Timestamp   # 山・谷の形成日時
    strength: int                # 1: Weak, 2: Medium, 3: Strong
    opacity: float               # 1.0 〜 0.0
    status: LevelStatus          # ACTIVE, BROKEN, EXPIRED
    bars_alive: int = 0          # 確定してからの経過足数
    end_at_idx: Optional[int] = None   # 終了足（ブレイク足または最新足）
    end_time: Optional[pd.Timestamp] = None
    touch_count: int = 1         # 反発・マージ回数


@dataclass
class DynamicLevelConfig:
    """動的サポート・レジスタンスの設定パラメータ"""
    # 検出ウィンドウ（左右比較本数）
    strong_window: int = 15      # Strong: 左右15本
    medium_window: int = 8       # Medium: 左右8本
    weak_window: int = 3         # Weak: 左右3本
    
    # 寿命・フェードアウトルール（足数）
    fade_medium_start: int = 40  # Medium: 減衰開始足数
    fade_medium_end: int = 60    # Medium: 完全消滅足数
    fade_weak_start: int = 15    # Weak: 減衰開始足数
    fade_weak_end: int = 25      # Weak: 完全消滅足数
    
    # 近接マージ閾値
    merge_threshold_points: float = 20.0  # 近接判定の値幅（円、先物呼値ベース）
    merge_threshold_pct: float = 0.0015   # 近接判定の価格比率（0.15%）
    use_pct_merge: bool = False           # True: 比率使用, False: 固定値幅使用
    
    # 描画色設定（RGBAベース HEX）
    color_resistance: str = "#ff4444"  # 抵抗線ベース色（赤系）
    color_support: str = "#00e676"     # 支持線ベース色（緑系）


class DynamicLevelEngine:
    """
    動的サポート・レジスタンスの計算・ライフサイクル管理エンジン
    """

    def __init__(self, config: Optional[DynamicLevelConfig] = None):
        self.config = config or DynamicLevelConfig()

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        """HEXカラーコードをRGBタプルに変換"""
        hex_clean = hex_str.lstrip("#")
        if len(hex_clean) == 6:
            return int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
        return (255, 255, 255)

    def calculate_levels(
        self,
        df: pd.DataFrame,
        include_broken: bool = False
    ) -> List[PriceLevel]:
        """
        OHLCV DataFrameからスイングハイ/ローを検出し、
        毎足のブレイク・フェードアウト・近接マージをシミュレーションして
        ライン一覧を返す。

        Parameters:
        -----------
        df : pd.DataFrame
            'open', 'high', 'low', 'close', 'time' 列を含むローソク足データ
        include_broken : bool
            Trueの場合、過去にブレイクされたラインも履歴として含める（描画検証用）
        
        Returns:
        --------
        List[PriceLevel]
            計算されたライン一覧（デフォルトは最新足時点でACTIVEなラインのみ）
        """
        if df.empty or len(df) < self.config.weak_window * 2 + 1:
            return []

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        times = pd.to_datetime(df["time"]).tolist()
        n_bars = len(df)

        cfg = self.config
        active_levels: List[PriceLevel] = []
        all_levels_history: List[PriceLevel] = []

        # 山・谷の同足重複トラッキング用マップ (peak_idx, LevelType) -> PriceLevel
        existing_peak_map: Dict[Tuple[int, LevelType], PriceLevel] = {}

        # 逐次足確定ループ（バーを進めながらリアルタイムシミュレーション）
        for c in range(n_bars):
            cur_close = closes[c]
            cur_time = times[c]

            # ----------------------------------------------------
            # 1. ブレイク判定（確定足の終値がラインを突き抜けたか）
            # ----------------------------------------------------
            remaining_active: List[PriceLevel] = []
            for lvl in active_levels:
                is_broken = False
                if lvl.type == LevelType.RESISTANCE:
                    # 終値が抵抗線を上回った
                    if cur_close > lvl.price:
                        is_broken = True
                elif lvl.type == LevelType.SUPPORT:
                    # 終値が支持線を下回った
                    if cur_close < lvl.price:
                        is_broken = True

                if is_broken:
                    lvl.status = LevelStatus.BROKEN
                    lvl.end_at_idx = c
                    lvl.end_time = cur_time
                else:
                    remaining_active.append(lvl)

            active_levels = remaining_active

            # ----------------------------------------------------
            # 2. 経時フェードアウト計算（経過足数の加算と透明度更新）
            # ----------------------------------------------------
            unexpired_active: List[PriceLevel] = []
            for lvl in active_levels:
                lvl.bars_alive += 1

                if lvl.strength == LevelStrength.STRONG:
                    # Strong: 永続（ブレイクされるまで消えない）
                    lvl.opacity = 1.0
                    unexpired_active.append(lvl)

                elif lvl.strength == LevelStrength.MEDIUM:
                    # Medium: 40本経過後から徐々に薄くなり、60本で完全消滅
                    if lvl.bars_alive <= cfg.fade_medium_start:
                        lvl.opacity = 0.8
                        unexpired_active.append(lvl)
                    elif lvl.bars_alive < cfg.fade_medium_end:
                        fade_range = cfg.fade_medium_end - cfg.fade_medium_start
                        progress = (lvl.bars_alive - cfg.fade_medium_start) / fade_range
                        lvl.opacity = max(0.0, 0.8 * (1.0 - progress))
                        unexpired_active.append(lvl)
                    else:
                        lvl.status = LevelStatus.EXPIRED
                        lvl.opacity = 0.0
                        lvl.end_at_idx = c
                        lvl.end_time = cur_time

                elif lvl.strength == LevelStrength.WEAK:
                    # Weak: 15本経過後から徐々に薄くなり、25本で完全消滅
                    if lvl.bars_alive <= cfg.fade_weak_start:
                        lvl.opacity = 0.5
                        unexpired_active.append(lvl)
                    elif lvl.bars_alive < cfg.fade_weak_end:
                        fade_range = cfg.fade_weak_end - cfg.fade_weak_start
                        progress = (lvl.bars_alive - cfg.fade_weak_start) / fade_range
                        lvl.opacity = max(0.0, 0.5 * (1.0 - progress))
                        unexpired_active.append(lvl)
                    else:
                        lvl.status = LevelStatus.EXPIRED
                        lvl.opacity = 0.0
                        lvl.end_at_idx = c
                        lvl.end_time = cur_time

            active_levels = unexpired_active

            # ----------------------------------------------------
            # 3. スイングハイ/ローの検出
            # ----------------------------------------------------
            # 現在確定した足 c において、右側 N 本が満たされた過去の足 i = c - w を評価
            windows = [
                (cfg.strong_window, LevelStrength.STRONG),
                (cfg.medium_window, LevelStrength.MEDIUM),
                (cfg.weak_window, LevelStrength.WEAK),
            ]

            for w, strength_val in windows:
                i = c - w
                if i < w:
                    continue

                # --- スイングハイ（山 / 抵抗線候補）の判定 ---
                # 左側: highs[i] >= highs[j] (j in [i-w, i-1])
                # 右側: highs[i] > highs[j]  (j in [i+1, c])
                left_highs = highs[i - w : i]
                right_highs = highs[i + 1 : c + 1]
                val_h = highs[i]

                if np.all(val_h >= left_highs) and np.all(val_h > right_highs):
                    self._process_detected_level(
                        lvl_type=LevelType.RESISTANCE,
                        price=val_h,
                        peak_idx=i,
                        confirmed_idx=c,
                        strength=strength_val,
                        times=times,
                        active_levels=active_levels,
                        all_levels_history=all_levels_history,
                        existing_peak_map=existing_peak_map
                    )

                # --- スイングロー（谷 / 支持線候補）の判定 ---
                # 左側: lows[i] <= lows[j] (j in [i-w, i-1])
                # 右側: lows[i] < lows[j]  (j in [i+1, c])
                left_lows = lows[i - w : i]
                right_lows = lows[i + 1 : c + 1]
                val_l = lows[i]

                if np.all(val_l <= left_lows) and np.all(val_l < right_lows):
                    self._process_detected_level(
                        lvl_type=LevelType.SUPPORT,
                        price=val_l,
                        peak_idx=i,
                        confirmed_idx=c,
                        strength=strength_val,
                        times=times,
                        active_levels=active_levels,
                        all_levels_history=all_levels_history,
                        existing_peak_map=existing_peak_map
                    )

        # 最終足時点でのACTIVEラインの終了位置を現在足に設定
        latest_idx = n_bars - 1
        latest_time = times[latest_idx]
        for lvl in active_levels:
            lvl.end_at_idx = latest_idx
            lvl.end_time = latest_time

        if include_broken:
            return all_levels_history
        return active_levels

    def _process_detected_level(
        self,
        lvl_type: LevelType,
        price: float,
        peak_idx: int,
        confirmed_idx: int,
        strength: LevelStrength,
        times: List[pd.Timestamp],
        active_levels: List[PriceLevel],
        all_levels_history: List[PriceLevel],
        existing_peak_map: Dict[Tuple[int, LevelType], PriceLevel]
    ):
        """検出された山・谷の昇格、近接マージ、または新規登録を処理"""
        cfg = self.config

        # 1. 同一ピーク足からの昇格チェック（例: Weakとして登録済みだが、足が進んでMedium/Strong条件を満たした）
        peak_key = (peak_idx, lvl_type)
        if peak_key in existing_peak_map:
            existing = existing_peak_map[peak_key]
            if strength.value > existing.strength:
                existing.strength = strength.value
                if strength == LevelStrength.STRONG:
                    existing.opacity = 1.0
                elif strength == LevelStrength.MEDIUM:
                    existing.opacity = max(existing.opacity, 0.8)
                existing.bars_alive = 0  # 昇格により寿命リフレッシュ
            return

        # 2. 近接マージ判定（既存のアクティブな同種ラインとの距離）
        merged_line = None
        for active_lvl in active_levels:
            if active_lvl.type == lvl_type:
                # 距離計算
                diff = abs(active_lvl.price - price)
                threshold = (active_lvl.price * cfg.merge_threshold_pct) if cfg.use_pct_merge else cfg.merge_threshold_points
                if diff <= threshold:
                    merged_line = active_lvl
                    break

        if merged_line is not None:
            # 近接ラインが存在する場合は新規生成せず、既存ラインの強度を引き上げ
            merged_line.strength = min(3, merged_line.strength + 1)
            merged_line.touch_count += 1
            # 価格をより強い極値（抵抗なら高い方、支持なら低い方）に更新
            if lvl_type == LevelType.RESISTANCE:
                merged_line.price = max(merged_line.price, price)
            else:
                merged_line.price = min(merged_line.price, price)
            # 反発により寿命カウントリセット & 不透明度回復
            merged_line.bars_alive = 0
            if merged_line.strength == LevelStrength.STRONG:
                merged_line.opacity = 1.0
            elif merged_line.strength == LevelStrength.MEDIUM:
                merged_line.opacity = 0.8
            else:
                merged_line.opacity = 0.5

            existing_peak_map[peak_key] = merged_line
            return

        # 3. 新規ラインの生成
        count = len(all_levels_history) + 1
        initial_opacity = 1.0 if strength == LevelStrength.STRONG else (0.8 if strength == LevelStrength.MEDIUM else 0.5)
        new_level = PriceLevel(
            id=f"{lvl_type.value[:3].lower()}_{count}",
            type=lvl_type,
            price=price,
            created_at_idx=peak_idx,
            confirmed_at_idx=confirmed_idx,
            created_time=times[peak_idx],
            strength=strength.value,
            opacity=initial_opacity,
            status=LevelStatus.ACTIVE,
            bars_alive=0,
            touch_count=1
        )
        active_levels.append(new_level)
        all_levels_history.append(new_level)
        existing_peak_map[peak_key] = new_level

    def to_highstock_series(
        self,
        levels: List[PriceLevel],
        future_extension_ms: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Highcharts Stockで描画するためのシリーズ定義辞書の配列に変換する。
        """
        series_list = []
        cfg = self.config
        res_rgb = self._hex_to_rgb(cfg.color_resistance)
        sup_rgb = self._hex_to_rgb(cfg.color_support)

        for lvl in levels:
            if lvl.opacity <= 0.0:
                continue

            x_start = int(lvl.created_time.timestamp() * 1000)
            end_time = lvl.end_time or lvl.created_time
            x_end = int(end_time.timestamp() * 1000) + future_extension_ms

            # スタイル設定
            if lvl.strength == LevelStrength.STRONG:
                line_width = 2.0
                dash_style = "Solid"
                strength_label = "Strong"
            elif lvl.strength == LevelStrength.MEDIUM:
                line_width = 1.5
                dash_style = "Solid"
                strength_label = "Medium"
            else:
                line_width = 1.0
                dash_style = "Dash"
                strength_label = "Weak"

            # 色設定（不透明度を反映）
            base_rgb = res_rgb if lvl.type == LevelType.RESISTANCE else sup_rgb
            rgba_str = f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {lvl.opacity:.2f})"
            type_label = "抵抗線" if lvl.type == LevelType.RESISTANCE else "支持線"

            series_name = f"{type_label} ({strength_label}: ¥{lvl.price:,.0f})"
            if lvl.touch_count > 1:
                series_name += f" [反発{lvl.touch_count}回]"

            series_list.append({
                "type": "line",
                "id": f"sr_{lvl.id}",
                "name": series_name,
                "data": [
                    {"x": x_start, "y": float(lvl.price)},
                    {"x": x_end, "y": float(lvl.price)}
                ],
                "color": rgba_str,
                "lineWidth": line_width,
                "dashStyle": dash_style,
                "yAxis": 0,
                "enableMouseTracking": True,
                "showInLegend": False,
                "marker": {"enabled": False},
                "zIndex": 3 + lvl.strength,
                "tooltip": {
                    "pointFormat": (
                        f'<span style="color:{rgba_str}">●</span> '
                        f'<b>{type_label} ({strength_label})</b>: ¥{{point.y:,.0f}}<br/>'
                        f'　強度: {lvl.strength} / 不透明度: {lvl.opacity:.1%}<br/>'
                        f'　反発回数: {lvl.touch_count}回 / 経過足数: {lvl.bars_alive}本<br/>'
                    )
                }
            })

        return series_list
