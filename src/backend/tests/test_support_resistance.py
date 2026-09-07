"""
動的サポート・レジスタンス描画エンジンの単体テスト
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backend.support_resistance import (
    DynamicLevelEngine,
    DynamicLevelConfig,
    LevelType,
    LevelStrength,
    LevelStatus,
    PriceLevel,
)


class TestDynamicLevelEngine(unittest.TestCase):

    def _create_sample_df(self, prices, start_time="2026-08-01 09:00:00", freq_min=5):
        """テスト用のシンプルなOHLCV DataFrameを作成"""
        base_time = pd.to_datetime(start_time)
        times = [base_time + timedelta(minutes=i * freq_min) for i in range(len(prices))]
        
        # open, high, low, close を作成
        # prices がタプル (high, low, close) か単一数値かで分岐
        highs, lows, closes, opens = [], [], [], []
        for p in prices:
            if isinstance(p, (tuple, list)):
                h, l, c = p[0], p[1], p[2]
                o = (h + l) / 2
            else:
                h, l, c, o = p, p, p, p
            highs.append(h)
            lows.append(l)
            closes.append(c)
            opens.append(o)

        return pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * len(prices)
        })

    def test_swing_high_detection_weak(self):
        """Weak（左右3本）のスイングハイ検出テスト"""
        # 山の頂点をインデックス5に作成 (価格 100 -> 110 -> 120 -> 130 -> 140 -> 200 -> 140 -> 130 -> 120 -> 110)
        prices = [100, 110, 120, 130, 140, 200, 140, 130, 120, 110, 100]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            strong_window=15,
            medium_window=8,
            weak_window=3,
        )
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df)

        # 頂点 (200) が抵抗線として検出されているはず
        res_levels = [lvl for lvl in levels if lvl.type == LevelType.RESISTANCE]
        self.assertGreaterEqual(len(res_levels), 1)
        peak_level = next((lvl for lvl in res_levels if lvl.price == 200.0), None)
        self.assertIsNotNone(peak_level)
        self.assertEqual(peak_level.strength, LevelStrength.WEAK)
        self.assertEqual(peak_level.created_at_idx, 5)

    def test_swing_low_detection_medium(self):
        """Medium（左右8本）のスイングロー検出テスト"""
        # 谷の底をインデックス10に作成
        prices = [100 - abs(i - 10) * 5 for i in range(25)]
        # インデックス10が最安値 (100 - 0 = 100 ではなく、谷にしたいので逆向きにする)
        prices = [100 + abs(i - 10) * 5 for i in range(25)]  # index 10 is 100 (minimum)
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            strong_window=15,
            medium_window=8,
            weak_window=3,
        )
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df)

        sup_levels = [lvl for lvl in levels if lvl.type == LevelType.SUPPORT]
        self.assertGreaterEqual(len(sup_levels), 1)
        bottom_level = next((lvl for lvl in sup_levels if lvl.price == 100.0), None)
        self.assertIsNotNone(bottom_level)
        self.assertGreaterEqual(bottom_level.strength, LevelStrength.MEDIUM)

    def test_break_invalidation(self):
        """ブレイクによるライン無効化（BROKEN）テスト"""
        # 山（価格150）を作成した後、後続の足で終値が160になって上抜けブレイクする
        prices = [
            (100, 100, 100),
            (110, 110, 110),
            (120, 120, 120),
            (130, 130, 130),
            (150, 140, 145), # index 4: High 150 (peak)
            (130, 130, 130),
            (120, 120, 120),
            (110, 110, 110), # index 7: confirmed weak resistance at 150
            (120, 120, 120),
            (160, 140, 155), # index 9: Close 155 > 150 -> BROKEN!
            (130, 130, 130)
        ]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(weak_window=3)
        engine = DynamicLevelEngine(cfg)
        
        # デフォルト（ACTIVEのみ）では消滅しているはず
        active_levels = engine.calculate_levels(df, include_broken=False)
        self.assertFalse(any(lvl.price == 150.0 for lvl in active_levels))

        # include_broken=True では BROKEN 状態で取得できるはず
        all_levels = engine.calculate_levels(df, include_broken=True)
        broken_peak = next((lvl for lvl in all_levels if lvl.price == 150.0), None)
        self.assertIsNotNone(broken_peak)
        self.assertEqual(broken_peak.status, LevelStatus.BROKEN)
        self.assertEqual(broken_peak.end_at_idx, 9)

    def test_fadeout_weak(self):
        """Weakラインの経時フェードアウト（15本後から減衰、25本で消滅）テスト"""
        # 山を作った後、ブレイクせずに26本以上レンジ相場を継続
        prices = [100, 110, 120, 130, 200, 130, 120, 110] # peak at index 4 (confirmed at index 7)
        flat_prices = [100] * 30 # 30 bars flat (well beyond 25 bars)
        df = self._create_sample_df(prices + flat_prices)

        cfg = DynamicLevelConfig(
            weak_window=3,
            fade_weak_start=15,
            fade_weak_end=25
        )
        engine = DynamicLevelEngine(cfg)
        
        # 25本経過したのでEXPIREDになり、ACTIVEからは除外されているはず
        active_levels = engine.calculate_levels(df, include_broken=False)
        self.assertFalse(any(lvl.price == 200.0 for lvl in active_levels))

        # 履歴にはEXPIREDとして記録されていること
        all_levels = engine.calculate_levels(df, include_broken=True)
        expired_peak = next((lvl for lvl in all_levels if lvl.price == 200.0), None)
        self.assertIsNotNone(expired_peak)
        self.assertEqual(expired_peak.status, LevelStatus.EXPIRED)
        self.assertEqual(expired_peak.opacity, 0.0)

    def test_proximity_merge(self):
        """近接マージテスト（同値近辺の山がマージされて強度上昇・反発回数加算・極値更新）"""
        # 1回目の山: 200 (index 4)
        # 2回目の山: 202 (index 13, 差額2円 <= 閾値20円、ただし終値195でブレイクせず反発)
        prices = [
            # 1つ目の山 (High 200, Close 190)
            100, 110, 120, 130, (200, 180, 190), 130, 120, 110, 100,
            # 谷を挟んで2つ目の山 (High 202, Close 195: 終値は抵抗線200以下で反発)
            110, 120, 130, 140, (202, 185, 195), 140, 130, 120, 110, 100
        ]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            weak_window=3,
            merge_threshold_points=20.0
        )
        engine = DynamicLevelEngine(cfg)
        active_levels = engine.calculate_levels(df)

        res_levels = [lvl for lvl in active_levels if lvl.type == LevelType.RESISTANCE]
        # 2つに分裂せず、1つのマージされたラインになっていること
        self.assertEqual(len(res_levels), 1)
        merged = res_levels[0]
        # 価格がより高い極値（202）に更新されていること
        self.assertEqual(merged.price, 202.0)
        # 反発回数が 2 回になっていること
        self.assertEqual(merged.touch_count, 2)
        # 強度が 1(Weak) から 2(Medium) に引き上げられていること
        self.assertEqual(merged.strength, LevelStrength.MEDIUM)

    def test_to_highstock_series(self):
        """Highcharts シリーズへの変換テスト"""
        prices = [100, 110, 120, 130, 200, 130, 120, 110, 100]
        df = self._create_sample_df(prices)
        engine = DynamicLevelEngine(DynamicLevelConfig(weak_window=3))
        levels = engine.calculate_levels(df)

        series_list = engine.to_highstock_series(levels)
        self.assertGreaterEqual(len(series_list), 1)
        s = series_list[0]
        self.assertEqual(s["type"], "line")
        self.assertIn("data", s)
        self.assertEqual(len(s["data"]), 2)
        self.assertEqual(s["data"][0]["y"], 200.0)
        self.assertEqual(s["data"][1]["y"], 200.0)
        self.assertIn("rgba", s["color"])


    def test_strong_persistence(self):
        """Strong（左右15本）がブレイクされない限り100本以上経過してもACTIVEのままであることのテスト"""
        # 山の頂点 (200) を作成（左右15本）
        prices = [100 + i * 5 for i in range(15)] + [(200, 190, 195)] + [195 - i * 5 for i in range(15)]
        # その後、100本間ブレイクせずに100円付近で推移
        flat_prices = [100] * 100
        df = self._create_sample_df(prices + flat_prices)

        cfg = DynamicLevelConfig(
            strong_window=15,
            medium_window=8,
            weak_window=3
        )
        engine = DynamicLevelEngine(cfg)
        active_levels = engine.calculate_levels(df)

        strong_peaks = [lvl for lvl in active_levels if lvl.price == 200.0]
        self.assertEqual(len(strong_peaks), 1)
        peak = strong_peaks[0]
        self.assertEqual(peak.strength, LevelStrength.STRONG)
        self.assertEqual(peak.status, LevelStatus.ACTIVE)
        self.assertEqual(peak.opacity, 1.0)
        self.assertGreaterEqual(peak.bars_alive, 100)

    def test_flat_bar_handling(self):
        """同値（フラットバー）が連続する場合でも山を取りこぼさないことのテスト"""
        # 山の頂点が2本連続で同値 (200, 200)
        prices = [100, 110, 120, 130, 200, 200, 130, 120, 110, 100]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(weak_window=3)
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df)

        res_levels = [lvl for lvl in levels if lvl.type == LevelType.RESISTANCE and lvl.price == 200.0]
        self.assertGreaterEqual(len(res_levels), 1)

    def test_large_dataset_performance(self):
        """3,000本の大規模データに対する高速実行テスト（0.5秒以内）"""
        import time
        np.random.seed(42)
        n = 3000
        # ランダムウォーク価格データ生成
        returns = np.random.normal(0, 5, n)
        raw_prices = 38000 + np.cumsum(returns)
        highs = raw_prices + np.random.uniform(0, 15, n)
        lows = raw_prices - np.random.uniform(0, 15, n)
        closes = np.clip(raw_prices + np.random.normal(0, 5, n), lows, highs)
        opens = raw_prices

        base_time = pd.to_datetime("2026-01-01 09:00:00")
        times = [base_time + timedelta(minutes=i * 5) for i in range(n)]

        df = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * n
        })

        engine = DynamicLevelEngine()
        t0 = time.time()
        levels = engine.calculate_levels(df)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.5, f"計算時間が長すぎます: {elapsed:.3f}秒")
        # 正常にラインが算出されていること
        self.assertGreater(len(levels), 0)


if __name__ == "__main__":
    unittest.main()

