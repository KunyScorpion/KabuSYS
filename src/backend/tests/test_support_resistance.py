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
        prices = [100, 110, 120, 130, 140, 200, 140, 130, 120, 110, 100]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            strong_window=15,
            medium_window=8,
            weak_window=3,
        )
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df, apply_filter=False)

        res_levels = [lvl for lvl in levels if lvl.type == LevelType.RESISTANCE]
        self.assertGreaterEqual(len(res_levels), 1)
        peak_level = next((lvl for lvl in res_levels if lvl.price == 200.0), None)
        self.assertIsNotNone(peak_level)
        self.assertEqual(peak_level.strength, LevelStrength.WEAK)
        self.assertEqual(peak_level.created_at_idx, 5)

    def test_swing_low_detection_medium(self):
        """Medium（左右8本）のスイングロー検出テスト"""
        prices = [100 + abs(i - 10) * 5 for i in range(25)]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            strong_window=15,
            medium_window=8,
            weak_window=3,
        )
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df, apply_filter=False)

        sup_levels = [lvl for lvl in levels if lvl.type == LevelType.SUPPORT]
        self.assertGreaterEqual(len(sup_levels), 1)
        bottom_level = next((lvl for lvl in sup_levels if lvl.price == 100.0), None)
        self.assertIsNotNone(bottom_level)
        self.assertGreaterEqual(bottom_level.strength, LevelStrength.MEDIUM)

    def test_break_invalidation(self):
        """ブレイクによるライン無効化（BROKEN）テスト"""
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
        
        active_levels = engine.calculate_levels(df, include_broken=False, apply_filter=False)
        self.assertFalse(any(lvl.price == 150.0 for lvl in active_levels))

        all_levels = engine.calculate_levels(df, include_broken=True)
        broken_peak = next((lvl for lvl in all_levels if lvl.price == 150.0), None)
        self.assertIsNotNone(broken_peak)
        self.assertEqual(broken_peak.status, LevelStatus.BROKEN)
        self.assertEqual(broken_peak.end_at_idx, 9)

    def test_fadeout_weak(self):
        """Weakラインの経時フェードアウト（15本後から減衰、25本で消滅）テスト"""
        prices = [100, 110, 120, 130, 200, 130, 120, 110]
        flat_prices = [100] * 30
        df = self._create_sample_df(prices + flat_prices)

        cfg = DynamicLevelConfig(
            weak_window=3,
            fade_weak_start=15,
            fade_weak_end=25
        )
        engine = DynamicLevelEngine(cfg)
        
        active_levels = engine.calculate_levels(df, include_broken=False, apply_filter=False)
        self.assertFalse(any(lvl.price == 200.0 for lvl in active_levels))

        all_levels = engine.calculate_levels(df, include_broken=True)
        expired_peak = next((lvl for lvl in all_levels if lvl.price == 200.0), None)
        self.assertIsNotNone(expired_peak)
        self.assertEqual(expired_peak.status, LevelStatus.EXPIRED)
        self.assertEqual(expired_peak.opacity, 0.0)

    def test_proximity_merge(self):
        """近接マージテスト（同値近辺の山がマージされて強度上昇・反発回数加算・極値更新）"""
        prices = [
            100, 110, 120, 130, (200, 180, 190), 130, 120, 110, 100,
            110, 120, 130, 140, (202, 185, 195), 140, 130, 120, 110, 100
        ]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            weak_window=3,
            merge_threshold_points=20.0
        )
        engine = DynamicLevelEngine(cfg)
        active_levels = engine.calculate_levels(df, apply_filter=False)

        res_levels = [lvl for lvl in active_levels if lvl.type == LevelType.RESISTANCE]
        self.assertEqual(len(res_levels), 1)
        merged = res_levels[0]
        self.assertEqual(merged.price, 202.0)
        self.assertEqual(merged.touch_count, 2)
        self.assertEqual(merged.strength, LevelStrength.MEDIUM)

    def test_to_highstock_series(self):
        """Highcharts シリーズへの変換テスト"""
        prices = [100, 110, 120, 130, 200, 130, 120, 110, 100]
        df = self._create_sample_df(prices)
        engine = DynamicLevelEngine(DynamicLevelConfig(weak_window=3))
        levels = engine.calculate_levels(df, apply_filter=False)

        series_list = engine.to_highstock_series(levels)
        self.assertGreaterEqual(len(series_list), 1)
        s = series_list[0]
        self.assertEqual(s["type"], "line")
        self.assertIn("data", s)
        self.assertEqual(len(s["data"]), 2)
        self.assertEqual(s["data"][0]["y"], 200.0)
        self.assertEqual(s["data"][1]["y"], 200.0)
        self.assertIn("rgba", s["color"])

    def test_strong_persistence_and_max_bars(self):
        """Strong（左右15本）がブレイクされずに放置された場合の最大寿命（strong_max_bars）テスト"""
        # 山の頂点 (200) を作成（左右15本）
        prices = [100 + i * 5 for i in range(15)] + [(200, 190, 195)] + [195 - i * 5 for i in range(15)]
        
        # 100本経過時点（まだ消えていない）
        df_100 = self._create_sample_df(prices + [100] * 100)
        cfg = DynamicLevelConfig(
            strong_window=15,
            medium_window=8,
            weak_window=3,
            strong_max_bars=300
        )
        engine = DynamicLevelEngine(cfg)
        lvls_100 = engine.calculate_levels(df_100, apply_filter=False)
        self.assertTrue(any(l.price == 200.0 and l.status == LevelStatus.ACTIVE for l in lvls_100))

        # 301本経過時点（strong_max_bars=300 を超えてEXPIREDになる）
        df_350 = self._create_sample_df(prices + [100] * 350)
        lvls_350 = engine.calculate_levels(df_350, apply_filter=False)
        self.assertFalse(any(l.price == 200.0 for l in lvls_350))

        # include_broken=True では EXPIRED として記録されていること
        all_lvls = engine.calculate_levels(df_350, include_broken=True)
        expired_strong = next((l for l in all_lvls if l.price == 200.0), None)
        self.assertIsNotNone(expired_strong)
        self.assertEqual(expired_strong.status, LevelStatus.EXPIRED)

    def test_flat_bar_handling(self):
        """同値（フラットバー）が連続する場合でも山を取りこぼさないことのテスト"""
        prices = [100, 110, 120, 130, 200, 200, 130, 120, 110, 100]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(weak_window=3)
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df, apply_filter=False)

        res_levels = [lvl for lvl in levels if lvl.type == LevelType.RESISTANCE and lvl.price == 200.0]
        self.assertGreaterEqual(len(res_levels), 1)

    def test_price_distance_filter(self):
        """現在価格からの距離フィルター（±2.5%）テスト"""
        # 現在価格 65,000円
        # ラインA: 66,000円 (距離 1.5% -> 保持)
        # ラインB: 70,000円 (距離 7.7% -> 除外)
        # ラインC: 64,000円 (距離 1.5% -> 保持)
        # ラインD: 60,000円 (距離 7.7% -> 除外)
        levels = [
            PriceLevel("r1", LevelType.RESISTANCE, 66000.0, 0, 3, pd.Timestamp.now(), 2, 0.8, LevelStatus.ACTIVE),
            PriceLevel("r2", LevelType.RESISTANCE, 70000.0, 0, 3, pd.Timestamp.now(), 3, 1.0, LevelStatus.ACTIVE),
            PriceLevel("s1", LevelType.SUPPORT, 64000.0, 0, 3, pd.Timestamp.now(), 2, 0.8, LevelStatus.ACTIVE),
            PriceLevel("s2", LevelType.SUPPORT, 60000.0, 0, 3, pd.Timestamp.now(), 3, 1.0, LevelStatus.ACTIVE),
        ]
        cfg = DynamicLevelConfig(price_distance_pct=0.025) # ±2.5%
        engine = DynamicLevelEngine(cfg)

        filtered = engine.filter_for_display(levels, current_price=65000.0)
        prices = [l.price for l in filtered]
        self.assertIn(66000.0, prices)
        self.assertIn(64000.0, prices)
        self.assertNotIn(70000.0, prices)
        self.assertNotIn(60000.0, prices)

    def test_max_levels_per_side(self):
        """最大保持本数（スロット数：上下各4本）制限テスト"""
        # 現在価格 65,000円に対し、抵抗線が6本、支持線が6本ある場合
        current_p = 65000.0
        resistances = [
            PriceLevel(f"r{i}", LevelType.RESISTANCE, current_p + (i + 1) * 50, 0, 3, pd.Timestamp.now(), 2, 0.8, LevelStatus.ACTIVE)
            for i in range(6) # 65050, 65100, 65150, 65200, 65250, 65300
        ]
        supports = [
            PriceLevel(f"s{i}", LevelType.SUPPORT, current_p - (i + 1) * 50, 0, 3, pd.Timestamp.now(), 2, 0.8, LevelStatus.ACTIVE)
            for i in range(6) # 64950, 64900, 64850, 64800, 64750, 64700
        ]
        cfg = DynamicLevelConfig(
            max_levels_per_side=4,
            merge_threshold_points=10.0 # マージを避けるため小さめに設定
        )
        engine = DynamicLevelEngine(cfg)

        filtered = engine.filter_for_display(resistances + supports, current_price=current_p)
        filtered_res = [l for l in filtered if l.type == LevelType.RESISTANCE]
        filtered_sup = [l for l in filtered if l.type == LevelType.SUPPORT]

        # 上下それぞれ最大4本に抑えられていること
        self.assertEqual(len(filtered_res), 4)
        self.assertEqual(len(filtered_sup), 4)
        # 現在価格に近い順に選ばれていること
        self.assertEqual([l.price for l in filtered_res], [65050.0, 65100.0, 65150.0, 65200.0])
        self.assertEqual([l.price for l in filtered_sup], [64950.0, 64900.0, 64850.0, 64800.0])

    def test_close_levels_merge_100yen(self):
        """100円未満の近接ラインが二次マージで1本に統合されることのテスト"""
        current_p = 65000.0
        # 65,100円 と 65,140円（差額40円 < 100円）の2本の抵抗線
        levels = [
            PriceLevel("r1", LevelType.RESISTANCE, 65100.0, 0, 3, pd.Timestamp.now(), 1, 0.5, LevelStatus.ACTIVE, touch_count=1),
            PriceLevel("r2", LevelType.RESISTANCE, 65140.0, 0, 3, pd.Timestamp.now(), 2, 0.8, LevelStatus.ACTIVE, touch_count=2),
        ]
        cfg = DynamicLevelConfig(merge_threshold_points=100.0)
        engine = DynamicLevelEngine(cfg)

        filtered = engine.filter_for_display(levels, current_price=current_p)
        self.assertEqual(len(filtered), 1)
        merged = filtered[0]
        # 極値（65,140円）に更新されていること
        self.assertEqual(merged.price, 65140.0)
        # 反発回数が合算 (1 + 2 = 3回) されていること
        self.assertEqual(merged.touch_count, 3)
        # 強度が最大値 (Medium=2) になっていること
        self.assertEqual(merged.strength, LevelStrength.MEDIUM)

    def test_large_dataset_performance(self):
        """3,000本の大規模データに対する高速実行テスト（0.5秒以内）"""
        import time
        np.random.seed(42)
        n = 3000
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
        self.assertGreater(len(levels), 0)

    def test_past_significant_levels_included(self):
        """過去にブレイクされた有意なラインが描画用リストに含まれることのテスト"""
        # 山（200円）ができた後、ブレイク（250円）されてBROKENになるデータ
        # 左右3本でMedium山として認識させ、11本生存後にブレイク
        prices = [100, 120, 150, 200, 150, 120, 100] + [100] * 12 + [250, 260, 270]
        df = self._create_sample_df(prices)

        cfg = DynamicLevelConfig(
            medium_window=3,
            weak_window=2,
            include_past_levels=True,
            max_past_levels=10
        )
        engine = DynamicLevelEngine(cfg)
        levels = engine.calculate_levels(df, apply_filter=True)

        # 過去にブレイクされた200円のラインが結果に含まれていること
        past_broken = [l for l in levels if l.price == 200.0 and l.status == LevelStatus.BROKEN]
        self.assertEqual(len(past_broken), 1)
        self.assertIsNotNone(past_broken[0].end_time)


if __name__ == "__main__":
    unittest.main()
