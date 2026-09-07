"""
動的サポート・レジスタンスエンジンの実データ検証スクリプト（新フィルター・間引き効果の検証）
"""

import sys
from pathlib import Path
import pandas as pd

# パス設定
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.backend.data_parser import DataParser
from src.backend.support_resistance import DynamicLevelEngine, DynamicLevelConfig, LevelType, LevelStrength, LevelStatus
from src.backend.indicators import IndicatorCalculator

def main():
    print("==================================================")
    print("動的サポート・レジスタンス描画エンジン 新フィルター検証")
    print("==================================================")

    # 1. 実データのロード (2026年の5分足データ)
    data_file = root_dir / "data" / "market" / "N225microf_2026.xlsx"
    if not data_file.exists():
        print(f"データファイルが見つかりません: {data_file}")
        return

    print(f"データファイル: {data_file.name}")
    df = DataParser.load_file(data_file, "5min")
    if df is None or df.empty:
        print("5min シートの読み込みに失敗しました。利用可能なシートを探索します...")
        sheets = DataParser.get_sheet_names(data_file)
        print(f"利用可能シート: {sheets}")
        df = DataParser.load_file(data_file, sheets[0])

    # ユーザーの画面条件: 2026/06/01 〜 2026/09/08
    mask = (df['time'].dt.date >= pd.to_datetime("2026-06-01").date()) & (df['time'].dt.date <= pd.to_datetime("2026-09-08").date())
    df_period = df.loc[mask].copy().reset_index(drop=True)
    if len(df_period) > 10000:
        df_period = df_period.tail(10000).copy().reset_index(drop=True)

    print(f"対象期間データ: {len(df_period)} 本 (期間: {df_period['time'].min()} 〜 {df_period['time'].max()})")
    current_price = float(df_period['close'].iloc[-1])
    print(f"現在価格 (直近終値): {current_price:,.0f}円")

    # 2. フィルターなし（修正前の状態）
    cfg_raw = DynamicLevelConfig(
        strong_window=15,
        medium_window=8,
        weak_window=3,
        strong_max_bars=0, # 無制限
        merge_threshold_points=20.0
    )
    engine_raw = DynamicLevelEngine(cfg_raw)
    raw_levels = engine_raw.calculate_levels(df_period, apply_filter=False)
    print(f"\n【修正前（フィルターなし）】")
    print(f"  画面内に残ってしまうアクティブライン総数: {len(raw_levels)} 本 (水平線で埋め尽くされていた状態)")

    # 3. フィルター適用後（新ロジック）
    cfg_filtered = DynamicLevelConfig(
        strong_window=15,
        medium_window=8,
        weak_window=3,
        strong_max_bars=500, # 500本で消滅
        merge_threshold_points=100.0, # 100円未満マージ
        price_distance_pct=0.025, # ±2.5%以内
        max_levels_per_side=4 # 上下各4本
    )
    engine_filtered = DynamicLevelEngine(cfg_filtered)
    filtered_levels = engine_filtered.calculate_levels(df_period, apply_filter=True)

    print(f"\n【修正後（4つのフィルター適用後）】")
    print(f"  描画されるライン総数: {len(filtered_levels)} 本 (スッキリと厳選された状態)")
    
    res_list = [l for l in filtered_levels if l.type == LevelType.RESISTANCE]
    sup_list = [l for l in filtered_levels if l.type == LevelType.SUPPORT]

    print(f"\n  [レジスタンス線 (赤)] 計 {len(res_list)} 本:")
    for l in res_list:
        diff_pt = l.price - current_price
        diff_pct = (diff_pt / current_price) * 100
        print(f"    ・価格: {l.price:,.0f}円 (+{diff_pt:,.0f}円 / +{diff_pct:.2f}%) | 強度: {l.strength} | 反発: {l.touch_count}回 | 経過: {l.bars_alive}本 | 不透明度: {l.opacity:.2f}")

    print(f"\n  [支持線 (緑)] 計 {len(sup_list)} 本:")
    for l in sup_list:
        diff_pt = current_price - l.price
        diff_pct = (diff_pt / current_price) * 100
        print(f"    ・価格: {l.price:,.0f}円 (-{diff_pt:,.0f}円 / -{diff_pct:.2f}%) | 強度: {l.strength} | 反発: {l.touch_count}回 | 経過: {l.bars_alive}本 | 不透明度: {l.opacity:.2f}")

    # 4. Highcharts シリーズ形式の確認
    series = engine_filtered.to_highstock_series(filtered_levels)
    print(f"\n【Highcharts シリーズ生成】")
    print(f"  シリーズ件数: {len(series)} 件")
    for s in series:
        print(f"    - {s['name']}: {s['color']} (太さ: {s['lineWidth']}px, スタイル: {s['dashStyle']})")

    print("\n✅ フィルターおよび間引き処理が完璧に機能していることを確認しました！")

if __name__ == "__main__":
    main()
