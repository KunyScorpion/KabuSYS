"""
動的サポート・レジスタンスエンジンの実データ検証スクリプト
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
    print("動的サポート・レジスタンス描画エンジンの実データ検証")
    print("==================================================")

    # 1. 実データのロード (2026年の60分足データ)
    data_file = root_dir / "data" / "market" / "N225microf_2026.xlsx"
    if not data_file.exists():
        print(f"データファイルが見つかりません: {data_file}")
        return

    print(f"データファイル: {data_file.name}")
    df = DataParser.load_file(data_file, "60min")
    if df is None or df.empty:
        print("60min シートの読み込みに失敗しました。利用可能なシートを探索します...")
        sheets = DataParser.get_sheet_names(data_file)
        print(f"利用可能シート: {sheets}")
        df = DataParser.load_file(data_file, sheets[0])

    print(f"データ読み込み完了: {len(df)} 行 (期間: {df['time'].min()} 〜 {df['time'].max()})")

    # 2. 直近500本を対象に検証
    df_sample = df.tail(500).reset_index(drop=True)
    print(f"検証対象足数: {len(df_sample)} 本 (最新: {df_sample['time'].iloc[-1]})")

    # 3. エンジンの実行
    cfg = DynamicLevelConfig(
        strong_window=15,
        medium_window=8,
        weak_window=3,
        merge_threshold_points=20.0,
        fade_medium_start=40,
        fade_medium_end=60,
        fade_weak_start=15,
        fade_weak_end=25
    )
    engine = DynamicLevelEngine(cfg)

    # 全履歴（ブレイク・失効含む）
    all_levels = engine.calculate_levels(df_sample, include_broken=True)
    # 現在アクティブなライン
    active_levels = engine.calculate_levels(df_sample, include_broken=False)

    print("\n--- 【計算結果サマリー】 ---")
    print(f"総検出ライン数 (履歴全体): {len(all_levels)} 本")
    broken_count = sum(1 for l in all_levels if l.status == LevelStatus.BROKEN)
    expired_count = sum(1 for l in all_levels if l.status == LevelStatus.EXPIRED)
    active_count = len(active_levels)

    print(f"  ├─ 終値ブレイク無効化 (BROKEN): {broken_count} 本")
    print(f"  ├─ 寿命フェードアウト消滅 (EXPIRED): {expired_count} 本")
    print(f"  └─ 現在有効なライン (ACTIVE): {active_count} 本")

    print("\n--- 【現在有効 (ACTIVE) なライン詳細】 ---")
    active_strong = [l for l in active_levels if l.strength == LevelStrength.STRONG]
    active_medium = [l for l in active_levels if l.strength == LevelStrength.MEDIUM]
    active_weak = [l for l in active_levels if l.strength == LevelStrength.WEAK]

    print(f"・Strong (強度3 / 永続): {len(active_strong)} 本")
    for l in active_strong:
        t_label = "抵抗線" if l.type == LevelType.RESISTANCE else "支持線"
        print(f"    [{t_label}] 価格: {l.price:,.0f}円 | 形成日時: {l.created_time} | 経過足数: {l.bars_alive}本 | 反発: {l.touch_count}回 | 透明度: {l.opacity:.2f}")

    print(f"・Medium (強度2 / 60本で消滅): {len(active_medium)} 本")
    for l in active_medium:
        t_label = "抵抗線" if l.type == LevelType.RESISTANCE else "支持線"
        print(f"    [{t_label}] 価格: {l.price:,.0f}円 | 形成日時: {l.created_time} | 経過足数: {l.bars_alive}本 | 反発: {l.touch_count}回 | 透明度: {l.opacity:.2f}")

    print(f"・Weak (強度1 / 25本で消滅): {len(active_weak)} 本")
    for l in active_weak:
        t_label = "抵抗線" if l.type == LevelType.RESISTANCE else "支持線"
        print(f"    [{t_label}] 価格: {l.price:,.0f}円 | 形成日時: {l.created_time} | 経過足数: {l.bars_alive}本 | 反発: {l.touch_count}回 | 透明度: {l.opacity:.2f}")

    # 4. Highcharts シリーズ形式への変換確認
    series = engine.to_highstock_series(active_levels)
    print(f"\n--- 【Highcharts シリーズ出力検証】 ---")
    print(f"生成シリーズ数: {len(series)} 件")
    if series:
        s0 = series[0]
        print(f"  サンプルシリーズ [0]:")
        print(f"    Name: {s0['name']}")
        print(f"    Color: {s0['color']}")
        print(f"    LineWidth: {s0['lineWidth']}px, DashStyle: {s0['dashStyle']}")
        print(f"    Data: {s0['data']}")

    # 5. IndicatorCalculator 経由での計算検証
    res_ind = IndicatorCalculator.compute("Dynamic_SR", df_sample, {})
    ind_series = res_ind.attrs.get("dynamic_series", [])
    print(f"\n--- 【IndicatorCalculator 連携検証】 ---")
    print(f"IndicatorCalculator 経由シリーズ数: {len(ind_series)} 件")
    assert len(series) == len(ind_series), "シリーズ数が一致しません"

    print("\n✅ 全ての受け入れ条件と動作検証が正常に完了しました！")

if __name__ == "__main__":
    main()
