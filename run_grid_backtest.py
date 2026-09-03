"""
日経225マイクロ先物「高値安値ライン突破・逆張りグリッドナンピン戦略」バックテスト実行スクリプト
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

# Windowsのコンソール出力エンコーディングをUTF-8に設定
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# プロジェクトルートをインポートパスに追加
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.backend.data_parser import DataParser
from src.backend.grid_nanpin_strategy import GridNanpinConfig, GridNanpinBacktester

def main():
    parser = argparse.ArgumentParser(description="日経225マイクロ先物 逆張り/順張りIFグリッドナンピン戦略 バックテスト実行")
    parser.add_argument("--data", type=str, default="data/market/N225microf_2026.xlsx", help="データファイルのパス")
    parser.add_argument("--sheet", type=str, default="15min", help="Excelのシート名 (時間足: 1min, 5min, 15min, 60min 等)")
    parser.add_argument("--entry_mode", type=str, default="contrarian", choices=["contrarian", "trend"], help="エントリー方向 (contrarian: 逆張り, trend: 順張りIFモード)")
    parser.add_argument("--n_period", type=int, default=20, help="基準線(High/Low)の期間N (最大500)")
    parser.add_argument("--entry_delta", type=float, default=200.0, help="エントリー突破幅 (円)")
    parser.add_argument("--grid_step", type=float, default=250.0, help="ナンピン逆行幅 (円)")
    parser.add_argument("--tp_delta", type=float, default=200.0, help="平均取得価格からの利確幅 (円, 最大2000円, 100円刻み)")
    parser.add_argument("--sl_amount", type=float, default=500000.0, help="損切含み損金額 (円)")
    parser.add_argument("--emergency_exit", action="store_true", help="【試験実装】ナンピン段階が半分を超えたらゼロ円清算を目指す（同値撤退モード）")
    parser.add_argument("--lots", type=str, default="1,1,2,2,3,3,3,4,4,5,5,6,7,8,9,10", help="ナンピンロットテーブル (カンマ区切り)")
    parser.add_argument("--capital", type=float, default=3000000.0, help="初期資金 (円)")
    parser.add_argument("--margin", type=float, default=25000.0, help="必要証拠金 (円/枚)")
    parser.add_argument("--fee", type=float, default=11.0, help="片道手数料 (円/枚, デフォルト: 11.0円)")
    parser.add_argument("--output_dir", type=str, default="output", help="レポート・画像保存先ディレクトリ")
    parser.add_argument("--max_bars_plot", type=int, default=2000, help="チャート描画に含める最大バー数")
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root_dir / data_path
        
    print("=" * 65)
    print(f"📊 データ読み込み中: {data_path.name} (足種: {args.sheet})")
    print("=" * 65)
    
    if not data_path.exists():
        print(f"❌ エラー: ファイルが存在しません: {data_path}")
        return
        
    df = DataParser.load_file(data_path, sheet_name=args.sheet)
    print(f"✅ データ読み込み完了: {len(df):,} 行 ({df['time'].min()} 〜 {df['time'].max()})")
    
    # ロットテーブルのパース
    try:
        lot_table = [int(x.strip()) for x in args.lots.split(',') if x.strip()]
    except:
        lot_table = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]

    mode_val = "TREND" if args.entry_mode.lower() == "trend" else "CONTRARIAN"

    # 戦略設定の初期化
    config = GridNanpinConfig(
        entry_mode=mode_val,
        n_period=args.n_period,
        entry_delta=args.entry_delta,
        grid_step=args.grid_step,
        lot_table=lot_table,
        tp_delta=args.tp_delta,
        sl_amount=args.sl_amount,
        emergency_breakeven_exit=args.emergency_exit,
        multiplier=10.0,     # 日経225マイクロ
        tick_size=5.0,       # 呼値5円
        fee_per_lot=args.fee, # 片道手数料
        initial_capital=args.capital,
        margin_per_lot=args.margin
    )
    
    print("\n🚀 バックテスト実行中...")
    engine = GridNanpinBacktester(config)
    result = engine.run(df)
    
    # 結果のコンソール出力
    result.print_summary()
    
    # 出力先ディレクトリの作成
    out_dir = root_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 売買シグナル付きチャート画像の保存
    chart_img_path = out_dir / "grid_nanpin_backtest_chart.png"
    result.plot_chart(save_path=chart_img_path, max_bars=args.max_bars_plot)
    
    # 2. ナンピン分布画像の保存
    nanpin_img_path = out_dir / "nanpin_distribution.png"
    result.plot_nanpin_distribution(save_path=nanpin_img_path)
    
    # 3. トレード履歴CSVの保存
    if not result.trades_df.empty:
        csv_path = out_dir / "trade_logs.csv"
        result.trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"📝 取引履歴ログを保存しました: {csv_path.resolve()}")
        
    print("\n✨ バックテストおよびチャート生成が完了しました！")

if __name__ == "__main__":
    main()
