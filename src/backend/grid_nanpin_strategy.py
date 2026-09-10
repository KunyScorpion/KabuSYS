import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

@dataclass
class GridNanpinConfig:
    """
    高値安値ライン突破・逆張り/順張りグリッドナンピン戦略の設定クラス
    """
    name: str = "高値安値ライン突破・逆張りグリッドナンピン戦略"
    
    # エントリー方向設定 ("CONTRARIAN": 逆張り, "TREND": 順張り IFモード)
    entry_mode: str = "CONTRARIAN"
    
    # 基準線設定
    n_period: int = 20           # 直近N期間のHigh/Lowライン
    
    # エントリー設定
    entry_delta: float = 200.0   # 基準線突破幅 (逆張り: High+200円でショート, Low-200円でロング / 順張りIF: High+200円でロング, Low-200円でショート)
    
    # ナンピン設定 (ロング・ショート共通のロットテーブル)
    grid_step: float = 250.0     # 逆行幅 (250円ごとにナンピン)
    lot_table: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10])
    
    # 後方互換用
    short_lot_table: Optional[List[int]] = None
    long_lot_table: Optional[List[int]] = None
    
    # エグジット設定 (値幅のみに固定)
    tp_delta: float = 200.0      # 平均取得価格からの利確幅 (+200円)
    tp_use_baseline: bool = False # 基準線への戻り利確 (値幅のみ固定のため無効)
    tp_baseline_type: str = "none"
    sl_amount: float = 500000.0  # ポジション全体含み損での損切金額 (500,000円)
    emergency_breakeven_exit: bool = False # 【試験実装】ナンピン段階が半分を超えたらゼロ円清算を目指す

    # 取引対象・コスト設定 (日経225マイクロ先物)
    multiplier: float = 10.0     # 乗数 (10倍: 1pt = 10円)
    tick_size: float = 5.0       # 呼値 (5円刻み)
    fee_per_lot: float = 11.0    # 片道手数料 (11円/枚: 主要ネット証券標準)
    slippage: float = 0.0        # スリッページ (pt)
    initial_capital: float = 3000000.0 # 初期資本金 (3,000,000円)
    margin_per_lot: float = 25000.0    # 必要証拠金 (25,000円/枚)

    def __post_init__(self):
        # entry_mode の正規化と戦略名更新
        if str(self.entry_mode).upper() in ["TREND", "FOLLOW", "順張り"]:
            self.entry_mode = "TREND"
            if self.name == "高値安値ライン突破・逆張りグリッドナンピン戦略":
                self.name = "高値安値ライン突破・順張りIFグリッドナンピン戦略"
        else:
            self.entry_mode = "CONTRARIAN"
            
        # short_lot_table / long_lot_table が指定されている場合は lot_table に同期
        if self.short_lot_table is not None and self.lot_table == [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]:
            self.lot_table = self.short_lot_table
        elif self.long_lot_table is not None and self.lot_table == [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]:
            self.lot_table = self.long_lot_table

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_mode": self.entry_mode,
            "n_period": self.n_period,
            "entry_delta": self.entry_delta,
            "grid_step": self.grid_step,
            "lot_table": self.lot_table,
            "tp_delta": self.tp_delta,
            "sl_amount": self.sl_amount,
            "emergency_breakeven_exit": self.emergency_breakeven_exit,
            "multiplier": self.multiplier,
            "tick_size": self.tick_size,
            "fee_per_lot": self.fee_per_lot,
            "slippage": self.slippage,
            "initial_capital": self.initial_capital,
            "margin_per_lot": self.margin_per_lot
        }


@dataclass
class PositionEntry:
    level: int           # ナンピン階層 (0: 初回エントリー, 1: 1回目ナンピン...)
    time: pd.Timestamp
    price: float
    lots: int
    fee: float

@dataclass
class Position:
    pos_type: str        # 'LONG' or 'SHORT'
    entries: List[PositionEntry] = field(default_factory=list)
    initial_baseline: float = 0.0 # エントリー時の基準線価格
    
    @property
    def total_lots(self) -> int:
        return sum(e.lots for e in self.entries)
    
    @property
    def avg_price(self) -> float:
        if self.total_lots == 0:
            return 0.0
        return sum(e.price * e.lots for e in self.entries) / self.total_lots
    
    @property
    def total_entry_fee(self) -> float:
        return sum(e.fee for e in self.entries)
    
    @property
    def nanpin_count(self) -> int:
        return max(0, len(self.entries) - 1)
    
    @property
    def first_entry_time(self) -> pd.Timestamp:
        return self.entries[0].time if self.entries else None

    @property
    def first_entry_price(self) -> float:
        return self.entries[0].price if self.entries else 0.0


class BacktestResult:
    """
    バックテスト結果を保持・分析・可視化するクラス
    """
    def __init__(
        self,
        config: GridNanpinConfig,
        df_bars: pd.DataFrame,
        trade_logs: List[Dict[str, Any]],
        equity_curve: pd.DataFrame,
        signal_events: List[Dict[str, Any]]
    ):
        self.config = config
        self.df_bars = df_bars
        self.trades_df = pd.DataFrame(trade_logs) if trade_logs else pd.DataFrame()
        self.equity_df = equity_curve
        self.equity_curve = equity_curve # Highcharts描画連携用エイリアス
        self.signals_df = pd.DataFrame(signal_events) if signal_events else pd.DataFrame()
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> Dict[str, Any]:
        if self.trades_df.empty:
            return {
                "total_trades": 0,
                "win_trades": 0,
                "loss_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_win_pnl": 0.0,
                "avg_loss_pnl": 0.0,
                "payoff_ratio": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "max_nanpin_reached": 0,
                "avg_nanpin_count": 0.0,
                "total_fees": 0.0,
                "return_on_capital": 0.0
            }

        pnl = self.trades_df['net_pnl']
        wins = self.trades_df[pnl > 0]
        losses = self.trades_df[pnl <= 0]
        
        total_trades = len(self.trades_df)
        win_trades = len(wins)
        loss_trades = len(losses)
        win_rate = (win_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        
        total_pnl = pnl.sum()
        gross_profit = wins['net_pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.99 if gross_profit > 0 else 0.0)
        
        # エクイティカーブからドローダウン計算
        equity = self.equity_df['equity']
        peak = equity.cummax()
        drawdown = peak - equity
        max_dd = drawdown.max()
        dd_pct = (drawdown / peak).replace([np.inf, -np.inf], 0).fillna(0) * 100.0
        max_dd_pct = dd_pct.max()

        avg_trade_pnl = pnl.mean()
        avg_win_pnl = wins['net_pnl'].mean() if not wins.empty else 0.0
        avg_loss_pnl = abs(losses['net_pnl'].mean()) if not losses.empty else 0.0
        payoff_ratio = (avg_win_pnl / avg_loss_pnl) if avg_loss_pnl > 0 else 0.0

        nanpin_counts = self.trades_df['nanpin_count']
        max_nanpin_reached = nanpin_counts.max() if not nanpin_counts.empty else 0
        avg_nanpin_count = nanpin_counts.mean() if not nanpin_counts.empty else 0.0
        total_nanpin_orders = int(nanpin_counts.sum()) if not nanpin_counts.empty else 0
        total_lots_traded = int(self.trades_df['total_lots'].sum()) if not self.trades_df.empty else 0
        total_fees = self.trades_df['total_fee'].sum() if not self.trades_df.empty else 0.0
        
        roc = (total_pnl / self.config.initial_capital) * 100.0

        # 連勝・連敗計算
        is_win = (pnl > 0).astype(int).tolist()
        max_consec_win = 0
        max_consec_loss = 0
        cur_w = 0
        cur_l = 0
        for w in is_win:
            if w == 1:
                cur_w += 1
                cur_l = 0
                max_consec_win = max(max_consec_win, cur_w)
            else:
                cur_l += 1
                cur_w = 0
                max_consec_loss = max(max_consec_loss, cur_l)

        return {
            "total_trades": total_trades,
            "total_entries": total_trades,
            "total_exits": total_trades,
            "total_nanpin_orders": total_nanpin_orders,
            "total_lots_traded": total_lots_traded,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 1),
            "gross_profit": round(gross_profit, 1),
            "gross_loss": round(gross_loss, 1),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_dd, 1),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "avg_trade_pnl": round(avg_trade_pnl, 1),
            "avg_win_pnl": round(avg_win_pnl, 1),
            "avg_loss_pnl": round(avg_loss_pnl, 1),
            "payoff_ratio": round(payoff_ratio, 2),
            "max_consecutive_wins": max_consec_win,
            "max_consecutive_losses": max_consec_loss,
            "max_nanpin_reached": int(max_nanpin_reached),
            "avg_nanpin_count": round(avg_nanpin_count, 2),
            "total_fees": round(total_fees, 1),
            "return_on_capital": round(roc, 2)
        }

    def print_summary(self):
        """サマリーレポートをコンソール出力"""
        m = self.metrics
        c = self.config
        mode_label = "【IFモード】順張り (Highブレイク買い / Lowブレイク売り)" if c.entry_mode == "TREND" else "【通常】逆張り (Highブレイク売り / Lowブレイク買い)"
        print("\n" + "=" * 65)
        print(f"  {c.name} - バックテスト結果サマリー")
        print(f"  売買モード: {mode_label}")
        print("=" * 65)
        print(f"【基本設定】")
        print(f"  基準線期間(N): {c.n_period}本 | エントリー幅: ±{c.entry_delta}円 | グリッド幅: {c.grid_step}円")
        print(f"  利確幅: +{c.tp_delta}円 (値幅固定) | 損切ライン: -¥{c.sl_amount:,.0f}")
        print(f"  ナンピンロット設定: {sum(c.lot_table)}枚 ({len(c.lot_table)}段) - {c.lot_table}")
        print(f"  必要証拠金: ¥{c.margin_per_lot:,.0f}/枚 | 呼値: {c.tick_size}円 | 倍率: {c.multiplier}倍 | 手数料: ¥{c.fee_per_lot}/枚")
        print("-" * 65)
        print(f"【主要パフォーマンス指標 (KPI)】")
        print(f"  純損益 (Total P&L)      : ¥{m['total_pnl']:+,.0f} ({m['return_on_capital']:+.2f}%)")
        print(f"  総トレード数             : {m['total_trades']} 回 (勝: {m['win_trades']} / 負: {m['loss_trades']})")
        print(f"  勝率 (Win Rate)         : {m['win_rate']:.2f}%")
        print(f"  プロフィットファクター   : {m['profit_factor']:.2f}")
        print(f"  最大ドローダウン (MDD)  : -¥{m['max_drawdown']:,.0f} ({m['max_drawdown_pct']:.2f}%)")
        print(f"  総利益 / 総損失         : +¥{m['gross_profit']:,.0f} / -¥{m['gross_loss']:,.0f}")
        print(f"  平均トレード損益         : ¥{m['avg_trade_pnl']:+,.0f}")
        print(f"  平均勝ち / 平均負け     : ¥{m['avg_win_pnl']:+,.0f} / ¥{m['avg_loss_pnl']:+,.0f}")
        print(f"  ペイオフレシオ           : {m['payoff_ratio']:.2f}")
        print(f"  最大連勝 / 最大連敗     : {m['max_consecutive_wins']} 連勝 / {m['max_consecutive_losses']} 連敗")
        print(f"  最大ナンピン到達回数     : {m['max_nanpin_reached']} 回 (平均: {m['avg_nanpin_count']:.2f} 回)")
        print(f"  総支払手数料             : ¥{m['total_fees']:,.0f}")
        print("=" * 65)

    def plot_chart(self, save_path: Optional[str | Path] = None, show: bool = False, max_bars: int = 1500):
        """
        チャート上にエントリー、ナンピン、利確、損切をプロットし、
        資産曲線とナンピン回数分布をあわせて描画する。
        """
        plt.rcParams['font.sans-serif'] = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(16, 12), dpi=120)
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1], width_ratios=[3, 1], hspace=0.28, wspace=0.22)
        
        # 1. メイン価格チャート (左上)
        ax_price = fig.add_subplot(gs[0, :])
        
        df_sub = self.df_bars.copy()
        if len(df_sub) > max_bars:
            df_sub = df_sub.tail(max_bars).copy()
        
        start_time = df_sub['time'].iloc[0]
        end_time = df_sub['time'].iloc[-1]
        
        ax_price.plot(df_sub['time'], df_sub['close'], label='日経225マイクロ 終値', color='#90caf9', linewidth=1.2, alpha=0.9)
        if 'high_line' in df_sub.columns:
            ax_price.plot(df_sub['time'], df_sub['high_line'], label=f'Highライン (N={self.config.n_period})', color='#ff7043', linestyle='--', linewidth=1.0, alpha=0.7)
        if 'low_line' in df_sub.columns:
            ax_price.plot(df_sub['time'], df_sub['low_line'], label=f'Lowライン (N={self.config.n_period})', color='#66bb6a', linestyle='--', linewidth=1.0, alpha=0.7)
            
        # シグナルプロット
        if not self.signals_df.empty:
            sig_sub = self.signals_df[(self.signals_df['time'] >= start_time) & (self.signals_df['time'] <= end_time)]
            
            # ショート初期エントリー
            short_entries = sig_sub[sig_sub['event'] == 'SHORT_ENTRY']
            if not short_entries.empty:
                s_lbl = f'ショート開始 (Low-{self.config.entry_delta:.0f} 順張り)' if self.config.entry_mode == "TREND" else f'ショート開始 (High+{self.config.entry_delta:.0f} 逆張り)'
                ax_price.scatter(short_entries['time'], short_entries['price'], marker='v', color='#ff1744', s=90, label=s_lbl, zorder=5)
                
            # ロング初期エントリー
            long_entries = sig_sub[sig_sub['event'] == 'LONG_ENTRY']
            if not long_entries.empty:
                l_lbl = f'ロング開始 (High+{self.config.entry_delta:.0f} 順張り)' if self.config.entry_mode == "TREND" else f'ロング開始 (Low-{self.config.entry_delta:.0f} 逆張り)'
                ax_price.scatter(long_entries['time'], long_entries['price'], marker='^', color='#00e676', s=90, label=l_lbl, zorder=5)
                
            # ナンピン追加 (白丸)
            nanpins = sig_sub[sig_sub['event'] == 'NANPIN']
            if not nanpins.empty:
                ax_price.scatter(nanpins['time'], nanpins['price'], marker='o', color='#ffffff', edgecolors='#424242', linewidths=0.7, s=45, label='ナンピン追加 (白●)', zorder=4)
                
            # 利確エグジット (黄星)
            tp_exits = sig_sub[sig_sub['event'] == 'TP_EXIT']
            if not tp_exits.empty:
                ax_price.scatter(tp_exits['time'], tp_exits['price'], marker='*', color='#ffd600', edgecolors='#f57f17', linewidths=0.5, s=150, label='利確決済 (黄星★)', zorder=6)
                
            # 損切エグジット (紫星)
            sl_exits = sig_sub[sig_sub['event'] == 'SL_EXIT']
            if not sl_exits.empty:
                ax_price.scatter(sl_exits['time'], sl_exits['price'], marker='*', color='#d500f9', edgecolors='#4a148c', linewidths=0.5, s=150, label='損切決済 (紫星★)', zorder=6)

        ax_price.set_title(f"【日経225マイクロ先物】{self.config.name} (売買エントリー・ナンピン・決済チャート)", fontsize=13, fontweight='bold', pad=10)
        ax_price.set_ylabel("価格 (円)", fontsize=11)
        ax_price.grid(True, linestyle=':', alpha=0.6)
        ax_price.legend(loc='upper left', framealpha=0.85, fontsize=9, ncol=3)
        ax_price.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # 2. 累積損益曲線 (左中段)
        ax_equity = fig.add_subplot(gs[1, 0])
        eq_sub = self.equity_df[(self.equity_df['time'] >= start_time) & (self.equity_df['time'] <= end_time)]
        
        ax_equity.plot(eq_sub['time'], eq_sub['equity'], color='#00e676' if self.metrics['total_pnl'] >= 0 else '#ff1744', linewidth=1.5, label='資産推移 (Equity)')
        ax_equity.axhline(self.config.initial_capital, color='#78909c', linestyle=':', label='初期資金')
        ax_equity.fill_between(eq_sub['time'], self.config.initial_capital, eq_sub['equity'], where=(eq_sub['equity'] >= self.config.initial_capital), color='#00e676', alpha=0.15)
        ax_equity.fill_between(eq_sub['time'], self.config.initial_capital, eq_sub['equity'], where=(eq_sub['equity'] < self.config.initial_capital), color='#ff1744', alpha=0.15)
        
        ax_equity.set_title("資産推移曲線 (Equity Curve)", fontsize=11, fontweight='bold')
        ax_equity.set_ylabel("総資産 (円)", fontsize=10)
        ax_equity.grid(True, linestyle=':', alpha=0.6)
        ax_equity.legend(loc='upper left', fontsize=8)
        ax_equity.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # 3. ナンピン回数分布 (右中段)
        ax_nanpin = fig.add_subplot(gs[1, 1])
        if not self.trades_df.empty:
            nanpin_counts = self.trades_df['nanpin_count'].value_counts().sort_index()
            bars = ax_nanpin.bar(nanpin_counts.index, nanpin_counts.values, color='#42a5f5', edgecolor='#1565c0', width=0.6)
            ax_nanpin.set_title("ナンピン到達回数の分布 (回数別トレード頻度)", fontsize=11, fontweight='bold')
            ax_nanpin.set_xlabel("ナンピン回数 (0: 初期エントリーのみ)", fontsize=9)
            ax_nanpin.set_ylabel("トレード件数", fontsize=9)
            ax_nanpin.grid(True, linestyle=':', alpha=0.5, axis='y')
            for bar in bars:
                h = bar.get_height()
                ax_nanpin.text(bar.get_x() + bar.get_width() / 2., h + 0.5, f'{int(h)}', ha='center', va='bottom', fontsize=8)
            ax_nanpin.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # 4. パフォーマンスサマリー表 (下段全体)
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        m = self.metrics
        table_data = [
            ["純損益", f"¥{m['total_pnl']:+,.0f}", "勝率", f"{m['win_rate']:.1f}%", "総トレード数", f"{m['total_trades']} 回 (勝:{m['win_trades']} 負:{m['loss_trades']})"],
            ["プロフィットファクター", f"{m['profit_factor']:.2f}", "最大ドローダウン", f"-¥{m['max_drawdown']:,.0f} ({m['max_drawdown_pct']:.1f}%)", "平均損益/回", f"¥{m['avg_trade_pnl']:+,.0f}"],
            ["最大ナンピン到達", f"{m['max_nanpin_reached']} 回", "平均ナンピン回数", f"{m['avg_nanpin_count']:.2f} 回", "総手数料", f"¥{m['total_fees']:,.0f}"]
        ]
        
        table = ax_table.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            bbox=[0.05, 0.05, 0.9, 0.9]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for i in range(len(table_data)):
            for j in range(6):
                cell = table[i, j]
                if j % 2 == 0:
                    cell.set_facecolor('#eceff1')
                    cell.set_text_props(weight='bold', color='#263238')
                else:
                    cell.set_facecolor('#ffffff')
                    if "純損益" in table_data[i][j-1]:
                        cell.set_text_props(weight='bold', color='#2e7d32' if m['total_pnl'] >= 0 else '#c62828')
        
        for ax in [ax_price, ax_equity]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            fig.autofmt_xdate()

        plt.subplots_adjust(top=0.95, bottom=0.06, left=0.07, right=0.95, hspace=0.35, wspace=0.25)
        
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, bbox_inches='tight')
            print(f"📊 バックテストチャート画像を保存しました: {p.resolve()}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_nanpin_distribution(self, save_path: Optional[str | Path] = None, show: bool = False):
        """ナンピン回数分布のみを詳細にプロット"""
        if self.trades_df.empty:
            return
            
        plt.rcParams['font.sans-serif'] = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        nanpin_counts = self.trades_df['nanpin_count'].value_counts().sort_index()
        total_trades = len(self.trades_df)
        
        bars = ax.bar(nanpin_counts.index, nanpin_counts.values, color='#1976d2', edgecolor='#0d47a1', alpha=0.85, width=0.6)
        ax.set_title(f"【{self.config.name}】ナンピン到達回数の分布 (全{total_trades}トレード)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("ナンピン回数 (0: 初期エントリーのみで決済)", fontsize=10)
        ax.set_ylabel("トレード件数", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        for bar in bars:
            h = bar.get_height()
            pct = (h / total_trades) * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.3, f'{int(h)}件\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8.5)
            
        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, bbox_inches='tight')
            print(f"📊 ナンピン分布画像を保存しました: {p.resolve()}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)


class GridNanpinBacktester:
    """
    高値安値ライン突破・逆張りグリッドナンピン戦略のバックテストエンジン
    """
    def __init__(self, config: Optional[GridNanpinConfig] = None):
        self.config = config or GridNanpinConfig()

    def run(self, df_ohlcv: pd.DataFrame) -> BacktestResult:
        """
        OHLCV DataFrame に対してバックテストを実行
        """
        df = df_ohlcv.copy().sort_values('time').reset_index(drop=True)
        
        # 1. 基準線 (High/Lowライン) の計算
        # ルックアヘッドバイアスを排除するため、直前 (t-1) までのN本を基準とする
        n = self.config.n_period
        df['high_line'] = df['high'].shift(1).rolling(window=n).max()
        df['low_line'] = df['low'].shift(1).rolling(window=n).min()

        # 状態追跡用
        pos: Optional[Position] = None
        trade_logs: List[Dict[str, Any]] = []
        signal_events: List[Dict[str, Any]] = []
        
        current_equity = self.config.initial_capital
        equity_records = []

        # バー単位のシミュレーションループ
        for idx in range(len(df)):
            row = df.iloc[idx]
            t = row['time']
            o = float(row['open'])
            h = float(row['high'])
            l = float(row['low'])
            c = float(row['close'])
            h_line = row['high_line']
            l_line = row['low_line']

            # 基準線がまだ計算できない期間はスキップ
            if pd.isna(h_line) or pd.isna(l_line):
                equity_records.append({
                    'time': t,
                    'equity': current_equity,
                    'unrealized_pnl': 0.0,
                    'pos_lots': 0
                })
                continue

            # ----------------------------------------------------
            # A. ポジション保有中の場合: 損切 / 利確 / ナンピン判定
            # ----------------------------------------------------
            if pos is not None:
                is_closed = False
                
                # --- (1) 損切 (Stop Loss) 判定 ---
                # 含み損が -sl_amount に達した時点で全玉成行決済
                if pos.pos_type == 'SHORT':
                    loss_buffer = self.config.sl_amount - pos.total_entry_fee
                    sl_price = pos.avg_price + (loss_buffer / (pos.total_lots * self.config.multiplier))
                    
                    if h >= sl_price:
                        exit_price = max(o, sl_price) + self.config.slippage
                        exit_price = round(exit_price / self.config.tick_size) * self.config.tick_size
                        exit_fee = pos.total_lots * self.config.fee_per_lot
                        gross_pnl = (pos.avg_price - exit_price) * pos.total_lots * self.config.multiplier
                        net_pnl = gross_pnl - pos.total_entry_fee - exit_fee
                        
                        trade_logs.append({
                            'trade_no': len(trade_logs) + 1,
                            'pos_type': 'SHORT',
                            'entry_time': pos.first_entry_time,
                            'exit_time': t,
                            'entry_price': pos.first_entry_price,
                            'avg_entry_price': pos.avg_price,
                            'exit_price': exit_price,
                            'total_lots': pos.total_lots,
                            'nanpin_count': pos.nanpin_count,
                            'exit_reason': 'STOP_LOSS',
                            'gross_pnl': gross_pnl,
                            'total_fee': pos.total_entry_fee + exit_fee,
                            'net_pnl': net_pnl,
                            'holding_bars': idx - df[df['time'] == pos.first_entry_time].index[0]
                        })
                        
                        signal_events.append({
                            'time': t,
                            'price': exit_price,
                            'event': 'SL_EXIT',
                            'detail': f'SHORT損切 ({pos.total_lots}枚, 損益: ¥{net_pnl:,.0f})'
                        })
                        
                        current_equity += net_pnl
                        pos = None
                        is_closed = True

                elif pos.pos_type == 'LONG':
                    loss_buffer = self.config.sl_amount - pos.total_entry_fee
                    sl_price = pos.avg_price - (loss_buffer / (pos.total_lots * self.config.multiplier))
                    
                    if l <= sl_price:
                        exit_price = min(o, sl_price) - self.config.slippage
                        exit_price = round(exit_price / self.config.tick_size) * self.config.tick_size
                        exit_fee = pos.total_lots * self.config.fee_per_lot
                        gross_pnl = (exit_price - pos.avg_price) * pos.total_lots * self.config.multiplier
                        net_pnl = gross_pnl - pos.total_entry_fee - exit_fee
                        
                        trade_logs.append({
                            'trade_no': len(trade_logs) + 1,
                            'pos_type': 'LONG',
                            'entry_time': pos.first_entry_time,
                            'exit_time': t,
                            'entry_price': pos.first_entry_price,
                            'avg_entry_price': pos.avg_price,
                            'exit_price': exit_price,
                            'total_lots': pos.total_lots,
                            'nanpin_count': pos.nanpin_count,
                            'exit_reason': 'STOP_LOSS',
                            'gross_pnl': gross_pnl,
                            'total_fee': pos.total_entry_fee + exit_fee,
                            'net_pnl': net_pnl,
                            'holding_bars': idx - df[df['time'] == pos.first_entry_time].index[0]
                        })
                        
                        signal_events.append({
                            'time': t,
                            'price': exit_price,
                            'event': 'SL_EXIT',
                            'detail': f'LONG損切 ({pos.total_lots}枚, 損益: ¥{net_pnl:,.0f})'
                        })
                        
                        current_equity += net_pnl
                        pos = None
                        is_closed = True

                # --- (2) 利確 (Take Profit) 判定 (値幅利確のみに固定) ---
                if not is_closed and pos is not None:
                    tp_triggered = False
                    exit_price = 0.0
                    tp_reason = ""
                    
                    lot_table = self.config.lot_table
                    half_nanpin_threshold = max(1, len(lot_table) // 2)
                    is_emergency_mode = self.config.emergency_breakeven_exit and (pos.nanpin_count >= half_nanpin_threshold)
                    
                    if pos.pos_type == 'SHORT':
                        if is_emergency_mode:
                            # 【試験実装】ナンピン半分超え時のゼロ円清算（手数料分のみカバーする同値撤退）
                            fee_offset = np.ceil((pos.total_entry_fee + pos.total_lots * self.config.fee_per_lot) / (pos.total_lots * self.config.multiplier) / self.config.tick_size) * self.config.tick_size
                            be_target_price = pos.avg_price - fee_offset
                            if l <= be_target_price:
                                tp_triggered = True
                                tp_reason = f"EMERGENCY_BREAKEVEN (ナンピン#{pos.nanpin_count}段)"
                                exit_price = min(o, be_target_price) - self.config.slippage
                                exit_price = round(exit_price / self.config.tick_size) * self.config.tick_size
                        else:
                            target_pnl_price = pos.avg_price - self.config.tp_delta
                            if l <= target_pnl_price:
                                tp_triggered = True
                                tp_reason = f"TP_DELTA (+{self.config.tp_delta:.0f}円)"
                                exit_price = min(o, target_pnl_price) - self.config.slippage
                                exit_price = round(exit_price / self.config.tick_size) * self.config.tick_size
                            
                    elif pos.pos_type == 'LONG':
                        if is_emergency_mode:
                            # 【試験実装】ナンピン半分超え時のゼロ円清算（手数料分のみカバーする同値撤退）
                            fee_offset = np.ceil((pos.total_entry_fee + pos.total_lots * self.config.fee_per_lot) / (pos.total_lots * self.config.multiplier) / self.config.tick_size) * self.config.tick_size
                            be_target_price = pos.avg_price + fee_offset
                            if h >= be_target_price:
                                tp_triggered = True
                                tp_reason = f"EMERGENCY_BREAKEVEN (ナンピン#{pos.nanpin_count}段)"
                                exit_price = max(o, be_target_price) + self.config.slippage
                                exit_price = round(exit_price / self.config.tick_size) * self.config.tick_size
                        else:
                            target_pnl_price = pos.avg_price + self.config.tp_delta
                            if h >= target_pnl_price:
                                tp_triggered = True
                                tp_reason = f"TP_DELTA (+{self.config.tp_delta:.0f}円)"
                                exit_price = max(o, target_pnl_price) + self.config.slippage
                                exit_price = round(exit_price / self.config.tick_size) * self.config.tick_size

                    if tp_triggered:
                        exit_fee = pos.total_lots * self.config.fee_per_lot
                        if pos.pos_type == 'SHORT':
                            gross_pnl = (pos.avg_price - exit_price) * pos.total_lots * self.config.multiplier
                        else:
                            gross_pnl = (exit_price - pos.avg_price) * pos.total_lots * self.config.multiplier
                            
                        net_pnl = gross_pnl - pos.total_entry_fee - exit_fee
                        
                        trade_logs.append({
                            'trade_no': len(trade_logs) + 1,
                            'pos_type': pos.pos_type,
                            'entry_time': pos.first_entry_time,
                            'exit_time': t,
                            'entry_price': pos.first_entry_price,
                            'avg_entry_price': pos.avg_price,
                            'exit_price': exit_price,
                            'total_lots': pos.total_lots,
                            'nanpin_count': pos.nanpin_count,
                            'exit_reason': f'TAKE_PROFIT ({tp_reason})',
                            'gross_pnl': gross_pnl,
                            'total_fee': pos.total_entry_fee + exit_fee,
                            'net_pnl': net_pnl,
                            'holding_bars': idx - df[df['time'] == pos.first_entry_time].index[0]
                        })
                        
                        signal_events.append({
                            'time': t,
                            'price': exit_price,
                            'event': 'TP_EXIT',
                            'detail': f'{pos.pos_type}利確 ({pos.total_lots}枚, 損益: ¥{net_pnl:,.0f})'
                        })
                        
                        current_equity += net_pnl
                        pos = None
                        is_closed = True

                # --- (3) ナンピン (Grid 追加発注) 判定 ---
                if not is_closed and pos is not None:
                    lot_table = self.config.lot_table
                    next_level = len(pos.entries)
                    
                    while next_level < len(lot_table):
                        if pos.pos_type == 'SHORT':
                            target_nanpin_price = pos.first_entry_price + next_level * self.config.grid_step
                            if h >= target_nanpin_price:
                                nanpin_p = max(o, target_nanpin_price) + self.config.slippage
                                nanpin_p = round(nanpin_p / self.config.tick_size) * self.config.tick_size
                                add_lots = lot_table[next_level]
                                add_fee = add_lots * self.config.fee_per_lot
                                
                                pos.entries.append(PositionEntry(
                                    level=next_level,
                                    time=t,
                                    price=nanpin_p,
                                    lots=add_lots,
                                    fee=add_fee
                                ))
                                
                                signal_events.append({
                                    'time': t,
                                    'price': nanpin_p,
                                    'event': 'NANPIN',
                                    'detail': f'SHORTナンピン#{next_level} ({add_lots}枚 @ ¥{nanpin_p:,.0f}, 平均: ¥{pos.avg_price:,.0f})'
                                })
                                next_level += 1
                            else:
                                break
                        elif pos.pos_type == 'LONG':
                            target_nanpin_price = pos.first_entry_price - next_level * self.config.grid_step
                            if l <= target_nanpin_price:
                                nanpin_p = min(o, target_nanpin_price) - self.config.slippage
                                nanpin_p = round(nanpin_p / self.config.tick_size) * self.config.tick_size
                                add_lots = lot_table[next_level]
                                add_fee = add_lots * self.config.fee_per_lot
                                
                                pos.entries.append(PositionEntry(
                                    level=next_level,
                                    time=t,
                                    price=nanpin_p,
                                    lots=add_lots,
                                    fee=add_fee
                                ))
                                
                                signal_events.append({
                                    'time': t,
                                    'price': nanpin_p,
                                    'event': 'NANPIN',
                                    'detail': f'LONGナンピン#{next_level} ({add_lots}枚 @ ¥{nanpin_p:,.0f}, 平均: ¥{pos.avg_price:,.0f})'
                                })
                                next_level += 1
                            else:
                                break

            # ----------------------------------------------------
            # B. ポジション非保有の場合: 新規エントリー判定
            # ----------------------------------------------------
            if pos is None:
                is_trend = (self.config.entry_mode == "TREND")
                high_trigger_price = h_line + self.config.entry_delta
                low_trigger_price = l_line - self.config.entry_delta
                
                high_breached = (h >= high_trigger_price)
                low_breached = (l <= low_trigger_price)
                
                if high_breached and low_breached:
                    if abs(o - high_trigger_price) <= abs(o - low_trigger_price):
                        low_breached = False
                    else:
                        high_breached = False
                        
                if high_breached:
                    # Highライン突破: 逆張りならSHORT、順張りIFならLONG
                    entry_side = 'LONG' if is_trend else 'SHORT'
                    entry_p = max(o, high_trigger_price) + self.config.slippage
                    entry_p = round(entry_p / self.config.tick_size) * self.config.tick_size
                    init_lots = self.config.lot_table[0]
                    init_fee = init_lots * self.config.fee_per_lot
                    
                    pos = Position(
                        pos_type=entry_side,
                        entries=[PositionEntry(level=0, time=t, price=entry_p, lots=init_lots, fee=init_fee)],
                        initial_baseline=h_line
                    )
                    
                    mode_tag = " [IF順張り]" if is_trend else ""
                    signal_events.append({
                        'time': t,
                        'price': entry_p,
                        'event': f'{entry_side}_ENTRY',
                        'detail': f'新規{entry_side}エントリー{mode_tag} (High突破 {init_lots}枚 @ ¥{entry_p:,.0f})'
                    })

                    # ★【重要修正】エントリーした同一バー内でのナンピン執行
                    # 逆張り（not is_trend）の場合:
                    #   High突破（SHORT）に対してさらなる高値hの上昇、Low下抜け（LONG）に対してさらなる安値lの下落という
                    #   同一ベクトル方向のオーバーシュートであるため時系列的に整合し、即時ナンピンを執行する。
                    # 順張り（is_trend）の場合:
                    #   High突破（LONG）時に反対側の極値l、Low下抜け（SHORT）時に反対側の極値hを参照すると、
                    #   エントリー前の過去のヒゲで誤って即時ナンピンしてしまう時系列矛盾（先食い）が生じるため、
                    #   順張りモードでは同一バー内ナンピンを行わず次バー以降に正しく判定する。
                    if not is_trend and pos.pos_type == 'SHORT':
                        lot_table = self.config.lot_table
                        next_level = 1
                        while next_level < len(lot_table):
                            target_nanpin_price = pos.first_entry_price + next_level * self.config.grid_step
                            if h >= target_nanpin_price:
                                nanpin_p = round(target_nanpin_price / self.config.tick_size) * self.config.tick_size
                                add_lots = lot_table[next_level]
                                add_fee = add_lots * self.config.fee_per_lot
                                
                                pos.entries.append(PositionEntry(
                                    level=next_level,
                                    time=t,
                                    price=nanpin_p,
                                    lots=add_lots,
                                    fee=add_fee
                                ))
                                
                                signal_events.append({
                                    'time': t,
                                    'price': nanpin_p,
                                    'event': 'NANPIN',
                                    'detail': f'SHORTナンピン#{next_level} ({add_lots}枚 @ ¥{nanpin_p:,.0f}, 平均: ¥{pos.avg_price:,.0f})'
                                })
                                next_level += 1
                            else:
                                break
                    
                elif low_breached:
                    # Lowライン突破: 逆張りならLONG、順張りIFならSHORT
                    entry_side = 'SHORT' if is_trend else 'LONG'
                    entry_p = min(o, low_trigger_price) - self.config.slippage
                    entry_p = round(entry_p / self.config.tick_size) * self.config.tick_size
                    init_lots = self.config.lot_table[0]
                    init_fee = init_lots * self.config.fee_per_lot
                    
                    pos = Position(
                        pos_type=entry_side,
                        entries=[PositionEntry(level=0, time=t, price=entry_p, lots=init_lots, fee=init_fee)],
                        initial_baseline=l_line
                    )
                    
                    mode_tag = " [IF順張り]" if is_trend else ""
                    signal_events.append({
                        'time': t,
                        'price': entry_p,
                        'event': f'{entry_side}_ENTRY',
                        'detail': f'新規{entry_side}エントリー{mode_tag} (Low突破 {init_lots}枚 @ ¥{entry_p:,.0f})'
                    })

                    # ★【重要修正】エントリーした同一バー内でのナンピン執行
                    # 逆張り（not is_trend）かつ LONG の場合のみ執行（順張り時は次バー以降に正しく判定）
                    if not is_trend and pos.pos_type == 'LONG':
                        lot_table = self.config.lot_table
                        next_level = 1
                        while next_level < len(lot_table):
                            target_nanpin_price = pos.first_entry_price - next_level * self.config.grid_step
                            if l <= target_nanpin_price:
                                nanpin_p = round(target_nanpin_price / self.config.tick_size) * self.config.tick_size
                                add_lots = lot_table[next_level]
                                add_fee = add_lots * self.config.fee_per_lot
                                
                                pos.entries.append(PositionEntry(
                                    level=next_level,
                                    time=t,
                                    price=nanpin_p,
                                    lots=add_lots,
                                    fee=add_fee
                                ))
                                
                                signal_events.append({
                                    'time': t,
                                    'price': nanpin_p,
                                    'event': 'NANPIN',
                                    'detail': f'LONGナンピン#{next_level} ({add_lots}枚 @ ¥{nanpin_p:,.0f}, 平均: ¥{pos.avg_price:,.0f})'
                                })
                                next_level += 1
                            else:
                                break

                # ★【重要新機能】新規エントリーした同一バー内での利確（同足エグジット）判定
                # エントリー同一足の中でナンピンがなく（1枚のまま）、かつ利確目標価格に達していた場合、
                # 翌足への持ち越しによる約定価格の歪みを防ぎ、同一バー内で目標指値にて美しく決済を完結させる。
                if pos is not None and pos.nanpin_count == 0:
                    same_bar_tp = False
                    target_tp = 0.0
                    
                    if not is_trend:
                        if pos.pos_type == 'SHORT':
                            target_tp = pos.avg_price - self.config.tp_delta
                            # 終値が利確ライン以下で引けた場合（高値ブレイク後に確実に利確ラインを下抜け通過）
                            if c <= target_tp:
                                same_bar_tp = True
                        elif pos.pos_type == 'LONG':
                            target_tp = pos.avg_price + self.config.tp_delta
                            # 終値が利確ライン以上で引けた場合（安値ブレイク後に確実に利確ラインを上抜け通過）
                            if c >= target_tp:
                                same_bar_tp = True
                    else: # is_trend
                        if pos.pos_type == 'LONG':
                            target_tp = pos.avg_price + self.config.tp_delta
                            if h >= target_tp:
                                same_bar_tp = True
                        elif pos.pos_type == 'SHORT':
                            target_tp = pos.avg_price - self.config.tp_delta
                            if l <= target_tp:
                                same_bar_tp = True

                    if same_bar_tp:
                        exit_price = round(target_tp / self.config.tick_size) * self.config.tick_size
                        exit_fee = pos.total_lots * self.config.fee_per_lot
                        if pos.pos_type == 'SHORT':
                            gross_pnl = (pos.avg_price - exit_price) * pos.total_lots * self.config.multiplier
                        else:
                            gross_pnl = (exit_price - pos.avg_price) * pos.total_lots * self.config.multiplier
                        net_pnl = gross_pnl - pos.total_entry_fee - exit_fee
                        
                        trade_logs.append({
                            'trade_no': len(trade_logs) + 1,
                            'pos_type': pos.pos_type,
                            'entry_time': pos.first_entry_time,
                            'exit_time': t,
                            'entry_price': pos.first_entry_price,
                            'avg_entry_price': pos.avg_price,
                            'exit_price': exit_price,
                            'total_lots': pos.total_lots,
                            'nanpin_count': pos.nanpin_count,
                            'exit_reason': f'TAKE_PROFIT (TP_DELTA (+{self.config.tp_delta:.0f}円) [同足決済])',
                            'gross_pnl': gross_pnl,
                            'total_fee': pos.total_entry_fee + exit_fee,
                            'net_pnl': net_pnl,
                            'holding_bars': 0
                        })
                        
                        signal_events.append({
                            'time': t,
                            'price': exit_price,
                            'event': 'TP_EXIT',
                            'detail': f'{pos.pos_type}同足利確 ({pos.total_lots}枚, 損益: ¥{net_pnl:,.0f})'
                        })
                        
                        current_equity += net_pnl
                        pos = None

            # ----------------------------------------------------
            # C. バー終了時の含み損益・エクイティ・ポジション状態記録
            # ----------------------------------------------------
            unrealized_pnl = 0.0
            pos_lots = 0
            pos_type_str = 'NONE'
            avg_p = 0.0
            if pos is not None:
                pos_lots = pos.total_lots
                pos_type_str = pos.pos_type
                avg_p = pos.avg_price
                if pos.pos_type == 'SHORT':
                    unrealized_pnl = (pos.avg_price - c) * pos_lots * self.config.multiplier - pos.total_entry_fee
                else:
                    unrealized_pnl = (c - pos.avg_price) * pos_lots * self.config.multiplier - pos.total_entry_fee

            equity_records.append({
                'time': t,
                'equity': current_equity + unrealized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'pos_type': pos_type_str,
                'pos_lots': pos_lots,
                'avg_price': avg_p
            })

        # 未決済ポジションの処理:
        # 期間終了時の強制決済は行わず（直近の未完結ポジションを勝敗判定に含めない）、
        # かつチャート上にはエントリーやナンピンの履歴マーカーを正常に表示するため、
        # signal_events はそのまま保持する。

        equity_df = pd.DataFrame(equity_records)
        return BacktestResult(
            config=self.config,
            df_bars=df,
            trade_logs=trade_logs,
            equity_curve=equity_df,
            signal_events=signal_events
        )

