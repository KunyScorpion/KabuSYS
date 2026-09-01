import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Type
from abc import ABC, abstractmethod


@dataclass
class BaseStrategyConfig:
    """
    戦略設定の基底クラス
    """
    name: str = "Base Strategy"
    timeframe: str = "30min"         # デフォルト対象足種
    multiplier: float = 10.0         # 乗数 (日経225マイクロ: 10倍)
    tick_size: float = 5.0           # 呼値 (5円刻み)
    fee_per_lot: float = 15.0        # 片道手数料 (15円/枚)
    slippage: float = 0.0            # スリッページ (pt)
    initial_capital: float = 1000000.0 # 初期資本金 (1,000,000円)
    margin_per_lot: float = 25000.0  # 必要証拠金 (25,000円/枚)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """辞書からインスタンスを生成"""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class TradeLog:
    """
    1回の完結したトレード（または分割決済トレード）の記録
    """
    trade_id: int
    pos_type: str                    # 'LONG' or 'SHORT'
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    lots: int
    entry_fee: float
    exit_fee: float
    pnl_points: float                # 獲得値幅 (pt)
    pnl_yen: float                   # 実損益 (円, 手数料引前または後)
    net_pnl_yen: float               # 手数料差引後純損益 (円)
    exit_reason: str                 # 'TP', 'SL', 'MA_STOP', 'SESSION_CLOSE', 'SIGNAL_REVERSE' 等
    holding_period_bars: int = 0
    extra_info: Dict[str, Any] = field(default_factory=dict)


class BacktestResult:
    """
    戦略共通のバックテスト結果保持・KPI計算クラス
    """
    def __init__(
        self,
        config: BaseStrategyConfig,
        df_bars: pd.DataFrame,
        trades: List[TradeLog],
        equity_curve: pd.DataFrame,
        signals_df: Optional[pd.DataFrame] = None,
        signal_events: Optional[List[Dict[str, Any]]] = None
    ):
        self.config = config
        self.df_bars = df_bars
        self.trades = trades
        self.trades_df = self._create_trades_df(trades)
        self.equity_df = equity_curve
        self.equity_curve = equity_curve # Highcharts描画連携用エイリアス
        self.signals_df = signals_df if signals_df is not None else pd.DataFrame()
        self.signal_events = signal_events if signal_events is not None else []
        self.metrics = self._calculate_metrics()
        self.period_analysis = self._calculate_period_analysis()

    def _create_trades_df(self, trades: List[TradeLog]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            d = {
                "trade_id": t.trade_id,
                "pos_type": t.pos_type,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "lots": t.lots,
                "entry_fee": t.entry_fee,
                "exit_fee": t.exit_fee,
                "total_fee": t.entry_fee + t.exit_fee,
                "pnl_points": t.pnl_points,
                "pnl_yen": t.pnl_yen,
                "net_pnl_yen": t.net_pnl_yen,
                "exit_reason": t.exit_reason,
                "holding_bars": t.holding_period_bars
            }
            if t.extra_info:
                for k, v in t.extra_info.items():
                    d[f"extra_{k}"] = v
            rows.append(d)
        return pd.DataFrame(rows)

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
                "sharpe_ratio": 0.0,
                "total_fee": 0.0
            }

        df = self.trades_df
        total_trades = len(df)
        win_trades = int((df["net_pnl_yen"] > 0).sum())
        loss_trades = int((df["net_pnl_yen"] <= 0).sum())
        win_rate = (win_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = float(df[df["net_pnl_yen"] > 0]["net_pnl_yen"].sum())
        gross_loss = float(abs(df[df["net_pnl_yen"] <= 0]["net_pnl_yen"].sum()))
        total_pnl = float(df["net_pnl_yen"].sum())
        total_fee = float(df["total_fee"].sum())

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0

        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        avg_win_pnl = gross_profit / win_trades if win_trades > 0 else 0.0
        avg_loss_pnl = gross_loss / loss_trades if loss_trades > 0 else 0.0
        payoff_ratio = (avg_win_pnl / avg_loss_pnl) if avg_loss_pnl > 0 else 0.0

        # ドローダウン計算
        if not self.equity_df.empty and "capital" in self.equity_df.columns:
            cap = self.equity_df["capital"].values
            peak = np.maximum.accumulate(cap)
            dd = peak - cap
            max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0.0
            with np.errstate(divide='ignore', invalid='ignore'):
                dd_pct = np.where(peak > 0, dd / peak, 0.0)
            max_drawdown_pct = float(np.max(dd_pct) * 100.0) if len(dd_pct) > 0 else 0.0
        else:
            max_drawdown = 0.0
            max_drawdown_pct = 0.0

        # 連勝・連敗計算
        is_win = (df["net_pnl_yen"] > 0).values
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        cur_wins = 0
        cur_losses = 0
        for w in is_win:
            if w:
                cur_wins += 1
                cur_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, cur_losses)

        # シャープレシオ
        if total_trades > 1:
            pnl_std = df["net_pnl_yen"].std()
            sharpe_ratio = (avg_trade_pnl / pnl_std * np.sqrt(total_trades)) if pnl_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "avg_trade_pnl": avg_trade_pnl,
            "avg_win_pnl": avg_win_pnl,
            "avg_loss_pnl": avg_loss_pnl,
            "payoff_ratio": payoff_ratio,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "sharpe_ratio": sharpe_ratio,
            "total_fee": total_fee
        }

    def _calculate_period_analysis(self) -> Dict[str, pd.DataFrame]:
        """年別・月別損益の集計"""
        if self.trades_df.empty:
            return {"yearly": pd.DataFrame(), "monthly": pd.DataFrame()}

        df = self.trades_df.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["year"] = df["exit_time"].dt.year
        df["year_month"] = df["exit_time"].dt.to_period("M")

        # 年別
        yearly = df.groupby("year").agg(
            total_trades=("net_pnl_yen", "count"),
            win_trades=("net_pnl_yen", lambda x: (x > 0).sum()),
            gross_profit=("net_pnl_yen", lambda x: x[x > 0].sum()),
            gross_loss=("net_pnl_yen", lambda x: abs(x[x <= 0].sum())),
            total_pnl=("net_pnl_yen", "sum"),
        ).reset_index()
        yearly["loss_trades"] = yearly["total_trades"] - yearly["win_trades"]
        yearly["win_rate"] = (yearly["win_trades"] / yearly["total_trades"]) * 100.0
        yearly["profit_factor"] = yearly.apply(
            lambda r: (r["gross_profit"] / r["gross_loss"]) if r["gross_loss"] > 0 else (float('inf') if r["gross_profit"] > 0 else 0.0),
            axis=1
        )

        # 月別
        monthly = df.groupby("year_month").agg(
            total_trades=("net_pnl_yen", "count"),
            win_trades=("net_pnl_yen", lambda x: (x > 0).sum()),
            gross_profit=("net_pnl_yen", lambda x: x[x > 0].sum()),
            gross_loss=("net_pnl_yen", lambda x: abs(x[x <= 0].sum())),
            total_pnl=("net_pnl_yen", "sum"),
        ).reset_index()
        monthly["loss_trades"] = monthly["total_trades"] - monthly["win_trades"]
        monthly["win_rate"] = (monthly["win_trades"] / monthly["total_trades"]) * 100.0
        monthly["profit_factor"] = monthly.apply(
            lambda r: (r["gross_profit"] / r["gross_loss"]) if r["gross_loss"] > 0 else (float('inf') if r["gross_profit"] > 0 else 0.0),
            axis=1
        )
        monthly["period_str"] = monthly["year_month"].astype(str)

        return {"yearly": yearly, "monthly": monthly}


class BaseStrategy(ABC):
    """
    全戦略モデルが実装すべき抽象基底クラス
    """
    def __init__(self, config: BaseStrategyConfig):
        self.config = config

    @abstractmethod
    def run_backtest(self, df_bars: pd.DataFrame) -> BacktestResult:
        """
        与えられた四本値DataFrameに対してバックテストを実行し、BacktestResultを返す
        """
        pass

    @abstractmethod
    def calculate_signals(self, df_bars: pd.DataFrame) -> pd.DataFrame:
        """
        テクニカル指標およびエントリー/エグジットシグナル列を付与したDataFrameを返す
        """
        pass
