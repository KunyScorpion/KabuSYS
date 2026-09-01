import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from src.backend.strategies.base import BaseStrategy, BaseStrategyConfig, BacktestResult, TradeLog


@dataclass
class MASwing30minConfig(BaseStrategyConfig):
    """
    30分足移動平均スイング戦略（MASwing30minβ）の設定クラス
    """
    name: str = "30分足移動平均スイング戦略"
    timeframe: str = "30min"

    # 移動平均パラメータ
    ma1_period: int = 3          # 短期移動平均 (本数)
    ma2_period: int = 86         # 長期移動平均 (本数)
    
    # フィルタパラメータ
    min_price_move: float = 20.0 # クロス時の最低値動き幅 (円以上)
    
    # ロスカットパラメータ
    sl_points: float = 90.0      # ロスカット幅 (円)
    
    # 取引枚数
    lots: int = 1                # 1トレードあたりの取引枚数

    # コスト設定
    multiplier: float = 10.0     # 乗数 (日経225マイクロ: 10倍)
    tick_size: float = 5.0       # 呼値 (5円刻み)
    fee_per_lot: float = 15.0    # 片道手数料 (15円/枚)
    slippage: float = 0.0        # スリッページ (pt)
    initial_capital: float = 1000000.0 # 初期資本金 (1,000,000円)


class MASwing30minStrategy(BaseStrategy):
    """
    30分足移動平均スイング戦略のバックテストエンジン
    """
    def __init__(self, config: Optional[MASwing30minConfig] = None):
        if config is None:
            config = MASwing30minConfig()
        super().__init__(config)
        self.config: MASwing30minConfig = config

    def calculate_signals(self, df_bars: pd.DataFrame) -> pd.DataFrame:
        """
        四本値データに対してMA1, MA2およびシグナルを計算
        """
        df = df_bars.copy()
        
        # 必要な列の確保
        if "close" not in df.columns:
            # 大文字小文字対応
            for c in ["Close", "終値"]:
                if c in df.columns:
                    df["close"] = df[c]
                    break
        if "open" not in df.columns:
            for c in ["Open", "始値"]:
                if c in df.columns:
                    df["open"] = df[c]
                    break
        if "high" not in df.columns:
            for c in ["High", "高値"]:
                if c in df.columns:
                    df["high"] = df[c]
                    break
        if "low" not in df.columns:
            for c in ["Low", "安値"]:
                if c in df.columns:
                    df["low"] = df[c]
                    break
        if "time" not in df.columns and "datetime" in df.columns:
            df["time"] = pd.to_datetime(df["datetime"])
        elif "time" not in df.columns and "日時" in df.columns:
            df["time"] = pd.to_datetime(df["日時"])

        # 移動平均の計算
        df["ma1"] = df["close"].rolling(window=self.config.ma1_period).mean()
        df["ma2"] = df["close"].rolling(window=self.config.ma2_period).mean()

        return df

    def run_backtest(self, df_bars: pd.DataFrame) -> BacktestResult:
        """
        Excelシート（MASwing30minβ）と完全互換のロジックでバックテストを実行
        """
        df = self.calculate_signals(df_bars)
        n = len(df)
        if n == 0:
            return BacktestResult(self.config, df, [], pd.DataFrame())

        # 配列抽出（高速計算用）
        closes = df["close"].values
        opens = df["open"].values
        ma1 = df["ma1"].values
        ma2 = df["ma2"].values
        
        # 時刻処理
        if "time" in df.columns:
            times = pd.to_datetime(df["time"]).values
        else:
            times = pd.date_range("2020-01-01", periods=n, freq="30min").values

        # 状態追跡用配列 (Excelの計算シート Col O, P, Q, R, S, U, W に完全準拠)
        col_O = np.zeros(n, dtype=int)     # Col O: クロス (1, -1, または前足維持)
        col_P = np.zeros(n, dtype=int)     # Col P: 引け反映
        col_Q = np.zeros(n, dtype=float)   # Col Q: 仕掛値 (建値)
        col_R = np.zeros(n, dtype=int)     # Col R: ロスカット発動フラグ (1: ロスカット中, 0: 通常)
        col_S = np.zeros(n, dtype=int)     # Col S: ロスカット反映
        col_U = np.zeros(n, dtype=int)     # Col U: 週末・祝祭日手仕舞い判定
        col_W = np.zeros(n, dtype=int)     # Col W: 場中Sign (最終ポジション: 1=LONG, -1=SHORT, 0=NONE)

        min_move = self.config.min_price_move
        sl_pt = self.config.sl_points

        for i in range(1, n):
            c_curr = closes[i]
            c_prev = closes[i-1]
            m1 = ma1[i]
            m2 = ma2[i]
            t_dt = pd.to_datetime(times[i])
            t_hour = t_dt.hour
            t_min = t_dt.minute
            
            # --- 1. クロス判定 (Col O) ---
            # =IF(OR(N13="",M13=""),"",IF(AND(M13>N13,G13>=G12+O$2),1,IF(AND(M13<N13,G13<=G12-O$2),-1,O12)))
            if np.isnan(m1) or np.isnan(m2) or i < self.config.ma2_period:
                o_val = 0
            else:
                if m1 > m2 and c_curr >= (c_prev + min_move):
                    o_val = 1
                elif m1 < m2 and c_curr <= (c_prev - min_move):
                    o_val = -1
                else:
                    o_val = col_O[i-1]
            col_O[i] = o_val

            # --- 2. 引け時間帯判定 (Col P) ---
            # 日中引け（15:15〜15:30）やナイト引け（5:30〜6:00）では前足維持
            is_closing_time = False
            if (t_hour == 15 and t_min in [15, 30]) or (t_hour in [5, 6] and t_min in [0, 30]):
                is_closing_time = True

            if o_val == 0:
                p_val = 0
            elif is_closing_time:
                p_val = col_P[i-1]
            else:
                p_val = o_val
            col_P[i] = p_val

            # --- 3. 仕掛値の更新 (Col Q) ---
            # =IF(OR(P13=0,P13=""),"",IF(AND(P12<>0,P12<>P11),D13,Q12))
            if p_val == 0:
                q_val = 0.0
            else:
                if i >= 2 and col_P[i-1] != 0 and col_P[i-1] != col_P[i-2]:
                    q_val = opens[i]
                elif col_Q[i-1] != 0.0:
                    q_val = col_Q[i-1]
                else:
                    q_val = opens[i]
            col_Q[i] = q_val

            # --- 4. ロスカット判定 (Col R) ---
            # =IF(G13="","",IF(Q13="",0,IF(P13<>P12,0,IF(OR(AND(P12=1,(Q13-G13)>=R$10),AND(P12=-1,(G13-Q13)>=R$10)),1,R12))))
            if q_val == 0.0:
                r_val = 0
            elif col_P[i] != col_P[i-1]:
                # シグナル切り替わり時はロスカットリセット
                r_val = 0
            else:
                # 建値からの逆行幅判定、または前足Rを維持
                if col_P[i-1] == 1 and (q_val - c_curr) >= sl_pt:
                    r_val = 1
                elif col_P[i-1] == -1 and (c_curr - q_val) >= sl_pt:
                    r_val = 1
                else:
                    r_val = col_R[i-1]
            col_R[i] = r_val

            # --- 5. ロスカット反映 (Col S) ---
            # =IF(R13="","",IF(OR(引け時間帯),S12,IF(R13<>0,0,P13)))
            if is_closing_time:
                s_val = col_S[i-1]
            elif r_val != 0:
                s_val = 0
            else:
                s_val = p_val
            col_S[i] = s_val

            # --- 6. 祝祭日・週末手仕舞い判定 (Col U) ---
            is_weekend_close = (t_dt.weekday() == 5 and t_hour >= 5)
            col_U[i] = 1 if is_weekend_close else 0

            # --- 7. 最終ポジション Sign (Col W) ---
            # =IF(G13="","",IF(AND(U13=1,U14=0),0,IF(OR(引け時間帯),W12,IF(S13<>S12,S13,W12))))
            if is_weekend_close:
                w_val = 0
            elif is_closing_time:
                w_val = col_W[i-1]
            else:
                if s_val != col_S[i-1]:
                    w_val = s_val
                else:
                    w_val = col_W[i-1]
            col_W[i] = w_val

        # --- 8. トレード履歴およびチャート描画用シグナルイベントの生成 (次足始値約定モデル) ---
        trades: List[TradeLog] = []
        signal_events: List[Dict[str, Any]] = []
        trade_id = 1

        current_pos = 0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0

        for i in range(1, n - 1):
            w_curr = col_W[i]
            
            # ポジション保有中の決済判定 (Wが変化した -> 次足 opens[i+1] で決済執行)
            if current_pos != 0:
                if w_curr != current_pos:
                    exit_price = opens[i+1]
                    exit_time = pd.to_datetime(times[i+1])

                    if current_pos == 1:
                        pnl_pts = exit_price - entry_price
                    else:
                        pnl_pts = entry_price - exit_price

                    # スリッページ
                    pnl_pts -= (self.config.slippage * 2)

                    pnl_yen = pnl_pts * self.config.multiplier * self.config.lots
                    fee_total = (self.config.fee_per_lot * 2) * self.config.lots
                    net_pnl_yen = pnl_yen - fee_total

                    # 正確な決済理由とイベント種別の分類
                    if col_R[i] == 1 and col_R[i-1] == 0:
                        exit_reason = "SL"
                        event_type = "SL_EXIT"
                        detail_str = f"ロスカット損切 ({self.config.lots}枚 @ ¥{exit_price:,.0f}, 損益: ¥{net_pnl_yen:+,.0f})"
                    elif col_U[i] == 1 or col_U[i-1] == 1:
                        exit_reason = "WEEKEND_CLOSE"
                        event_type = "PERIOD_END_EXIT"
                        detail_str = f"週末引け決済 ({self.config.lots}枚 @ ¥{exit_price:,.0f}, 損益: ¥{net_pnl_yen:+,.0f})"
                    elif w_curr == -current_pos:
                        exit_reason = "REVERSE"
                        event_type = "TP_EXIT" if net_pnl_yen >= 0 else "SL_EXIT"
                        detail_str = f"ドテン{'利確' if net_pnl_yen >= 0 else '損切'} ({self.config.lots}枚 @ ¥{exit_price:,.0f}, 損益: ¥{net_pnl_yen:+,.0f})"
                    else:
                        exit_reason = "SIGNAL_EXIT"
                        event_type = "TP_EXIT" if net_pnl_yen >= 0 else "SL_EXIT"
                        detail_str = f"シグナル{'利確' if net_pnl_yen >= 0 else '手仕舞い'} ({self.config.lots}枚 @ ¥{exit_price:,.0f}, 損益: ¥{net_pnl_yen:+,.0f})"

                    trades.append(TradeLog(
                        trade_id=trade_id,
                        pos_type="LONG" if current_pos == 1 else "SHORT",
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=exit_time,
                        exit_price=exit_price,
                        lots=self.config.lots,
                        entry_fee=self.config.fee_per_lot * self.config.lots,
                        exit_fee=self.config.fee_per_lot * self.config.lots,
                        pnl_points=pnl_pts,
                        pnl_yen=pnl_yen,
                        net_pnl_yen=net_pnl_yen,
                        exit_reason=exit_reason,
                        holding_period_bars=(i + 1) - entry_idx,
                        extra_info={"entry_bar": entry_idx, "exit_bar": i + 1}
                    ))

                    # チャート決済マーカー
                    signal_events.append({
                        "time": exit_time,
                        "price": exit_price,
                        "event": event_type,
                        "detail": detail_str
                    })

                    trade_id += 1
                    current_pos = 0

            # 新規エントリー判定 (新たにWが立った -> 次足 opens[i+1] でエントリー執行)
            if current_pos == 0 and w_curr != 0:
                current_pos = w_curr
                entry_price = opens[i+1]
                entry_idx = i + 1
                entry_time = pd.to_datetime(times[i+1])

                signal_events.append({
                    "time": entry_time,
                    "price": entry_price,
                    "event": "LONG_ENTRY" if current_pos == 1 else "SHORT_ENTRY",
                    "detail": f"新規{'LONG (買い)' if current_pos == 1 else 'SHORT (売り)'}エントリー ({self.config.lots}枚 @ ¥{entry_price:,.0f})"
                })

        # --- 9. エクイティカーブおよびバー毎ポジション状態の構築 ---
        capital = self.config.initial_capital
        equity_records = []

        pos_tracker = 0
        cur_entry_p = 0.0

        pnl_by_exit_time = {t.exit_time: t.net_pnl_yen for t in trades}
        entry_time_map = {t.entry_time: t for t in trades}

        for i in range(n):
            t_curr = pd.to_datetime(times[i])
            c_curr = closes[i]

            # 確定損益の反映
            if t_curr in pnl_by_exit_time:
                capital += pnl_by_exit_time[t_curr]
                pos_tracker = 0
                cur_entry_p = 0.0

            # エントリー発生
            if t_curr in entry_time_map:
                t_obj = entry_time_map[t_curr]
                pos_tracker = 1 if t_obj.pos_type == "LONG" else -1
                cur_entry_p = t_obj.entry_price

            # 含み損益の計算
            if pos_tracker == 1:
                unrealized = (c_curr - cur_entry_p) * self.config.multiplier * self.config.lots
                pos_type_str = "LONG"
                pos_lots = self.config.lots
            elif pos_tracker == -1:
                unrealized = (cur_entry_p - c_curr) * self.config.multiplier * self.config.lots
                pos_type_str = "SHORT"
                pos_lots = self.config.lots
            else:
                unrealized = 0.0
                pos_type_str = "NONE"
                pos_lots = 0

            equity_records.append({
                "time": t_curr,
                "capital": capital,
                "equity": capital + unrealized,
                "close": c_curr,
                "pos_type": pos_type_str,
                "pos_lots": pos_lots,
                "avg_price": cur_entry_p,
                "unrealized_pnl": unrealized
            })

        equity_df = pd.DataFrame(equity_records)
        df["signal"] = col_W

        return BacktestResult(
            config=self.config,
            df_bars=df,
            trades=trades,
            equity_curve=equity_df,
            signals_df=df[["time", "close", "ma1", "ma2", "signal"]] if "time" in df.columns else df[["close", "ma1", "ma2", "signal"]],
            signal_events=signal_events
        )
