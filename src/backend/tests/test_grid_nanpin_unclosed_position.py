import pytest
import pandas as pd
import numpy as np
from src.backend.grid_nanpin_strategy import GridNanpinConfig, GridNanpinBacktester

def test_unclosed_position_signals_retained():
    times = pd.date_range('2026-01-01', periods=30, freq='1h')
    
    # 最初の15本は横ばい (基準線 N=5 を安定させる)
    prices = [1000.0] * 15
    # 16本目以降上昇し、High 1000 + 50 = 1050 で新規ショートエントリー、ナンピン発生
    prices.extend([1100.0, 1180.0, 1260.0, 1340.0, 1420.0,
                   1500.0, 1550.0, 1600.0, 1650.0, 1700.0,
                   1750.0, 1800.0, 1850.0, 1900.0, 1950.0])

    df = pd.DataFrame({
        'time': times,
        'open': prices,
        'high': [p + 20.0 for p in prices],
        'low': [p - 20.0 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    })

    config = GridNanpinConfig(
        entry_mode="CONTRARIAN",
        n_period=5,
        entry_delta=50.0,
        grid_step=80.0,
        lot_table=[1, 1, 2, 2, 3, 3],
        tp_delta=100.0,
        sl_amount=10000000.0,
        emergency_breakeven_exit=False
    )

    tester = GridNanpinBacktester(config)
    result = tester.run(df)

    assert len(result.trades_df) == 0, "未決済ポジションは trades_df に含まれないべき"
    assert not result.signals_df.empty, "未決済ポジションであっても signals_df は消去されず保持されるべき"
    
    events = result.signals_df['event'].tolist()
    assert 'SHORT_ENTRY' in events, "ショートエントリーシグナルが signals_df に存在するべき"
    assert 'NANPIN' in events, "ナンピンシグナルが signals_df に存在するべき"

    assert not result.equity_df.empty
    last_equity = result.equity_df.iloc[-1]
    assert last_equity['pos_lots'] > 0, "最終バーで未決済ポジションの保有枚数が記録されているべき"
    assert last_equity['pos_type'] == 'SHORT', "最終バーで SHORT ポジションであるべき"
