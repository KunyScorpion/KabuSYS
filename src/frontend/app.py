import streamlit as st
import sys
import re
import datetime
from typing import Optional, List, Any, Dict, Union, Tuple
import pandas as pd
from pathlib import Path

# srcディレクトリをパスに追加してimport可能にする
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import importlib
import src.backend.data_parser as data_parser_module
importlib.reload(data_parser_module)
from src.backend.data_parser import DataParser

import src.backend.indicators as indicators_module
importlib.reload(indicators_module)
from src.backend.indicators import IndicatorCalculator

import src.backend.settings_manager as settings_manager_module
importlib.reload(settings_manager_module)
from src.backend.settings_manager import SettingsManager

import src.backend.grid_nanpin_strategy as grid_nanpin_module
importlib.reload(grid_nanpin_module)
from src.backend.grid_nanpin_strategy import GridNanpinConfig, GridNanpinBacktester, BacktestResult

import src.backend.strategies.base as strategies_base_module
importlib.reload(strategies_base_module)
from src.backend.strategies.base import BaseStrategy, BaseStrategyConfig, BacktestResult as BaseBacktestResult, TradeLog

import src.backend.strategies as strategies_module
importlib.reload(strategies_module)
from src.backend.strategies import list_strategies, create_strategy_instance, delete_strategy_plugin, get_strategy_info

import src.frontend.chart as chart_module
importlib.reload(chart_module)
from src.frontend.chart import render_highstock_chart

def format_sheet_name_safe(sheet_name: str) -> str:
    """シート名から足種表示名を取得（フォールバック付き）"""
    mapping = {
        '1min': '1分足',
        '3min': '3分足',
        '5min': '5分足',
        '10min': '10分足',
        '15min': '15分足',
        '20min': '20分足',
        '30min': '30分足',
        '60min': '60分足',
        '日中日足': '日中日足',
        'ナイト場足': 'ナイト場足',
        '終日日足': '終日日足',
        '取引日日足': '取引日日足',
        '日足': '日足',
    }
    if hasattr(DataParser, 'format_timeframe_name'):
        return DataParser.format_timeframe_name(sheet_name)
    return mapping.get(sheet_name, sheet_name)

@st.cache_data(show_spinner=False)
def load_cached_data(file_path_str: str, sheet_name: str = None) -> Optional[pd.DataFrame]:
    """キャッシュを利用してファイルを読み込む"""
    return DataParser.load_file(Path(file_path_str), sheet_name)

def apply_custom_styles():
    """参考画像（画像1）に準拠したフルワイド・ダークテーマのCSSスタイルを適用"""
    st.markdown("""
    <style>
        /* 全体背景と文字色 */
        .stApp {
            background-color: #12141a;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* サイドバーを非表示（フルワイド化） */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* ページ上部・左右の余白を最適化 */
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            max-width: 100% !important;
        }

        /* Streamlitの透明ヘッダーによるクリック遮蔽を完全解消 */
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0px !important;
            pointer-events: none !important;
            z-index: -1 !important;
        }

        /* 銘柄バッジスタイル（参考画像準拠：赤枠・黒背景・太字） */
        .symbol-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1.5px solid #d32f2f;
            background-color: #0b0c0e;
            color: #ffffff;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.5px;
            padding: 5px 14px;
            border-radius: 4px;
            height: 38px;
            box-sizing: border-box;
            white-space: nowrap;
        }

        /* 上部バーの注記テキスト */
        .header-note {
            color: #8e9297;
            font-size: 11.5px;
            text-align: right;
            padding-top: 10px;
            white-space: nowrap;
        }

        /* Streamlit ウィジェットのダークモード最適化 */
        div[data-baseweb="select"] > div {
            background-color: #1a1d24 !important;
            border-color: #363a45 !important;
            color: #ffffff !important;
            border-radius: 4px !important;
            min-height: 38px !important;
        }
        div[data-baseweb="select"] * {
            color: #ffffff !important;
        }
        
        /* 日付入力ピッカー */
        div[data-baseweb="input"] > div {
            background-color: #1a1d24 !important;
            border-color: #363a45 !important;
            color: #ffffff !important;
            border-radius: 4px !important;
            min-height: 38px !important;
        }
        div[data-baseweb="input"] input {
            color: #ffffff !important;
        }

        /* 「表示する」ボタン（参考画像のワインレッド調） */
        div.stButton > button:first-child {
            background-color: #6b2626 !important;
            color: #ffffff !important;
            border: 1px solid #8e3535 !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            height: 38px !important;
            padding: 0 18px !important;
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #8c3232 !important;
            border-color: #b34242 !important;
            color: #ffffff !important;
        }
        div.stButton > button:first-child:active {
            background-color: #551d1d !important;
        }

        /* エクスパンダー・カードスタイル */
        .streamlit-expanderHeader {
            background-color: #1a1d24 !important;
            border: 1px solid #2b2f3a !important;
            border-radius: 4px !important;
            color: #d1d4dc !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] {
            background-color: #16181f !important;
            border: 1px solid #2b2f3a !important;
            border-radius: 6px !important;
            margin-bottom: 12px !important;
        }

        /* バックテスト・指標カード枠 */
        .control-card {
            background-color: #16181f;
            border: 1px solid #2b2f3a;
            border-radius: 6px;
            padding: 16px;
            margin-top: 8px;
            margin-bottom: 16px;
        }
        .control-card-title {
            font-size: 14px;
            font-weight: 700;
            color: #e0e3eb;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* KPI メトリクスカード */
        .kpi-card {
            background-color: #1a1d24;
            border: 1px solid #2e3340;
            border-radius: 6px;
            padding: 12px 14px;
            text-align: center;
        }
        .kpi-label {
            font-size: 11px;
            color: #8e9297;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 18px;
            font-weight: 700;
            color: #e0e3eb;
        }
    </style>
    """, unsafe_allow_html=True)

def run_strategy_backtest(df_data: pd.DataFrame) -> Any:
    """現在のシストレパラメータ設定でバックテストを実行し、session_stateを更新・永続化する"""
    selected_strategy_key = st.session_state.get("selected_strategy_key", "grid_nanpin")
    bt_saved = st.session_state.get("backtest_settings", {})
    
    if selected_strategy_key != "grid_nanpin":
        # 汎用プラグイン戦略の動的実行
        strat_instance = create_strategy_instance(selected_strategy_key)
        bt_res = strat_instance.run_backtest(df_data)
        
        # チャート用シグナルの安全な取得とフォールバック
        sig_events = getattr(bt_res, "signal_events", None)
        if not sig_events and hasattr(bt_res, "trades") and bt_res.trades:
            sig_events = []
            for t in bt_res.trades:
                sig_events.append({
                    "time": str(t.entry_time),
                    "price": float(t.entry_price),
                    "event": "LONG_ENTRY" if t.pos_type == "LONG" else "SHORT_ENTRY",
                    "detail": f"新規{'LONG' if t.pos_type == 'LONG' else 'SHORT'}エントリー ({t.lots}枚 @ ¥{t.entry_price:,.0f})"
                })
                exit_ev = "SL_EXIT" if t.exit_reason == "SL" else ("PERIOD_END_EXIT" if "CLOSE" in str(t.exit_reason) else "TP_EXIT")
                sig_events.append({
                    "time": str(t.exit_time),
                    "price": float(t.exit_price),
                    "event": exit_ev,
                    "detail": f"決済: {t.exit_reason} ({t.lots}枚 @ ¥{t.exit_price:,.0f}, 損益: ¥{t.net_pnl_yen:+,.0f})"
                })
                
        st.session_state["backtest_result"] = bt_res
        st.session_state["trade_signals"] = sig_events if sig_events is not None else []
        return bt_res

    else:
        # 高値安値ライン突破・逆張り/順張りIFグリッドナンピン戦略
        p_entry_mode = st.session_state.get("bt_entry_mode", bt_saved.get("entry_mode", "CONTRARIAN"))
        p_n_period = st.session_state.get("bt_n_period", bt_saved.get("n_period", 20))
        p_entry_delta = st.session_state.get("bt_entry_delta", bt_saved.get("entry_delta", 200.0))
        p_grid_step = st.session_state.get("bt_grid_step", bt_saved.get("grid_step", 250.0))
        p_tp_delta = st.session_state.get("bt_tp_delta", bt_saved.get("tp_delta", 200.0))
        p_sl_amount = st.session_state.get("bt_sl_amount", bt_saved.get("sl_amount", 500000.0))
        p_emergency_exit = st.session_state.get("bt_emergency_exit_toggle", bt_saved.get("emergency_breakeven_exit", False))
        current_lot_text = st.session_state.get("bt_lot_table_text", bt_saved.get("lot_table", "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10"))
        bt_capital = st.session_state.get("bt_capital", bt_saved.get("capital", 3000000.0))
        bt_margin = st.session_state.get("bt_margin_input", bt_saved.get("margin_per_lot", 25000.0))
        bt_fee = st.session_state.get("bt_fee", bt_saved.get("fee", 11.0))

        try:
            parsed_lots = [int(x.strip()) for x in str(current_lot_text).split(',') if x.strip()]
            if not parsed_lots:
                parsed_lots = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]
        except Exception:
            parsed_lots = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]

        # 設定の永続化
        st.session_state["backtest_settings"] = {
            "strategy_key": "grid_nanpin",
            "entry_mode": str(p_entry_mode),
            "n_period": int(p_n_period),
            "entry_delta": float(p_entry_delta),
            "grid_step": float(p_grid_step),
            "tp_delta": float(p_tp_delta),
            "sl_amount": float(p_sl_amount),
            "fee": float(bt_fee),
            "capital": float(bt_capital),
            "margin_per_lot": float(bt_margin),
            "lot_table": str(current_lot_text),
            "emergency_breakeven_exit": bool(p_emergency_exit)
        }
        
        SettingsManager.save_settings({
            "show_volume": st.session_state.get("show_volume", False),
            "main_indicators": st.session_state.get("main_indicators", ["SMA", "HI_LOW_Bands"]),
            "sub_indicators": st.session_state.get("sub_indicators", []),
            "indicator_params": st.session_state.get("indicator_params", {}),
            "backtest_settings": st.session_state["backtest_settings"],
            "lot_table_templates": st.session_state.get("lot_table_templates", {}),
            "strategy_presets": st.session_state.get("strategy_presets", {})
        })
        
        cfg = GridNanpinConfig(
            entry_mode=str(p_entry_mode),
            n_period=int(p_n_period),
            entry_delta=float(p_entry_delta),
            grid_step=float(p_grid_step),
            lot_table=parsed_lots,
            tp_delta=float(p_tp_delta),
            sl_amount=float(p_sl_amount),
            emergency_breakeven_exit=bool(p_emergency_exit),
            fee_per_lot=float(bt_fee),
            initial_capital=float(bt_capital),
            margin_per_lot=float(bt_margin)
        )
        
        bt_engine = GridNanpinBacktester(cfg)
        bt_res = bt_engine.run(df_data)
        
        st.session_state["backtest_result"] = bt_res
        st.session_state["trade_signals"] = bt_res.signals_df.to_dict('records') if not bt_res.signals_df.empty else []
        return bt_res

def main():
    st.set_page_config(
        page_title="KabuSYS (株シス)",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    apply_custom_styles()
    
    # 永続化された設定をロード
    if "settings_loaded" not in st.session_state:
        saved_settings = SettingsManager.load_settings()
        st.session_state["show_volume"] = saved_settings.get("show_volume", False)
        st.session_state["main_indicators"] = saved_settings.get("main_indicators", ["SMA", "HI_LOW_Bands"])
        st.session_state["sub_indicators"] = saved_settings.get("sub_indicators", [])
        st.session_state["indicator_params"] = saved_settings.get("indicator_params", {})
        st.session_state["backtest_settings"] = saved_settings.get("backtest_settings", {
            "n_period": 20,
            "entry_delta": 200.0,
            "grid_step": 250.0,
            "tp_delta": 200.0,
            "sl_amount": 500000.0,
            "fee": 15.0,
            "capital": 3000000.0,
            "margin_per_lot": 25000.0,
            "lot_table": "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10",
            "emergency_breakeven_exit": False
        })
        loaded_tpls = saved_settings.get("lot_table_templates", {})
        if not loaded_tpls:
            loaded_tpls = {
                "標準16段 (73枚)": "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10",
                "堅実16段 (46枚)": "1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7",
                "攻め8段 (36枚)": "1, 2, 3, 4, 5, 6, 7, 8"
            }
        st.session_state["lot_table_templates"] = loaded_tpls

        loaded_presets = saved_settings.get("strategy_presets", {})
        if not loaded_presets:
            loaded_presets = {
                "標準設定 (N=20, 利確200円, 73枚)": {
                    "n_period": 20,
                    "entry_delta": 200.0,
                    "grid_step": 250.0,
                    "tp_delta": 200.0,
                    "sl_amount": 500000.0,
                    "fee": 15.0,
                    "capital": 3000000.0,
                    "margin_per_lot": 25000.0,
                    "lot_table": "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10",
                    "emergency_breakeven_exit": False
                },
                "安全重視・同値撤退モード (N=20, 利確200円, 73枚)": {
                    "n_period": 20,
                    "entry_delta": 200.0,
                    "grid_step": 250.0,
                    "tp_delta": 200.0,
                    "sl_amount": 500000.0,
                    "fee": 15.0,
                    "capital": 3000000.0,
                    "margin_per_lot": 25000.0,
                    "lot_table": "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10",
                    "emergency_breakeven_exit": True
                }
            }
        st.session_state["strategy_presets"] = loaded_presets
        
        # High/Low期間の初期同期 (指標とバックテストで完全連携)
        hl_p = st.session_state["backtest_settings"].get("n_period", 
               st.session_state["indicator_params"].get("HI_LOW_Bands", {}).get("period", 20))
        st.session_state["shared_hl_period"] = int(hl_p)
        st.session_state["indicator_params"].setdefault("HI_LOW_Bands", {})["period"] = int(hl_p)
        st.session_state["settings_loaded"] = True

    # 1. データディレクトリの確認
    data_dir = root_dir / "data" / "market"
    if not data_dir.exists() or not any(data_dir.iterdir()):
        st.warning(f"データディレクトリ ({data_dir}) にデータファイルが見つかりません。")
        return
        
    all_files = list(data_dir.glob("*"))
    
    # プレフィックスでファイルをグループ化
    groups = {}
    for f in all_files:
        if f.suffix in ['.xlsx', '.xls', '.csv']:
            m = re.match(r'^(.*?)_20\d{2}\.(xlsx|xls|csv)$', f.name)
            if m:
                prefix = m.group(1)
                groups.setdefault(prefix, []).append(f)
            else:
                groups.setdefault(f.name, []).append(f)
                
    if not groups:
        st.error("有効なデータファイルがありません。")
        return
        
    group_names = sorted(list(groups.keys()))
    # デフォルトのデータグループ
    selected_group = group_names[0]
    
    # 表示用の銘柄名ラベル（画像1に合わせた「日経225micro」または「日経225mini」）
    display_symbol_name = "日経225マイクロ"
    if "N225micro" in selected_group or "micro" in selected_group.lower():
        display_symbol_name = "日経225micro"
    elif "N225mini" in selected_group or "mini" in selected_group.lower():
        display_symbol_name = "日経225mini"
    else:
        display_symbol_name = selected_group

    files_in_group = groups[selected_group]
    files_in_group.sort(key=lambda x: x.name)
    latest_file = files_in_group[-1]
    
    # シート一覧の取得と足種フォーマット（最新ファイルを優先しつつ、グループ内全ファイルのシートを網羅）
    raw_sheet_names = []
    for f in reversed(files_in_group):
        for s in DataParser.get_sheet_names(f):
            if s not in raw_sheet_names:
                raw_sheet_names.append(s)
                
    if not raw_sheet_names:
        raw_sheet_names = ["1min"]
        
    # シート名と表示名のマッピング
    sheet_display_map = {s: format_sheet_name_safe(s) for s in raw_sheet_names}
    
    # デフォルトは「60min」(60分足) があればそれを選択、なければ先頭
    default_sheet_idx = 0
    if "60min" in raw_sheet_names:
        default_sheet_idx = raw_sheet_names.index("60min")
    elif len(raw_sheet_names) > 0:
        default_sheet_idx = 0

    # ----------------------------------------------------
    # 2. 画面上部コントロールバー（チャート上部・横1列）
    # ----------------------------------------------------
    # カラム幅配分: [銘柄タグ, 足種, 期間開始〜終了, 出来高表示, テスト実行ボタン, 注記]
    top_col1, top_col2, top_col3, top_col4, top_col5, top_col6 = st.columns([1.4, 1.3, 2.9, 1.2, 1.4, 2.0])
    
    with top_col1:
        st.markdown(f'<div class="symbol-badge">{display_symbol_name}</div>', unsafe_allow_html=True)

    with top_col2:
        selected_sheet = st.selectbox(
            "足種",
            options=raw_sheet_names,
            index=default_sheet_idx,
            format_func=lambda s: sheet_display_map.get(s, s),
            label_visibility="collapsed",
            key="top_sheet_select"
        )

    with top_col3:
        # デフォルト期間設定（参考画像に合わせて 2026/08/22 〜 2026/08/30）
        default_end = datetime.date(2026, 8, 30)
        default_start = datetime.date(2026, 8, 22)
        
        date_range = st.date_input(
            "表示期間",
            value=(default_start, default_end),
            label_visibility="collapsed",
            key="top_date_range"
        )
        
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = sorted([date_range[0], date_range[1]])
        elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
            start_date = date_range[0]
            end_date = start_date + datetime.timedelta(days=8)
        else:
            start_date = default_start
            end_date = default_end

    with top_col4:
        # 出来高表示トグル (デフォルト非表示)
        cur_vol = st.session_state.get("show_volume", False)
        chk_volume = st.checkbox("出来高表示", value=cur_vol, key="top_show_volume_toggle")
        if chk_volume != cur_vol:
            st.session_state["show_volume"] = chk_volume
            SettingsManager.save_settings({
                "show_volume": chk_volume,
                "main_indicators": st.session_state["main_indicators"],
                "sub_indicators": st.session_state["sub_indicators"],
                "indicator_params": st.session_state["indicator_params"],
                "backtest_settings": st.session_state["backtest_settings"],
                "lot_table_templates": st.session_state.get("lot_table_templates", {}),
                "strategy_presets": st.session_state.get("strategy_presets", {})
            })

    with top_col5:
        btn_top_run_bt = st.button("🚀 テスト実行", use_container_width=True, key="top_run_backtest_btn", help="現在の期間・足種データに対して、設定中のシストレパラメータで即座にバックテストを実行します。")

    with top_col6:
        st.markdown('<div class="header-note">※10,000データまでの表示となります</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # 3. データの読み込みと結合
    # ----------------------------------------------------
    try:
        dfs = []
        with st.spinner('データを読み込み中...'):
            for f in files_in_group:
                m = re.search(r'20\d{2}', f.name)
                if m:
                    file_year = int(m.group())
                    if file_year < start_date.year - 1 or file_year > end_date.year + 1:
                        continue
                        
                df_part = load_cached_data(str(f), selected_sheet)
                if df_part is not None and not df_part.empty:
                    dfs.append(df_part)
                
        if not dfs:
            st.error("指定された期間のデータが存在しません。")
            return
            
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
        
        mask = (df_all['time'].dt.date >= start_date) & (df_all['time'].dt.date <= end_date)
        df_plot = df_all.loc[mask].copy()
        
        if df_plot.empty:
            st.warning("選択された期間内にデータがありません。期間を変更して「テスト実行」を押してください。")
            return

        MAX_CANDLES = 10000
        if len(df_plot) > MAX_CANDLES:
            df_plot = df_plot.tail(MAX_CANDLES).copy()

        # 上部の「🚀 テスト実行」ボタンが押された場合の即時バックテスト実行
        if btn_top_run_bt:
            with st.spinner("バックテストを実行中..."):
                bt_res = run_strategy_backtest(df_plot)
                st.success(f"✅ バックテスト完了！ 総トレード数: {bt_res.metrics['total_trades']}回（エントリー・決済完了） | 純損益: ¥{bt_res.metrics['total_pnl']:+,.0f}")
                st.rerun()

        # ----------------------------------------------------
        # 4. テクニカル指標の計算
        # ----------------------------------------------------
        all_metadata = IndicatorCalculator.METADATA
        main_options = {k: v["name"] for k, v in all_metadata.items() if v["type"] == "main"}
        sub_options = {k: v["name"] for k, v in all_metadata.items() if v["type"] == "sub"}

        selected_main_keys = [k for k in st.session_state["main_indicators"] if k in main_options]
        selected_sub_keys = [k for k in st.session_state["sub_indicators"] if k in sub_options]

        main_indicator_data = {}
        main_indicator_meta = {}
        for ind_id in selected_main_keys:
            params = st.session_state["indicator_params"].get(ind_id, {})
            df_res = IndicatorCalculator.compute(ind_id, df_plot, params)
            main_indicator_data[ind_id] = df_res
            main_indicator_meta[ind_id] = all_metadata[ind_id]

        sub_indicator_data = {}
        sub_indicator_meta = {}
        for ind_id in selected_sub_keys:
            params = st.session_state["indicator_params"].get(ind_id, {})
            df_res = IndicatorCalculator.compute(ind_id, df_plot, params)
            sub_indicator_data[ind_id] = df_res
            sub_indicator_meta[ind_id] = all_metadata[ind_id]

        # ----------------------------------------------------
        # 5. メインチャート表示（Highcharts Stock）
        # ----------------------------------------------------
        bt_res_saved = st.session_state.get("backtest_result", None)
        bar_metrics_list = bt_res_saved.equity_curve.to_dict('records') if bt_res_saved is not None and hasattr(bt_res_saved, 'equity_curve') else []

        render_highstock_chart(
            df_ohlcv=df_plot,
            main_indicator_data=main_indicator_data,
            main_indicator_meta=main_indicator_meta,
            sub_indicator_data=sub_indicator_data,
            sub_indicator_meta=sub_indicator_meta,
            trade_signals=st.session_state.get("trade_signals", None),
            bar_metrics=bar_metrics_list,
            show_volume=st.session_state.get("show_volume", False),
            height=720,
            symbol_name=display_symbol_name
        )

        # ----------------------------------------------------
        # 6. チャート下部コントロールエリア
        # ----------------------------------------------------
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        # タブ形式で ①テクニカル指標設定 ②システムトレード（バックテスト） を美しく切り替え可能に配置
        tab_indicators, tab_backtest, tab_preview = st.tabs([
            "📊 テクニカル指標設定",
            "⚡ システムトレード・バックテスト",
            "📋 データプレビュー"
        ])

        with tab_indicators:
            st.markdown('<div class="control-card">', unsafe_allow_html=True)
            st.markdown('<div class="control-card-title">📈 チャート重畳・オシレーター指標の選択と設定</div>', unsafe_allow_html=True)
            
            col_ind1, col_ind2 = st.columns(2)
            with col_ind1:
                new_main_keys = st.multiselect(
                    "メインチャート指標 (価格チャートに重ねて表示)",
                    options=list(main_options.keys()),
                    default=selected_main_keys,
                    format_func=lambda k: main_options[k],
                    help="価格チャート上にオーバーレイ表示する指標（移動平均線、ボリンジャーバンド、一目均衡表など）を選択します。",
                    key="main_ind_multiselect"
                )
            with col_ind2:
                new_sub_keys = st.multiselect(
                    "サブチャート指標 (オシレーター等・下段に追加表示)",
                    options=list(sub_options.keys()),
                    default=selected_sub_keys,
                    format_func=lambda k: sub_options[k],
                    help="チャート下部に独立ペインとして表示する指標（MACD, RSI, ストキャスティクスなど）を選択します。",
                    key="sub_ind_multiselect"
                )

            # パラメータ調整エクスパンダー
            active_indicators = new_main_keys + new_sub_keys
            params_modified = False
            
            if active_indicators:
                with st.expander("⚙️ 選択指標の期間・計算パラメータ調整", expanded=False):
                    st.caption("※重要な計算式は固定され、期間等のパラメータのみ調整可能です。変更内容は自動保存されます。")
                    param_cols = st.columns(min(3, len(active_indicators)))
                    
                    for idx, ind_id in enumerate(active_indicators):
                        with param_cols[idx % len(param_cols)]:
                            meta = all_metadata.get(ind_id, {})
                            param_defs = meta.get("params", [])
                            if not param_defs:
                                continue
                                
                            st.markdown(f"**{meta.get('name', ind_id)}**")
                            cur_params = st.session_state["indicator_params"].setdefault(ind_id, {})
                            
                            for p_def in param_defs:
                                pid = p_def["id"]
                                pname = p_def["name"]
                                ptype = p_def["type"]
                                pdefault = p_def["default"]
                                pmin = p_def.get("min", 1)
                                pmax = p_def.get("max", 500)
                                pstep = p_def.get("step", 1 if ptype == "int" else 0.1)
                                
                                # High/Low バンドの場合は共有期間と同期
                                if ind_id == "HI_LOW_Bands" and pid == "period":
                                    val = st.session_state.get("shared_hl_period", cur_params.get(pid, pdefault))
                                else:
                                    val = cur_params.get(pid, pdefault)

                                if ptype == "int":
                                    new_val = st.number_input(
                                        f"{pname}",
                                        min_value=int(pmin),
                                        max_value=int(pmax),
                                        value=int(val),
                                        step=int(pstep),
                                        key=f"p_{ind_id}_{pid}"
                                    )
                                else:
                                    new_val = st.number_input(
                                        f"{pname}",
                                        min_value=float(pmin),
                                        max_value=float(pmax),
                                        value=float(val),
                                        step=float(pstep),
                                        key=f"p_{ind_id}_{pid}"
                                    )
                                    
                                if new_val != cur_params.get(pid):
                                    cur_params[pid] = new_val
                                    if ind_id == "HI_LOW_Bands" and pid == "period":
                                        st.session_state["shared_hl_period"] = int(new_val)
                                    params_modified = True

            # 変更時の永続化
            if (set(new_main_keys) != set(st.session_state["main_indicators"]) or 
                set(new_sub_keys) != set(st.session_state["sub_indicators"]) or 
                params_modified):
                
                st.session_state["main_indicators"] = new_main_keys
                st.session_state["sub_indicators"] = new_sub_keys
                
                SettingsManager.save_settings({
                    "main_indicators": new_main_keys,
                    "sub_indicators": new_sub_keys,
                    "indicator_params": st.session_state["indicator_params"]
                })

            st.markdown('</div>', unsafe_allow_html=True)

        with tab_backtest:
            st.markdown('<div class="control-card">', unsafe_allow_html=True)
            st.markdown('<div class="control-card-title">🤖 システムトレード（売買シミュレーション / バックテスト）コントロール</div>', unsafe_allow_html=True)
            
            # 保存されている設定の取得
            bt_saved = st.session_state.get("backtest_settings", {})
            lot_templates = st.session_state.get("lot_table_templates", {})
            strategy_presets = st.session_state.get("strategy_presets", {})

            # ----------------------------------------------------
            # ★ A. お気に入りパラメータ設定（戦略プリセット）管理 UI
            # ----------------------------------------------------
            st.markdown("**⭐ お気に入り設定（パラメータ一括保存・呼出）**")
            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([3, 1.2, 3, 1.2, 1.2])
            
            preset_names = list(strategy_presets.keys())
            with p_col1:
                selected_preset_name = st.selectbox(
                    "保存済みお気に入り設定",
                    options=preset_names if preset_names else ["(保存なし)"],
                    label_visibility="collapsed",
                    key="sel_preset_box"
                )
            with p_col2:
                btn_load_preset = st.button("呼出", use_container_width=True, key="btn_load_preset")
                if btn_load_preset and preset_names and selected_preset_name in strategy_presets:
                    preset_data = strategy_presets[selected_preset_name]
                    st.session_state["backtest_settings"].update(preset_data)
                    
                    # ウィジェットキーへ直接値を代入して画面上の全入力欄を即座に更新
                    if "n_period" in preset_data:
                        st.session_state["shared_hl_period"] = int(preset_data["n_period"])
                        st.session_state["indicator_params"].setdefault("HI_LOW_Bands", {})["period"] = int(preset_data["n_period"])
                        st.session_state["bt_n_period"] = int(preset_data["n_period"])
                    if "entry_delta" in preset_data:
                        st.session_state["bt_entry_delta"] = float(preset_data["entry_delta"])
                    if "grid_step" in preset_data:
                        st.session_state["bt_grid_step"] = float(preset_data["grid_step"])
                    if "tp_delta" in preset_data:
                        st.session_state["bt_tp_delta"] = float(preset_data["tp_delta"])
                    if "sl_amount" in preset_data:
                        st.session_state["bt_sl_amount"] = float(preset_data["sl_amount"])
                    if "fee" in preset_data:
                        st.session_state["bt_fee"] = int(preset_data["fee"])
                    if "capital" in preset_data:
                        st.session_state["bt_capital"] = int(preset_data["capital"])
                    if "margin_per_lot" in preset_data:
                        st.session_state["bt_margin_input"] = int(preset_data["margin_per_lot"])
                    if "lot_table" in preset_data:
                        st.session_state["bt_lot_table_text"] = preset_data["lot_table"]
                    if "emergency_breakeven_exit" in preset_data:
                        st.session_state["bt_emergency_exit_toggle"] = bool(preset_data["emergency_breakeven_exit"])

                    SettingsManager.save_settings({
                        "show_volume": st.session_state.get("show_volume", False),
                        "main_indicators": st.session_state["main_indicators"],
                        "sub_indicators": st.session_state["sub_indicators"],
                        "indicator_params": st.session_state["indicator_params"],
                        "backtest_settings": st.session_state["backtest_settings"],
                        "lot_table_templates": st.session_state.get("lot_table_templates", {}),
                        "strategy_presets": st.session_state.get("strategy_presets", {})
                    })
                    st.success(f"お気に入り設定「{selected_preset_name}」を呼び出し、全入力欄に反映しました！")
                    st.rerun()

            with p_col3:
                new_preset_name = st.text_input("お気に入り保存名", placeholder="例: 標準設定 (N=20, 73枚)", label_visibility="collapsed", key="txt_new_preset")
            with p_col4:
                btn_save_preset = st.button("保存", use_container_width=True, key="btn_save_preset")
                if btn_save_preset:
                    save_name = new_preset_name.strip()
                    if save_name:
                        new_preset_dict = {
                            "n_period": int(st.session_state.get("shared_hl_period", bt_saved.get("n_period", 20))),
                            "entry_delta": float(bt_saved.get("entry_delta", 200.0)),
                            "grid_step": float(bt_saved.get("grid_step", 250.0)),
                            "tp_delta": float(bt_saved.get("tp_delta", 200.0)),
                            "sl_amount": float(bt_saved.get("sl_amount", 500000.0)),
                            "fee": float(bt_saved.get("fee", 15.0)),
                            "capital": float(bt_saved.get("capital", 3000000.0)),
                            "margin_per_lot": float(bt_saved.get("margin_per_lot", 25000.0)),
                            "lot_table": bt_saved.get("lot_table", "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10"),
                            "emergency_breakeven_exit": bool(bt_saved.get("emergency_breakeven_exit", False))
                        }
                        updated_presets = SettingsManager.save_strategy_preset(save_name, new_preset_dict)
                        st.session_state["strategy_presets"] = updated_presets
                        st.success(f"お気に入り設定「{save_name}」を確実に保存しました！")
                        st.rerun()

            with p_col5:
                btn_del_preset = st.button("削除", use_container_width=True, key="btn_del_preset")
                if btn_del_preset and selected_preset_name in strategy_presets:
                    updated_presets = SettingsManager.delete_strategy_preset(selected_preset_name)
                    st.session_state["strategy_presets"] = updated_presets
                    st.info(f"お気に入り設定「{selected_preset_name}」を削除しました。")
                    st.rerun()

            st.markdown("---")

            # ----------------------------------------------------
            # ★ B. 基本取引パラメータ & 証拠金・資本金設定 & プラグイン管理
            # ----------------------------------------------------
            available_strats = list_strategies()
            strat_map = {s["key"]: s for s in available_strats}
            strat_name_to_key = {s["name"]: s["key"] for s in available_strats}
            strat_key_to_name = {s["key"]: s["name"] for s in available_strats}
            strategy_names = list(strat_name_to_key.keys())

            cur_strat_key = st.session_state.get("selected_strategy_key", "grid_nanpin")
            if cur_strat_key not in strat_map:
                cur_strat_key = "grid_nanpin"
                st.session_state["selected_strategy_key"] = cur_strat_key

            cur_strat_name = strat_key_to_name.get(cur_strat_key, strategy_names[0])
            cur_strat_idx = strategy_names.index(cur_strat_name) if cur_strat_name in strategy_names else 0

            bt_col1, bt_col2, bt_col3, bt_col4 = st.columns([3.5, 2.5, 2.5, 1.5])
            with bt_col1:
                selected_strategy_name = st.selectbox(
                    "売買ルール・戦略モデル", 
                    strategy_names, 
                    index=cur_strat_idx, 
                    key="bt_strategy_select_box"
                )
                selected_strategy_key = strat_name_to_key[selected_strategy_name]
                st.session_state["selected_strategy_key"] = selected_strategy_key
                
            with bt_col2:
                default_cap = 3000000
                bt_capital = st.number_input(
                    "初期資本金 (円)", 
                    min_value=100000, 
                    max_value=100000000, 
                    value=int(bt_saved.get("capital", default_cap)), 
                    step=500000, 
                    key="bt_capital"
                )
            with bt_col3:
                bt_margin = st.number_input(
                    "必要証拠金 (円/枚)", 
                    min_value=5000, 
                    max_value=500000, 
                    value=int(bt_saved.get("margin_per_lot", 25000)), 
                    step=5000, 
                    key="bt_margin_input",
                    help="日経225マイクロ先物の1枚あたり必要証拠金（SPAN証拠金）を設定します。デフォルト: 25,000円"
                )
            with bt_col4:
                bt_fee = st.number_input(
                    "片道手数料 (円/枚)", 
                    min_value=0, 
                    max_value=500, 
                    value=int(bt_saved.get("fee", 11)), 
                    step=1, 
                    key="bt_fee",
                    help="日経225マイクロ先物の片道手数料（主要ネット証券: 税込約11円/枚）。デフォルト: 11円"
                )

            # プラグイン削除（アンインストール）機能
            current_strat_info = strat_map.get(selected_strategy_key, {})
            if current_strat_info.get("is_deletable", False):
                del_c1, del_c2 = st.columns([7, 3])
                with del_c1:
                    st.caption(f"🧩 この戦略は独立したプラグインファイル（`{Path(current_strat_info.get('file_path', '')).name}`）としてロードされています。")
                with del_c2:
                    btn_uninstall_strat = st.button("🗑️ この戦略モデルを削除（ファイル完全削除）", use_container_width=True, key=f"btn_del_strat_{selected_strategy_key}")
                    if btn_uninstall_strat:
                        try:
                            delete_strategy_plugin(selected_strategy_key)
                            st.session_state["selected_strategy_key"] = "grid_nanpin"
                            if "backtest_result" in st.session_state:
                                del st.session_state["backtest_result"]
                            if "trade_signals" in st.session_state:
                                del st.session_state["trade_signals"]
                            st.success(f"戦略モデル「{selected_strategy_name}」のロジックファイルを安全にアンインストール・削除しました。")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"削除中にエラーが発生しました: {ex}")

            # ----------------------------------------------------
            # ★ C. 戦略詳細パラメータ設定エクスパンダー
            # ----------------------------------------------------
            with st.expander(f"⚙️ {selected_strategy_name} パラメータ設定 (クリックして開閉)", expanded=True):
                if selected_strategy_key == "grid_nanpin":
                    # --- 高値安値逆張り/順張りナンピン戦略のパラメータ ---
                    # 売買方向モード選択 (通常逆張り vs 順張りIF)
                    saved_mode = bt_saved.get("entry_mode", "CONTRARIAN")
                    mode_options = ["CONTRARIAN", "TREND"]
                    mode_labels = {
                        "CONTRARIAN": "🔄 通常モード: 逆張り (High突破でショート / Low突破でロング)",
                        "TREND": "🚀 IFモード: 順張り (High突破でロング / Low突破でショート)"
                    }
                    selected_mode = st.radio(
                        "**🎯 エントリー売買方向（IF検証モード切替）**",
                        options=mode_options,
                        index=0 if saved_mode != "TREND" else 1,
                        format_func=lambda x: mode_labels[x],
                        horizontal=True,
                        key="bt_entry_mode",
                        help="『もし同じHigh/Low突破判定ポイントで順張り（ブレイクアウト買い・売り）で入った場合』のシストレ挙動を同一のナンピン・利確ルールで比較検証できます。"
                    )
                    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

                    p_c1, p_c2, p_c3 = st.columns(3)
                    with p_c1:
                        current_shared_hl = int(st.session_state.get("shared_hl_period", bt_saved.get("n_period", 20)))
                        p_n_period = st.number_input(
                            "基準線期間 (N本)",
                            min_value=5,
                            max_value=500,
                            value=current_shared_hl,
                            step=1,
                            key="bt_n_period",
                            help="直近N本（最大500本）のHigh/Lowラインを基準線として算出します。チャート上のハイローバンド指標と自動連動します。"
                        )
                        if p_n_period != current_shared_hl:
                            st.session_state["shared_hl_period"] = int(p_n_period)
                            st.session_state["indicator_params"].setdefault("HI_LOW_Bands", {})["period"] = int(p_n_period)
                            st.session_state["backtest_settings"]["n_period"] = int(p_n_period)
                            SettingsManager.save_settings({
                                "show_volume": st.session_state.get("show_volume", False),
                                "main_indicators": st.session_state["main_indicators"],
                                "sub_indicators": st.session_state["sub_indicators"],
                                "indicator_params": st.session_state["indicator_params"],
                                "backtest_settings": st.session_state["backtest_settings"],
                                "lot_table_templates": st.session_state.get("lot_table_templates", {}),
                                "strategy_presets": st.session_state.get("strategy_presets", {})
                            })
                        p_entry_delta = st.number_input(
                            "エントリー突破幅 (円)", 
                            min_value=10.0, 
                            max_value=2000.0, 
                            value=float(bt_saved.get("entry_delta", 200.0)), 
                            step=50.0, 
                            key="bt_entry_delta",
                            help="Highライン上抜け・Lowライン下抜けでエントリーを開始するブレイク値幅です（逆張りならHighでショート/Lowでロング、順張りIFならHighでロング/Lowでショート）。"
                        )
                    with p_c2:
                        p_grid_step = st.number_input(
                            "ナンピン逆行幅 (円)", 
                            min_value=50.0, 
                            max_value=2000.0, 
                            value=float(bt_saved.get("grid_step", 250.0)), 
                            step=50.0, 
                            key="bt_grid_step",
                            help="初回エントリー価格からの逆行幅ごとにナンピン発注を行います。"
                        )
                        p_tp_delta = st.number_input(
                            "利確幅 (円) [値幅固定]", 
                            min_value=10.0, 
                            max_value=2000.0, 
                            value=float(bt_saved.get("tp_delta", 200.0)), 
                            step=100.0, 
                            key="bt_tp_delta", 
                            help="平均取得価格から指定幅（最大2000円、増減幅100円）有利に進んだ時点で全玉利確します。"
                        )
                    with p_c3:
                        p_sl_amount = st.number_input(
                            "損切含み損金額 (円)", 
                            min_value=50000.0, 
                            max_value=10000000.0, 
                            value=float(bt_saved.get("sl_amount", 500000.0)), 
                            step=50000.0, 
                            key="bt_sl_amount",
                            help="ポジション全体の含み損が指定金額に達した時点で全玉損切します。"
                        )
                        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
                        p_emergency_exit = st.checkbox(
                            "🛡️ 【試験実装】ナンピン段階が半分超でゼロ円清算（同値撤退モード）",
                            value=bool(bt_saved.get("emergency_breakeven_exit", False)),
                            key="bt_emergency_exit_toggle",
                            help="ONにすると、ナンピン段階が設定の半分以上（例: 8回目以降）に達した場合、通常の利確幅(+200円)を待たず、平均取得単価（手数料分カバー）で即座にポジションを全玉清算して脱出します。"
                        )

                    # ナンピンロットテーブル
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.markdown("**🔢 ナンピンロットテーブル（ロング・ショート共通）＆ テンプレート管理**")
                    
                    tpl_col1, tpl_col2, tpl_col3, tpl_col4, tpl_col5 = st.columns([3, 1.2, 3, 1.2, 1.2])
                    tpl_names = list(lot_templates.keys())
                    
                    with tpl_col1:
                        sel_tpl_name = st.selectbox(
                            "ロットテンプレート選択",
                            options=tpl_names if tpl_names else ["(保存なし)"],
                            label_visibility="collapsed",
                            key="sel_lot_tpl_box"
                        )
                    with tpl_col2:
                        btn_load_tpl = st.button("呼出", use_container_width=True, key="btn_load_lot_tpl")
                        if btn_load_tpl and tpl_names and sel_tpl_name in lot_templates:
                            tpl_val = lot_templates[sel_tpl_name]
                            st.session_state["backtest_settings"]["lot_table"] = tpl_val
                            st.session_state["bt_lot_table_text"] = tpl_val
                            SettingsManager.save_settings({
                                "show_volume": st.session_state.get("show_volume", False),
                                "main_indicators": st.session_state["main_indicators"],
                                "sub_indicators": st.session_state["sub_indicators"],
                                "indicator_params": st.session_state["indicator_params"],
                                "backtest_settings": st.session_state["backtest_settings"],
                                "lot_table_templates": st.session_state.get("lot_table_templates", {}),
                                "strategy_presets": st.session_state.get("strategy_presets", {})
                            })
                            st.success(f"ロットテンプレート「{sel_tpl_name}」を設定値欄に反映しました！")
                            st.rerun()

                    with tpl_col3:
                        new_tpl_name = st.text_input("テンプレ保存名", placeholder="例: 攻め8段 (36枚)", label_visibility="collapsed", key="txt_new_lot_tpl")
                    with tpl_col4:
                        btn_save_tpl = st.button("保存", use_container_width=True, key="btn_save_lot_tpl")
                    with tpl_col5:
                        btn_del_tpl = st.button("削除", use_container_width=True, key="btn_del_lot_tpl")

                    init_lot_text = bt_saved.get("lot_table", "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10")
                    current_lot_text = st.text_input(
                        "ロット設定値 (カンマ区切り)",
                        value=init_lot_text,
                        key="bt_lot_table_text",
                        help="逆行250円ごとの追加発注枚数をカンマ区切りで入力します。ロング・ショート共通で使用されます。"
                    )

                    if btn_save_tpl:
                        tpl_save_name = new_tpl_name.strip()
                        if tpl_save_name:
                            updated_tpls = SettingsManager.save_lot_template(tpl_save_name, current_lot_text.strip())
                            st.session_state["lot_table_templates"] = updated_tpls
                            st.success(f"テンプレート「{tpl_save_name}」を確実に保存しました！")
                            st.rerun()

                    if btn_del_tpl and sel_tpl_name in lot_templates:
                        updated_tpls = SettingsManager.delete_lot_template(sel_tpl_name)
                        st.session_state["lot_table_templates"] = updated_tpls
                        st.info(f"テンプレート「{sel_tpl_name}」を削除しました。")
                        st.rerun()

                    try:
                        parsed_lots = [int(x.strip()) for x in current_lot_text.split(',') if x.strip()]
                        if not parsed_lots:
                            parsed_lots = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]
                    except:
                        parsed_lots = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]

                    # 証拠金・資本金に対する購入可能枚数リアルタイム計算バー
                    total_table_lots = sum(parsed_lots)
                    max_buyable_lots = int(bt_capital // bt_margin) if bt_margin > 0 else 0
                    total_margin_needed = total_table_lots * bt_margin
                    remaining_margin_lots = max_buyable_lots - total_table_lots
                    
                    half_idx = max(1, len(parsed_lots) // 2)
                    margin_warning = total_table_lots > max_buyable_lots

                    warn_style = "color: #ff1744; font-weight: bold;" if margin_warning else "color: #00e676; font-weight: bold;"

                    st.markdown(f"""
                    <div style="background-color: #1a1d24; border: 1px solid {'#ff1744' if margin_warning else '#2e3340'}; border-radius: 6px; padding: 10px 14px; margin-top: 6px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 13px;">
                            <span>💰 資本金: <b>¥{bt_capital:,.0f}</b> (1枚証拠金: <b>¥{bt_margin:,.0f}</b>)</span>
                            <span>🛒 資本金での最大購入可能枚数: <b style="font-size: 14px; color: #ffd600;">{max_buyable_lots} 枚</b></span>
                            <span>📊 設定ナンピン合計: <b>{len(parsed_lots)} 段階</b> (半分: <b>第{half_idx}段</b>) / <b style="{warn_style}">{total_table_lots} 枚</b> (必要証拠金: ¥{total_margin_needed:,.0f})</span>
                            <span>🛡️ 残り余力: <b style="{warn_style}">{remaining_margin_lots:+d} 枚</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if margin_warning:
                        st.caption("⚠️ 警告: 設定ロットの合計枚数が必要証拠金ベースで資本金を超過しています。ナンピン後半で資金不足になる可能性があります。")

            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            bt_btn_col1, bt_btn_col2, bt_btn_col3 = st.columns([2.5, 2.5, 5])
            with bt_btn_col1:
                btn_run_bt = st.button("🚀 バックテスト実行", use_container_width=True, key="btn_run_backtest")
            with bt_btn_col2:
                btn_clear_bt = st.button("🧹 シグナル表示クリア", use_container_width=True, key="btn_clear_backtest")
            with bt_btn_col3:
                st.caption("※選択された期間・足種データに対して売買シミュレーションを実行し、上のチャートに売買・ナンピン・決済点を描画します。")

            if btn_clear_bt:
                if "backtest_result" in st.session_state:
                    del st.session_state["backtest_result"]
                if "trade_signals" in st.session_state:
                    del st.session_state["trade_signals"]
                st.rerun()

            if btn_run_bt:
                with st.spinner("バックテストを実行中..."):
                    bt_res = run_strategy_backtest(df_plot)
                    st.success(f"✅ バックテスト完了！ 総トレード数: {bt_res.metrics['total_trades']}回（エントリー・決済完了） | 純損益: ¥{bt_res.metrics['total_pnl']:+,.0f}")
                    st.rerun()

            # バックテスト結果の表示
            if "backtest_result" in st.session_state:
                res: BacktestResult = st.session_state["backtest_result"]
                m = res.metrics

                st.markdown("---")
                mode_str = ""
                if hasattr(res, 'config') and hasattr(res.config, 'entry_mode'):
                    if res.config.entry_mode == "TREND":
                        mode_str = " <span style='background-color: #0d47a1; color: #90caf9; padding: 3px 10px; border-radius: 4px; font-size: 13px; font-weight: bold;'>🚀 IFモード (順張り検証)</span>"
                    else:
                        mode_str = " <span style='background-color: #1b5e20; color: #a5d6a7; padding: 3px 10px; border-radius: 4px; font-size: 13px; font-weight: bold;'>🔄 通常モード (逆張り)</span>"
                st.markdown(f"**📊 シミュレーション結果サマリー (KPI)** {mode_str}", unsafe_allow_html=True)
                
                # サマリー指標カード
                kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
                with kpi_col1:
                    pnl_color = "#4caf50" if m['total_pnl'] >= 0 else "#ef5350"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">純損益 (Total P&L)</div>
                        <div class="kpi-value" style="color: {pnl_color};">¥{m['total_pnl']:+,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with kpi_col2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">勝率 (Win Rate)</div>
                        <div class="kpi-value">{m['win_rate']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with kpi_col3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">PF (プロフィットファクター)</div>
                        <div class="kpi-value">{m['profit_factor']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with kpi_col4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">最大ドローダウン</div>
                        <div class="kpi-value" style="color: #ef5350;">-¥{m['max_drawdown']:,.0f} ({m['max_drawdown_pct']:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with kpi_col5:
                    if "max_nanpin_reached" in m:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">最大ナンピン段数</div>
                            <div class="kpi-value">{m['max_nanpin_reached']} 回 (平均: {m.get('avg_nanpin_count', 0.0):.1f})</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">ペイオフレシオ (損益比)</div>
                            <div class="kpi-value">{m.get('payoff_ratio', 0.0):.2f} (平均益: ¥{m.get('avg_win_pnl', 0):,.0f})</div>
                        </div>
                        """, unsafe_allow_html=True)
                with kpi_col6:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">総トレード数 (完了分)</div>
                        <div class="kpi-value">{m['total_trades']} 回 (勝:{m['win_trades']} / 負:{m['loss_trades']})</div>
                    </div>
                    """, unsafe_allow_html=True)

                if "max_nanpin_reached" in m:
                    st.caption(f"※総トレード数 {m['total_trades']}回はエントリーから全玉決済完了までの取引サイクル数です。（ナンピン発注は同一トレード内の追加玉として扱い、トレード数には含めません。期間内ナンピン総発注: 計 {m.get('total_nanpin_orders', 0)}回、総取引枚数: {m.get('total_lots_traded', 0)}枚）")
                else:
                    st.caption(f"※総トレード数 {m['total_trades']}回 （最大連勝: {m.get('max_consecutive_wins', 0)}連勝 / 最大連敗: {m.get('max_consecutive_losses', 0)}連敗 / シャープレシオ: {m.get('sharpe_ratio', 0.0):.2f}）")

                # 戦略別集計 & トレードログ一覧
                res_col1, res_col2 = st.columns([4, 6])
                with res_col1:
                    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                    if "nanpin_count" in res.trades_df.columns:
                        st.markdown("**🔢 ナンピン到達回数別の集計**")
                        if not res.trades_df.empty:
                            nanpin_summary = res.trades_df.groupby('nanpin_count').agg(
                                トレード数=('trade_no' if 'trade_no' in res.trades_df.columns else 'trade_id', 'count'),
                                勝率=('net_pnl' if 'net_pnl' in res.trades_df.columns else 'net_pnl_yen', lambda x: f"{(x > 0).mean()*100:.1f}%"),
                                損益合計=('net_pnl' if 'net_pnl' in res.trades_df.columns else 'net_pnl_yen', lambda x: f"¥{x.sum():+,.0f}"),
                                平均損益=('net_pnl' if 'net_pnl' in res.trades_df.columns else 'net_pnl_yen', lambda x: f"¥{x.mean():+,.0f}")
                            ).reset_index()
                            nanpin_summary = nanpin_summary.rename(columns={'nanpin_count': 'ナンピン回数'})
                            st.dataframe(nanpin_summary, use_container_width=True, hide_index=True)
                    elif hasattr(res, 'period_analysis') and not res.period_analysis["monthly"].empty:
                        st.markdown("**📅 月別損益・パフォーマンス集計**")
                        m_df = res.period_analysis["monthly"].copy()
                        m_display = pd.DataFrame({
                            "年月": m_df["period_str"],
                            "回数": m_df["total_trades"],
                            "勝率": m_df["win_rate"].apply(lambda x: f"{x:.1f}%"),
                            "損益(円)": m_df["total_pnl"].apply(lambda x: f"¥{x:+,.0f}"),
                            "PF": m_df["profit_factor"].apply(lambda x: f"{x:.2f}" if x < 999 else "∞")
                        })
                        st.dataframe(m_display, use_container_width=True, hide_index=True)
                        
                with res_col2:
                    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                    st.markdown("**📝 取引履歴ログ (最新20件)**")
                    if not res.trades_df.empty:
                        df_tr = res.trades_df.copy()
                        if "trade_no" in df_tr.columns:
                            cols = ['trade_no', 'pos_type', 'entry_time', 'exit_time', 'avg_entry_price', 'exit_price', 'total_lots', 'nanpin_count', 'net_pnl', 'exit_reason']
                            renames = {
                                'trade_no': '#', 'pos_type': '種別', 'entry_time': 'エントリー', 'exit_time': '決済日時',
                                'avg_entry_price': '平均単価', 'exit_price': '決済価格', 'total_lots': '枚数',
                                'nanpin_count': 'ナンピン', 'net_pnl': '純損益(円)', 'exit_reason': '決済理由'
                            }
                            display_trades = df_tr[[c for c in cols if c in df_tr.columns]].rename(columns=renames)
                        else:
                            cols = ['trade_id', 'pos_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'lots', 'pnl_points', 'net_pnl_yen', 'exit_reason']
                            renames = {
                                'trade_id': '#', 'pos_type': '種別', 'entry_time': 'エントリー', 'exit_time': '決済日時',
                                'entry_price': '仕掛値', 'exit_price': '決済値', 'lots': '枚数',
                                'pnl_points': '値幅(pt)', 'net_pnl_yen': '純損益(円)', 'exit_reason': '決済理由'
                            }
                            display_trades = df_tr[[c for c in cols if c in df_tr.columns]].rename(columns=renames)
                            
                        st.dataframe(display_trades.tail(20), use_container_width=True, hide_index=True)
                        
                        csv_data = res.trades_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            "📥 全取引履歴CSVをダウンロード",
                            data=csv_data,
                            file_name=f"trade_history_{st.session_state.get('selected_strategy_key', 'strategy')}.csv",
                            mime="text/csv"
                        )

            st.markdown('</div>', unsafe_allow_html=True)

        with tab_preview:
            preview_df = df_plot.head(15).copy()
            for ind_id, df_res in {**main_indicator_data, **sub_indicator_data}.items():
                for col in df_res.columns:
                    preview_df[f"{ind_id}_{col}"] = df_res[col].head(15)
            st.dataframe(preview_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

