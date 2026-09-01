import json
from pathlib import Path
from typing import Dict, Any

SETTINGS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "user_settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    # チャート表示設定 (出来高はデフォルト非表示)
    "show_volume": False,
    # メインチャートに表示する指標（デフォルトは移動平均線: SMA と ハイローバンド: HI_LOW_Bands）
    "main_indicators": ["SMA", "HI_LOW_Bands"],
    # サブチャートに表示する指標
    "sub_indicators": [],
    # 各指標のパラメータ設定（期間パラメータなど）
    "indicator_params": {
        "SMA": {"period1": 5, "period2": 25, "period3": 75},
        "EMA": {"period1": 9, "period2": 13, "period3": 26},
        "WMA": {"period1": 5, "period2": 20},
        "Bollinger_Bands": {"period": 20, "nbdev": 2.0},
        "Envelope": {"period": 20, "deviation1": 1.0, "deviation2": 2.0},
        "Ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
        "HI_LOW_Bands": {"period": 20},
        "Keltner_Channel": {"ema_period": 20, "atr_period": 10, "multiplier": 1.5},
        "Meander": {"period": 13, "multiplier": 1.0},
        "MACD": {"fast_period": 9, "slow_period": 13, "signal_period": 3},
        "RSI": {"period": 9, "upper": 70, "lower": 30},
        "Stochastics": {"k_period": 5, "d_period": 3, "sd_period": 3},
        "Stochastics_RSI": {"rsi_period": 9, "k_period": 3, "d_period": 3, "sd_period": 3},
        "ATR": {"period": 10},
        "DMI": {"period": 14},
        "RCI": {"period1": 9, "period2": 26},
        "ROC": {"period": 7},
        "CMO": {"period": 14},
        "Psychological_Line": {"period": 12},
        "Balance_of_Power": {"period": 14},
        "Sinohara_Ratio": {"period": 26},
        "Volatility_Ratio": {"period": 9},
        "R_Oscillator": {"period": 5},
        "Estrangement_SMA": {"period": 25},
        "Estrangement_EMA": {"period": 25},
        "Estrangement_WMA": {"period": 25},
        "Binary_Wave": {"macd_fast": 9, "macd_slow": 13, "macd_signal": 3, "rsi_period": 9, "sma_period": 5},
        "Ultimate_Oscillator": {"period1": 7, "period2": 14, "period3": 28}
    },
    # バックテスト設定の永続化
    "backtest_settings": {
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
    # ナンピンロットテーブルの保存テンプレート辞書
    "lot_table_templates": {
        "標準16段 (73枚)": "1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10",
        "堅実16段 (46枚)": "1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7",
        "攻め8段 (36枚)": "1, 2, 3, 4, 5, 6, 7, 8"
    },
    # 戦略パラメータ全体のお気に入りプリセット辞書
    "strategy_presets": {
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
}

BACKUP_SETTINGS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "user_settings.backup.json"

class SettingsManager:
    """
    ユーザーの表示指標選択、期間パラメータ、バックテスト設定、ロットテンプレート、お気に入りプリセットをJSONファイルで永続化・管理するクラス。
    """
    
    @staticmethod
    def load_settings() -> Dict[str, Any]:
        """設定をロードする。ファイルが存在しない・破損している場合はデフォルト設定を返す。"""
        merged = DEFAULT_SETTINGS.copy()
        merged["lot_table_templates"] = DEFAULT_SETTINGS["lot_table_templates"].copy()
        merged["strategy_presets"] = DEFAULT_SETTINGS["strategy_presets"].copy()
        merged["indicator_params"] = {k: v.copy() for k, v in DEFAULT_SETTINGS["indicator_params"].items()}
        merged["backtest_settings"] = DEFAULT_SETTINGS["backtest_settings"].copy()

        if not SETTINGS_FILE.exists():
            return merged
        
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "show_volume" in data:
                merged["show_volume"] = bool(data["show_volume"])
            if "main_indicators" in data and isinstance(data["main_indicators"], list):
                merged["main_indicators"] = data["main_indicators"]
            if "sub_indicators" in data and isinstance(data["sub_indicators"], list):
                merged["sub_indicators"] = data["sub_indicators"]
            if "indicator_params" in data and isinstance(data["indicator_params"], dict):
                for k, v in data["indicator_params"].items():
                    if k in merged["indicator_params"]:
                        merged["indicator_params"][k].update(v)
                    else:
                        merged["indicator_params"][k] = v
            if "backtest_settings" in data and isinstance(data["backtest_settings"], dict):
                merged["backtest_settings"].update(data["backtest_settings"])
            if "lot_table_templates" in data and isinstance(data["lot_table_templates"], dict) and len(data["lot_table_templates"]) > 0:
                merged["lot_table_templates"] = data["lot_table_templates"]
            if "strategy_presets" in data and isinstance(data["strategy_presets"], dict) and len(data["strategy_presets"]) > 0:
                merged["strategy_presets"] = data["strategy_presets"]
            return merged
        except Exception as e:
            print(f"[SettingsManager] 設定の読み込みエラー（バックアップ確認）: {e}")
            if BACKUP_SETTINGS_FILE.exists():
                try:
                    with open(BACKUP_SETTINGS_FILE, "r", encoding="utf-8") as bf:
                        b_data = json.load(bf)
                    if isinstance(b_data, dict):
                        return b_data
                except Exception:
                    pass
            return merged

    @staticmethod
    def save_settings(settings: Dict[str, Any]) -> None:
        """
        設定を安全にJSONファイルに保存する。
        既存のディスク上の設定をマージし、空辞書によるテンプレート・お気に入りの消失を防止する。
        """
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # 既存のディスク上の設定を読み込み
            existing = {}
            if SETTINGS_FILE.exists():
                try:
                    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    # 直前バックアップを保存
                    with open(BACKUP_SETTINGS_FILE, "w", encoding="utf-8") as bf:
                        json.dump(existing, bf, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # マージ処理
            final_data = existing.copy() if existing else DEFAULT_SETTINGS.copy()
            for k, v in settings.items():
                if k in ["lot_table_templates", "strategy_presets"]:
                    # 空辞書の場合は既存のユーザーデータを上書き消去しない
                    if isinstance(v, dict) and len(v) > 0:
                        final_data[k] = v
                    elif k not in final_data:
                        final_data[k] = DEFAULT_SETTINGS.get(k, {}).copy()
                else:
                    final_data[k] = v

            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SettingsManager] 設定の保存エラー: {e}")

    @staticmethod
    def save_lot_template(name: str, lot_str: str) -> Dict[str, str]:
        """ロットテンプレートを1件追加・更新して即座に安全保存する。"""
        current = SettingsManager.load_settings()
        tpls = current.get("lot_table_templates", {})
        tpls[name.strip()] = lot_str.strip()
        current["lot_table_templates"] = tpls
        SettingsManager.save_settings(current)
        return tpls

    @staticmethod
    def delete_lot_template(name: str) -> Dict[str, str]:
        """ロットテンプレートを1件削除して即座に安全保存する。"""
        current = SettingsManager.load_settings()
        tpls = current.get("lot_table_templates", {})
        if name in tpls:
            del tpls[name]
        current["lot_table_templates"] = tpls
        # 明示的な削除時はファイルに直接保存
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SettingsManager] 削除保存エラー: {e}")
        return tpls

    @staticmethod
    def save_strategy_preset(name: str, preset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """お気に入りプリセットを1件追加・更新して即座に安全保存する。"""
        current = SettingsManager.load_settings()
        presets = current.get("strategy_presets", {})
        presets[name.strip()] = preset_dict
        current["strategy_presets"] = presets
        SettingsManager.save_settings(current)
        return presets

    @staticmethod
    def delete_strategy_preset(name: str) -> Dict[str, Any]:
        """お気に入りプリセットを1件削除して即座に安全保存する。"""
        current = SettingsManager.load_settings()
        presets = current.get("strategy_presets", {})
        if name in presets:
            del presets[name]
        current["strategy_presets"] = presets
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SettingsManager] プリセット削除保存エラー: {e}")
        return presets
