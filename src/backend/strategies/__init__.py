import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List, Any, Optional

from src.backend.strategies.base import BaseStrategy, BaseStrategyConfig, BacktestResult, TradeLog

STRATEGIES_DIR = Path(__file__).resolve().parent

# ビルトインコア戦略 (削除不可)
BUILTIN_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "grid_nanpin": {
        "key": "grid_nanpin",
        "name": "High＆Low逆張りナンピン戦略",
        "description": "直近N期間高値・安値ブレイクアウト逆張り ＋ グリッドナンピン＆利確",
        "default_timeframe": "15min",
        "supported_timeframes": ["5min", "15min", "30min", "60min"],
        "strategy_module": "src.backend.grid_nanpin_strategy",
        "strategy_class_name": "GridNanpinBacktester",
        "config_class_name": "GridNanpinConfig",
        "is_builtin": True,
        "is_deletable": False,
        "file_path": None
    }
}


def discover_strategy_plugins() -> Dict[str, Dict[str, Any]]:
    """
    src/backend/strategies ディレクトリをスキャンし、
    BaseStrategy を継承したプラグイン戦略ファイルを動的に自動検出・登録する。
    """
    registry = BUILTIN_STRATEGIES.copy()
    
    if not STRATEGIES_DIR.exists():
        return registry

    for py_file in STRATEGIES_DIR.glob("*.py"):
        if py_file.stem in ["__init__", "base"]:
            continue

        mod_name = f"src.backend.strategies.{py_file.stem}"
        try:
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)

            strat_cls = None
            cfg_cls = None

            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if inspect.isclass(attr):
                    if issubclass(attr, BaseStrategy) and attr is not BaseStrategy:
                        strat_cls = attr
                    elif issubclass(attr, BaseStrategyConfig) and attr is not BaseStrategyConfig:
                        cfg_cls = attr

            if strat_cls is not None:
                # 戦略キーの決定
                key = getattr(strat_cls, "STRATEGY_KEY", py_file.stem)
                name = getattr(strat_cls, "STRATEGY_NAME", py_file.stem)
                desc = getattr(strat_cls, "STRATEGY_DESCRIPTION", "")
                tf = getattr(strat_cls, "DEFAULT_TIMEFRAME", "30min")
                supported_tfs = getattr(strat_cls, "SUPPORTED_TIMEFRAMES", ["15min", "30min", "60min"])

                registry[key] = {
                    "key": key,
                    "name": name,
                    "description": desc,
                    "default_timeframe": tf,
                    "supported_timeframes": supported_tfs,
                    "strategy_class": strat_cls,
                    "config_class": cfg_cls or BaseStrategyConfig,
                    "is_builtin": False,
                    "is_deletable": True,
                    "file_path": str(py_file.resolve())
                }
        except Exception as e:
            print(f"[StrategyPlugin] プラグイン読み込み失敗 ({py_file.name}): {e}")

    return registry


def list_strategies() -> List[Dict[str, Any]]:
    """現在ロードされている全戦略モデル一覧を取得"""
    registry = discover_strategy_plugins()
    result = []
    for key, val in registry.items():
        result.append({
            "key": val["key"],
            "name": val["name"],
            "description": val.get("description", ""),
            "default_timeframe": val.get("default_timeframe", "15min"),
            "supported_timeframes": val.get("supported_timeframes", ["15min"]),
            "is_builtin": val.get("is_builtin", False),
            "is_deletable": val.get("is_deletable", True),
            "file_path": val.get("file_path", None)
        })
    return result


def get_strategy_info(strategy_key: str) -> Dict[str, Any]:
    """戦略のメタ情報を取得"""
    registry = discover_strategy_plugins()
    if strategy_key not in registry:
        raise ValueError(f"Unknown strategy key: {strategy_key}")
    return registry[strategy_key]


def create_strategy_instance(strategy_key: str, config_dict: Dict[str, Any] = None) -> BaseStrategy:
    """戦略インスタンスを動的に生成"""
    info = get_strategy_info(strategy_key)
    
    if "strategy_class" in info and info["strategy_class"] is not None:
        strategy_cls = info["strategy_class"]
        config_cls = info["config_class"]
        if config_dict:
            config = config_cls.from_dict(config_dict)
        else:
            config = config_cls()
        return strategy_cls(config)
    else:
        # ビルトイン（モジュール名から動的ロード）
        mod = importlib.import_module(info["strategy_module"])
        strategy_cls = getattr(mod, info["strategy_class_name"])
        config_cls = getattr(mod, info["config_class_name"])
        if config_dict:
            config = config_cls(**config_dict)
        else:
            config = config_cls()
        return strategy_cls(config)


def delete_strategy_plugin(strategy_key: str) -> bool:
    """
    指定した戦略モデルの専用ロジックファイルを削除（アンインストール）する。
    ビルトインコア戦略（grid_nanpin等）や基底ファイルは保護される。
    """
    registry = discover_strategy_plugins()
    if strategy_key not in registry:
        raise ValueError(f"戦略 '{strategy_key}' は見つかりません。")
        
    info = registry[strategy_key]
    if not info.get("is_deletable", False) or info.get("is_builtin", False):
        raise PermissionError(f"戦略 '{strategy_key}' はビルトインの基盤戦略のため削除できません。")

    fpath_str = info.get("file_path")
    if not fpath_str:
        raise FileNotFoundError(f"戦略 '{strategy_key}' のファイルパスが不明です。")

    target_file = Path(fpath_str)
    if target_file.exists() and target_file.is_file():
        # 安全のためにバックアップ/退避フォルダに移動するか、削除
        archive_dir = Path(__file__).resolve().parent.parent.parent / "data" / "strategies_archived"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 退避先に同名があれば上書き退避
        dest_file = archive_dir / target_file.name
        if dest_file.exists():
            dest_file.unlink()
        target_file.rename(dest_file)
        
        # モジュールキャッシュのクリーンアップ
        mod_name = f"src.backend.strategies.{target_file.stem}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
            
        return True
    else:
        raise FileNotFoundError(f"ファイル {target_file} が存在しません。")
