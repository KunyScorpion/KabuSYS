from dataclasses import dataclass

@dataclass
class InstrumentConfig:
    name: str
    tick_size: float
    multiplier: int

# 日経225マイクロ先物の設定
NIKKEI225_MICRO = InstrumentConfig(
    name="Nikkei 225 Micro",
    tick_size=5.0,  # 呼値（5円刻み）
    multiplier=10   # 倍率（10倍）
)

# デフォルト設定
DEFAULT_INSTRUMENT = NIKKEI225_MICRO
