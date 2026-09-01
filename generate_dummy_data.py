import pandas as pd
import numpy as np
from pathlib import Path

def generate_dummy_data(rows=1000):
    np.random.seed(42)
    # 2024-01-01からの1分足
    dates = pd.date_range(start='2024-01-01 09:00:00', periods=rows, freq='1min')
    
    # 始値をランダムウォークで生成（初期値40000円）
    steps = np.random.normal(loc=0, scale=10, size=rows)
    # 5円刻み（マイクロ先物の呼値）にする
    steps = np.round(steps / 5.0) * 5.0
    
    open_prices = 40000 + np.cumsum(steps)
    
    # 高値、安値、終値の生成
    high_prices = open_prices + np.abs(np.random.normal(0, 15, rows))
    low_prices = open_prices - np.abs(np.random.normal(0, 15, rows))
    close_prices = open_prices + np.random.normal(0, 10, rows)
    
    # 高安が始値終値を包むように調整
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # すべて5円刻みに丸める
    high_prices = np.round(high_prices / 5.0) * 5.0
    low_prices = np.round(low_prices / 5.0) * 5.0
    close_prices = np.round(close_prices / 5.0) * 5.0
    open_prices = np.round(open_prices / 5.0) * 5.0
    
    # 出来高の生成
    volumes = np.random.randint(10, 500, size=rows)
    
    df = pd.DataFrame({
        '日時': dates,
        '始値': open_prices,
        '高値': high_prices,
        '安値': low_prices,
        '終値': close_prices,
        '出来高': volumes
    })
    
    output_dir = Path(__file__).resolve().parent / 'data' / 'market'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'dummy_data.csv'
    df.to_csv(output_path, index=False)
    print(f"ダミーデータを生成しました: {output_path}")

if __name__ == "__main__":
    generate_dummy_data()
