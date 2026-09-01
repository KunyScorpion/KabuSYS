import pandas as pd
from pathlib import Path
from typing import Optional, List

class DataParser:
    """
    市場データを読み込み、統一フォーマットのPandas DataFrameに変換するパーサー。
    """
    
    # 想定されるカラム名のマッピング（入力: 統一後）
    COLUMN_MAPPING = {
        '日時': 'time',
        '日付': 'time',
        'Date': 'time',
        'Time': 'time',
        'Timestamp': 'time',
        '始値': 'open',
        'Open': 'open',
        '高値': 'high',
        'High': 'high',
        '安値': 'low',
        'Low': 'low',
        '終値': 'close',
        'Close': 'close',
        '出来高': 'volume',
        'Volume': 'volume'
    }

    TIMEFRAME_MAPPING = {
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

    @staticmethod
    def format_timeframe_name(sheet_name: str) -> str:
        """シート名をユーザー向けの時間足表記に変換"""
        return DataParser.TIMEFRAME_MAPPING.get(sheet_name, sheet_name)

    @staticmethod
    def get_sheet_names(file_path: str | Path) -> List[str]:
        """Excelファイルのシート名一覧を取得。CSVの場合は空リストを返す。"""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext in ['.xlsx', '.xls']:
            try:
                xl = pd.ExcelFile(path)
                return xl.sheet_names
            except Exception:
                return []
        return []

    @staticmethod
    def load_file(file_path: str | Path, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        # 拡張子に応じた読み込み
        ext = path.suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext in ['.xlsx', '.xls']:
            try:
                xl = pd.ExcelFile(path)
            except Exception as e:
                raise ValueError(f"Excelファイルの読み込みに失敗しました: {path} ({e})")

            target_sheet = None
            if sheet_name:
                available_sheets = xl.sheet_names
                # 完全一致の確認
                if sheet_name in available_sheets:
                    target_sheet = sheet_name
                else:
                    # 空白除去や大文字小文字を無視したフォールバックマッチング
                    cleaned_target = str(sheet_name).strip().lower()
                    for s in available_sheets:
                        if str(s).strip().lower() == cleaned_target:
                            target_sheet = s
                            break

                # 指定されたシートがこのファイル内に存在しない場合はNoneを返して安全にスキップ可能にする
                if target_sheet is None:
                    return None

                df = pd.read_excel(xl, sheet_name=target_sheet)
            else:
                df = pd.read_excel(xl)
        else:
            raise ValueError(f"未対応のファイル形式です: {ext}")

        return DataParser.normalize_dataframe(df)

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームのカラム名を統一し、日時をインデックスまたは専用カラムに設定しソートする。
        """
        # カラム名のリネーム（大文字小文字を区別せず、マッピングにあるものを変換）
        rename_dict = {}
        for col in df.columns:
            for key, val in DataParser.COLUMN_MAPPING.items():
                if str(col).lower() == key.lower():
                    rename_dict[col] = val
                    break
        
        df = df.rename(columns=rename_dict)
        
        # '日付'（今は 'time' にリネームされている）と '時間' が分かれている場合の特別対応
        # 先物特有の「営業日（Trading Date）」と「実カレンダー日（Calendar Date）」のズレを補正する
        time_col = None
        if '時間' in df.columns:
            time_col = '時間'
        elif 'Time' in df.columns:
            time_col = 'Time'
            
        if time_col and 'time' in df.columns:
            import datetime
            calendar_dates = []
            current_date = None
            last_time = None
            
            day_start = datetime.time(8, 0)
            day_end = datetime.time(15, 45)
            night_start = datetime.time(16, 0)
            
            for _, row in df.iterrows():
                trading_date = pd.to_datetime(row['time']).date()
                t_val = row[time_col]
                
                if isinstance(t_val, str):
                    try:
                        t_val = pd.to_datetime(t_val).time()
                    except:
                        t_val = datetime.time(0, 0)
                elif isinstance(t_val, datetime.datetime):
                    t_val = t_val.time()
                elif not isinstance(t_val, datetime.time):
                    t_val = datetime.time(0, 0)
                    
                # 日中セッション（08:00〜15:45頃）は営業日＝カレンダー日
                if day_start <= t_val <= day_end:
                    current_date = trading_date
                else:
                    # 初回行がナイトセッションの場合のフォールバック
                    if current_date is None:
                        if t_val >= night_start:
                            current_date = trading_date - datetime.timedelta(days=3 if trading_date.weekday() == 0 else 1)
                        else:
                            current_date = trading_date
                    
                    # 23:59 から 00:00 へ日跨ぎした場合、カレンダー日を+1日進める
                    if last_time is not None and last_time > t_val and last_time >= night_start:
                        current_date = current_date + datetime.timedelta(days=1)
                        
                calendar_dates.append(pd.Timestamp.combine(current_date, t_val))
                last_time = t_val
                
            df['time'] = calendar_dates
            # 使い終わった時間カラムは削除
            df = df.drop(columns=[time_col])
        elif 'time' not in df.columns:
            # 結合済みの '日時' 等がない場合のフォールバック
            if '日付' in df.columns and '時間' in df.columns:
                df['time'] = df['日付'].astype(str) + ' ' + df['時間'].astype(str)
            elif 'Date' in df.columns and 'Time' in df.columns:
                df['time'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
            elif '日付' in df.columns:
                df['time'] = df['日付']
            elif 'Date' in df.columns:
                df['time'] = df['Date']

        # 必須カラムのチェック
        required_cols = ['time', 'open', 'high', 'low', 'close']
        for req_col in required_cols:
            if req_col not in df.columns:
                raise ValueError(f"必須カラム '{req_col}' が見つかりません。")

        # 日時型の変換とソート
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
        
        # 出来高がない場合は0埋め
        if 'volume' not in df.columns:
            df['volume'] = 0

        # 数値型の強制変換
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # 欠損値（NaN）のフォワードフィル等（簡易対応）
        df = df.ffill().bfill()
        
        return df

    @staticmethod
    def get_available_files(directory: str | Path) -> List[Path]:
        """指定ディレクトリ内の読み込み可能なファイル一覧を取得"""
        path = Path(directory)
        if not path.exists():
            return []
        
        files = []
        for ext in ['*.csv', '*.xlsx', '*.xls']:
            files.extend(path.glob(ext))
        return sorted(files)
