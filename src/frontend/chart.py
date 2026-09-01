import json
import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit.components.v1 as components

def render_highstock_chart(
    df: pd.DataFrame = None,
    df_ohlcv: pd.DataFrame = None,
    main_indicator_data: Dict[str, pd.DataFrame] = None,
    main_indicator_meta: Dict[str, Dict[str, Any]] = None,
    sub_indicator_data: Dict[str, pd.DataFrame] = None,
    sub_indicator_meta: Dict[str, Dict[str, Any]] = None,
    trade_signals: List[Dict[str, Any]] = None,
    bar_metrics: List[Dict[str, Any]] = None,
    show_volume: bool = False,
    height: int = 720,
    symbol_name: str = "日経225マイクロ",
    **kwargs
):
    """
    OHLCVデータ、テクニカル指標、および売買シグナル（エントリー、ナンピン、決済）を受け取り、
    ダークテーマ・Highcharts Stock チャートを描画する。
    """
    df_ohlcv = df if df is not None else df_ohlcv
    if df_ohlcv is None or df_ohlcv.empty:
        return

    main_indicator_data = main_indicator_data or {}
    main_indicator_meta = main_indicator_meta or {}
    sub_indicator_data = sub_indicator_data or {}
    sub_indicator_meta = sub_indicator_meta or {}
    trade_signals = trade_signals or []
    bar_metrics = bar_metrics or []

    # タイムスタンプ（ミリ秒）
    # pandas datetime -> int64 (ns) -> ms
    times_ms = (df_ohlcv['time'].astype('int64') // 10**6).tolist()

    # 1. OHLCデータおよび出来高データの構築
    ohlc_data = []
    volume_data = []
    
    opens = df_ohlcv['open'].tolist()
    highs = df_ohlcv['high'].tolist()
    lows = df_ohlcv['low'].tolist()
    closes = df_ohlcv['close'].tolist()
    volumes = df_ohlcv['volume'].tolist() if 'volume' in df_ohlcv.columns else [0] * len(times_ms)

    for i in range(len(times_ms)):
        t = times_ms[i]
        o = float(opens[i])
        h = float(highs[i])
        l = float(lows[i])
        c = float(closes[i])
        v = float(volumes[i])
        
        ohlc_data.append([t, o, h, l, c])
        if show_volume:
            vol_color = '#3d6b99' if c >= o else '#8b3a3a'
            volume_data.append({
                'x': t,
                'y': v,
                'color': vol_color
            })

    # バックテスト情報のタイムスタンプ別マップ作成
    bt_map = {}
    for bm in bar_metrics:
        t_val = pd.to_datetime(bm.get('time'))
        t_ms = int(t_val.timestamp() * 1000)
        pos_type = bm.get('pos_type', 'NONE')
        pos_lots = bm.get('pos_lots', 0)
        avg_p = bm.get('avg_price', 0.0)
        unrealized = bm.get('unrealized_pnl', 0.0)
        
        if pos_lots > 0:
            pos_str = f"{pos_type} {pos_lots}枚 @ ¥{avg_p:,.0f}"
            pnl_str = f"¥{unrealized:+,.0f}"
        else:
            pos_str = "ノーポジ"
            pnl_str = "-"
            
        bt_map[t_ms] = {
            'pos': pos_str,
            'pnl': pnl_str,
            'unrealized': unrealized,
            'lots': pos_lots,
            'signals': []
        }

    # シグナル情報をマップに付与（同一バー内の複数シグナルをすべてリストで保持）
    for sig in trade_signals:
        t_val = pd.to_datetime(sig.get('time'))
        t_ms = int(t_val.timestamp() * 1000)
        detail = sig.get('detail', '')
        if t_ms not in bt_map:
            bt_map[t_ms] = {'pos': 'シグナル', 'pnl': '-', 'unrealized': 0, 'lots': 0, 'signals': []}
        if 'signals' not in bt_map[t_ms]:
            bt_map[t_ms]['signals'] = []
        bt_map[t_ms]['signals'].append(detail)

    # 2. メイン指標シリーズデータの構築（上段にオーバーレイ）
    main_series_list = []
    for ind_id, df_ind in main_indicator_data.items():
        meta = main_indicator_meta.get(ind_id, {})
        for s_info in meta.get("series", []):
            s_key = s_info["key"]
            if s_key in df_ind.columns:
                s_vals = df_ind[s_key].values
                points = []
                for i, val in enumerate(s_vals):
                    if pd.notnull(val) and not pd.isna(val):
                        points.append([times_ms[i], float(val)])
                
                dash_style = 'Dash' if s_info.get("style") == "dashed" else 'Solid'
                main_series_list.append({
                    "name": s_info.get("name", s_key),
                    "data": points,
                    "color": s_info.get("color", "#2962FF"),
                    "lineWidth": s_info.get("width", 1.5),
                    "dashStyle": dash_style,
                    "type": "line",
                    "yAxis": 0
                })

    # 3. サブ指標データの構築（下段に独立ペイン追加）
    sub_panes_list = []
    for ind_id, df_ind in sub_indicator_data.items():
        meta = sub_indicator_meta.get(ind_id, {})
        pane_obj = {
            "id": ind_id,
            "title": meta.get("name", ind_id),
            "baselines": meta.get("baselines", []),
            "series": []
        }
        for s_info in meta.get("series", []):
            s_key = s_info["key"]
            if s_key in df_ind.columns:
                s_vals = df_ind[s_key].values
                s_type = s_info.get("type", "line")
                points = []
                for i, val in enumerate(s_vals):
                    if pd.notnull(val) and not pd.isna(val):
                        v_float = float(val)
                        if s_type == "histogram":
                            h_col = '#3d6b99' if v_float >= 0 else '#8b3a3a'
                            points.append({'x': times_ms[i], 'y': v_float, 'color': h_col})
                        else:
                            points.append([times_ms[i], v_float])
                pane_obj["series"].append({
                    "name": s_info.get("name", s_key),
                    "data": points,
                    "color": s_info.get("color", "#2962FF"),
                    "lineWidth": s_info.get("width", 1.5),
                    "type": "column" if s_type == "histogram" else "line"
                })
        sub_panes_list.append(pane_obj)

    # 4. 売買シグナル（エントリー・ナンピン・利確・損切）データの構築
    signals_series_list = []
    if trade_signals:
        signal_types = {
            'SHORT_ENTRY': {'name': 'ショート開始', 'color': '#ff1744', 'symbol': 'triangle-down', 'radius': 7},
            'LONG_ENTRY': {'name': 'ロング開始', 'color': '#00e676', 'symbol': 'triangle', 'radius': 7},
            'NANPIN': {'name': 'ナンピン追加 (白)', 'color': '#ffffff', 'symbol': 'circle', 'radius': 4.5},
            'TP_EXIT': {'name': '利確決済 (黄星)', 'color': '#ffd600', 'symbol': 'star', 'radius': 9},
            'SL_EXIT': {'name': '損切決済 (紫星)', 'color': '#d500f9', 'symbol': 'star', 'radius': 9},
            'PERIOD_END_EXIT': {'name': '期間終了決済', 'color': '#90caf9', 'symbol': 'circle', 'radius': 6}
        }
        
        # イベント種別ごとにグループ化
        grouped_signals = {}
        for sig in trade_signals:
            ev = sig.get('event')
            t_val = pd.to_datetime(sig.get('time'))
            t_ms = int(t_val.timestamp() * 1000)
            p_val = float(sig.get('price', 0))
            detail = sig.get('detail', '')
            
            if ev in signal_types:
                grouped_signals.setdefault(ev, []).append({
                    'x': t_ms,
                    'y': p_val,
                    'detail': detail
                })
                
        for ev, st_cfg in signal_types.items():
            pts = grouped_signals.get(ev, [])
            if pts:
                signals_series_list.append({
                    'type': 'scatter',
                    'name': st_cfg['name'],
                    'data': pts,
                    'color': st_cfg['color'],
                    'marker': {
                        'symbol': st_cfg['symbol'],
                        'radius': st_cfg['radius'],
                        'enabled': True
                    },
                    'yAxis': 0,
                    'zIndex': 10,
                    'tooltip': {
                        'pointFormat': '<span style="color:{point.color}">●</span> <b>{series.name}</b>: {point.detail}<br/>'
                    }
                })

    # JSON シリアライズ
    ohlc_json = json.dumps(ohlc_data)
    volume_json = json.dumps(volume_data)
    main_series_json = json.dumps(main_series_list)
    sub_panes_json = json.dumps(sub_panes_list)
    signals_series_json = json.dumps(signals_series_list)
    bt_map_json = json.dumps(bt_map)
    show_volume_json = "true" if show_volume else "false"

    # ペインの高さ配分計算
    num_subs = len(sub_panes_list)
    chart_height = height + num_subs * 140

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highcharts/11.4.0/highstock.js"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background-color: #1e1e1e;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                overflow: hidden;
            }}
            #chart-wrapper {{
                position: relative;
                width: 100%;
                height: {chart_height}px;
            }}
            #container {{
                width: 100%;
                height: {chart_height}px;
            }}
            #reset-zoom-btn {{
                display: none;
                position: absolute;
                top: 10px;
                right: 95px;
                z-index: 100;
                background-color: #2a2e39;
                color: #e0e3eb;
                border: 1px solid #4a5061;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11.5px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
                transition: background-color 0.2s, border-color 0.2s;
            }}
            #reset-zoom-btn:hover {{
                background-color: #3d4454;
                border-color: #636b7e;
                color: #ffffff;
            }}
            #error-box {{
                display: none;
                padding: 16px;
                color: #ff6b6b;
                background-color: #2b1d1d;
                border: 1px solid #ff4444;
                border-radius: 6px;
                margin: 10px;
                font-family: monospace;
                font-size: 13px;
            }}
            .highcharts-tooltip span {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-size: 12px;
                line-height: 1.45;
            }}
        </style>
    </head>
    <body>
        <div id="error-box"></div>
        <div id="chart-wrapper">
            <button id="reset-zoom-btn" title="ズームをリセットして全期間表示に戻す">🔍 ズーム解除</button>
            <div id="container"></div>
        </div>

        <script>
            window.onerror = function(msg, url, line, col, error) {{
                var errDiv = document.getElementById('error-box');
                if (errDiv) {{
                    errDiv.style.display = 'block';
                    errDiv.innerText = 'Chart JS Error: ' + msg + ' (Line: ' + line + ')';
                }}
                return false;
            }};

            function initChart() {{
                if (typeof Highcharts === 'undefined' || !Highcharts.stockChart) {{
                    setTimeout(initChart, 50);
                    return;
                }}

                // 言語・ラベル設定
                Highcharts.setOptions({{
                    lang: {{
                        resetZoom: 'ズーム解除',
                        resetZoomTitle: 'ズームをリセットして全期間表示に戻す'
                    }}
                }});

                // カスタム星形マーカーシンボルの登録
                if (Highcharts.SVGRenderer && !Highcharts.SVGRenderer.prototype.symbols.star) {{
                    Highcharts.SVGRenderer.prototype.symbols.star = function (x, y, w, h) {{
                        var cx = x + w / 2,
                            cy = y + h / 2,
                            spikes = 5,
                            outerRadius = w / 2,
                            innerRadius = outerRadius * 0.45,
                            path = [],
                            step = Math.PI / spikes,
                            angle = -Math.PI / 2;

                        for (var i = 0; i < spikes * 2; i++) {{
                            var r = (i % 2 === 0) ? outerRadius : innerRadius;
                            var px = cx + Math.cos(angle) * r;
                            var py = cy + Math.sin(angle) * r;
                            if (i === 0) {{
                                path.push(['M', px, py]);
                            }} else {{
                                path.push(['L', px, py]);
                            }}
                            angle += step;
                        }}
                        path.push(['Z']);
                        return path;
                    }};
                }}

                try {{
                    const ohlcData = {ohlc_json};
                    const volumeData = {volume_json};
                    const mainSeries = {main_series_json};
                    const subPanes = {sub_panes_json};
                    const signalsSeries = {signals_series_json};
                    const btMap = {bt_map_json};
                    const showVolume = {show_volume_json};

                    // Y軸のレイアウト計算
                    const numSubPanes = subPanes.length;
                    let yAxes = [];
                    let series = [];

                    if (!showVolume) {{
                        // 【出来高非表示】価格チャートを最大化
                        if (numSubPanes === 0) {{
                            // サブ指標なし: 価格チャートが全画面 (96%)
                            yAxes.push({{
                                title: {{
                                    text: '4本値',
                                    rotation: 0,
                                    align: 'high',
                                    offset: -10,
                                    y: -10,
                                    style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                }},
                                labels: {{
                                    align: 'right',
                                    x: -6,
                                    style: {{ color: '#b0b5c0', fontSize: '11.5px', whiteSpace: 'nowrap' }},
                                    format: '{{value:,.0f}}'
                                }},
                                top: '2%',
                                height: '96%',
                                offset: 0,
                                lineWidth: 1,
                                lineColor: '#4a5061',
                                gridLineColor: '#464c5e',
                                gridLineWidth: 1,
                                opposite: true,
                                crosshair: {{
                                    enabled: true,
                                    color: '#7a808e',
                                    dashStyle: 'Dash',
                                    width: 1,
                                    snap: false,
                                    label: {{
                                        enabled: true,
                                        backgroundColor: '#2d3139',
                                        borderColor: '#4a5061',
                                        borderWidth: 1,
                                        style: {{ color: '#ffffff', fontSize: '11px', fontWeight: 'bold' }},
                                        formatter: function(val) {{
                                            const rounded = Math.round(val / 5) * 5;
                                            return rounded.toLocaleString();
                                        }}
                                    }}
                                }}
                            }});
                        }} else {{
                            // サブ指標あり: 上段65%、下段29%をサブ指標で均等割
                            const mainPct = 65;
                            const subEachPct = Math.floor(29 / numSubPanes);

                            yAxes.push({{
                                title: {{
                                    text: '4本値',
                                    rotation: 0,
                                    align: 'high',
                                    offset: -10,
                                    y: -10,
                                    style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                }},
                                labels: {{
                                    align: 'right',
                                    x: -6,
                                    style: {{ color: '#b0b5c0', fontSize: '11.5px', whiteSpace: 'nowrap' }},
                                    format: '{{value:,.0f}}'
                                }},
                                top: '2%',
                                height: mainPct + '%',
                                offset: 0,
                                lineWidth: 1,
                                lineColor: '#4a5061',
                                gridLineColor: '#464c5e',
                                gridLineWidth: 1,
                                opposite: true,
                                crosshair: {{
                                    enabled: true,
                                    color: '#7a808e',
                                    dashStyle: 'Dash',
                                    width: 1,
                                    snap: false,
                                    label: {{
                                        enabled: true,
                                        backgroundColor: '#2d3139',
                                        borderColor: '#4a5061',
                                        borderWidth: 1,
                                        style: {{ color: '#ffffff', fontSize: '11px', fontWeight: 'bold' }},
                                        formatter: function(val) {{
                                            const rounded = Math.round(val / 5) * 5;
                                            return rounded.toLocaleString();
                                        }}
                                    }}
                                }}
                            }});

                            let curTop = mainPct + 4;
                            subPanes.forEach((sp, idx) => {{
                                const yAxisIdx = 1 + idx;
                                const plotLines = (sp.baselines || []).map(b => ({{
                                    value: b,
                                    color: '#555e70',
                                    width: 1,
                                    dashStyle: 'ShortDash'
                                }}));

                                yAxes.push({{
                                    title: {{
                                        text: sp.title,
                                        rotation: 0,
                                        align: 'high',
                                        offset: -10,
                                        y: -10,
                                        style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                    }},
                                    labels: {{
                                        align: 'right',
                                        x: -6,
                                        style: {{ color: '#b0b5c0', fontSize: '11px', whiteSpace: 'nowrap' }}
                                    }},
                                    top: curTop + '%',
                                    height: subEachPct + '%',
                                    offset: 0,
                                    lineWidth: 1,
                                    lineColor: '#4a5061',
                                    gridLineColor: '#464c5e',
                                    gridLineWidth: 1,
                                    plotLines: plotLines,
                                    opposite: true,
                                    crosshair: {{
                                        enabled: true,
                                        color: '#7a808e',
                                        dashStyle: 'Dash',
                                        width: 1,
                                        snap: false,
                                        label: {{
                                            enabled: true,
                                            backgroundColor: '#2d3139',
                                            borderColor: '#4a5061',
                                            borderWidth: 1,
                                            style: {{ color: '#ffffff', fontSize: '10.5px' }},
                                            formatter: function(val) {{
                                                return val.toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
                                            }}
                                        }}
                                    }}
                                }});

                                sp.series.forEach(s => {{
                                    series.push({{
                                        type: s.type || 'line',
                                        name: s.name,
                                        data: s.data,
                                        color: s.color,
                                        lineWidth: s.lineWidth || 1.5,
                                        yAxis: yAxisIdx
                                    }});
                                }});

                                curTop += subEachPct + 2;
                            }});
                        }}
                    }} else {{
                        // 【出来高表示あり】
                        if (numSubPanes === 0) {{
                            // 上段: ローソク足 (68%), 下段: 出来高 (24%)
                            yAxes = [
                                {{
                                    title: {{
                                        text: '4本値',
                                        rotation: 0,
                                        align: 'high',
                                        offset: -10,
                                        y: -10,
                                        style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                    }},
                                    labels: {{
                                        align: 'right',
                                        x: -6,
                                        style: {{ color: '#b0b5c0', fontSize: '11.5px', whiteSpace: 'nowrap' }},
                                        format: '{{value:,.0f}}'
                                    }},
                                    top: '2%',
                                    height: '68%',
                                    offset: 0,
                                    lineWidth: 1,
                                    lineColor: '#4a5061',
                                    gridLineColor: '#464c5e',
                                    gridLineWidth: 1,
                                    opposite: true,
                                    crosshair: {{
                                        enabled: true,
                                        color: '#7a808e',
                                        dashStyle: 'Dash',
                                        width: 1,
                                        snap: false,
                                        label: {{
                                            enabled: true,
                                            backgroundColor: '#2d3139',
                                            borderColor: '#4a5061',
                                            borderWidth: 1,
                                            style: {{ color: '#ffffff', fontSize: '11px', fontWeight: 'bold' }},
                                            formatter: function(val) {{
                                                const rounded = Math.round(val / 5) * 5;
                                                return rounded.toLocaleString();
                                            }}
                                        }}
                                    }}
                                }},
                                {{
                                    title: {{
                                        text: '出来高',
                                        rotation: 0,
                                        align: 'high',
                                        offset: -10,
                                        y: -10,
                                        style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                    }},
                                    labels: {{
                                        align: 'right',
                                        x: -6,
                                        style: {{ color: '#b0b5c0', fontSize: '11px', whiteSpace: 'nowrap' }},
                                        format: '{{value:,.0f}}'
                                    }},
                                    top: '73%',
                                    height: '24%',
                                    offset: 0,
                                    lineWidth: 1,
                                    lineColor: '#4a5061',
                                    gridLineColor: '#464c5e',
                                    gridLineWidth: 1,
                                    opposite: true,
                                    crosshair: {{
                                        enabled: true,
                                        color: '#7a808e',
                                        dashStyle: 'Dash',
                                        width: 1,
                                        snap: false,
                                        label: {{
                                            enabled: true,
                                            backgroundColor: '#2d3139',
                                            borderColor: '#4a5061',
                                            borderWidth: 1,
                                            style: {{ color: '#ffffff', fontSize: '10.5px' }},
                                            format: '{{value:,.0f}}'
                                        }}
                                    }}
                                }}
                            ];
                        }} else {{
                            // サブ指標がある場合の動的配分
                            const mainPct = 50;
                            const volPct = 18;
                            const subEachPct = Math.floor(28 / numSubPanes);

                            yAxes.push({{
                                title: {{
                                    text: '4本値',
                                    rotation: 0,
                                    align: 'high',
                                    offset: -10,
                                    y: -10,
                                    style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                }},
                                labels: {{
                                    align: 'right',
                                    x: -6,
                                    style: {{ color: '#b0b5c0', fontSize: '11.5px', whiteSpace: 'nowrap' }},
                                    format: '{{value:,.0f}}'
                                }},
                                top: '2%',
                                height: mainPct + '%',
                                offset: 0,
                                lineWidth: 1,
                                lineColor: '#4a5061',
                                gridLineColor: '#464c5e',
                                gridLineWidth: 1,
                                opposite: true,
                                crosshair: {{
                                    enabled: true,
                                    color: '#7a808e',
                                    dashStyle: 'Dash',
                                    width: 1,
                                    snap: false,
                                    label: {{
                                        enabled: true,
                                        backgroundColor: '#2d3139',
                                        borderColor: '#4a5061',
                                        borderWidth: 1,
                                        style: {{ color: '#ffffff', fontSize: '11px', fontWeight: 'bold' }},
                                        formatter: function(val) {{
                                            const rounded = Math.round(val / 5) * 5;
                                            return rounded.toLocaleString();
                                        }}
                                    }}
                                }}
                            }});

                            yAxes.push({{
                                title: {{
                                    text: '出来高',
                                    rotation: 0,
                                    align: 'high',
                                    offset: -10,
                                    y: -10,
                                    style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                }},
                                labels: {{
                                    align: 'right',
                                    x: -6,
                                    style: {{ color: '#b0b5c0', fontSize: '11px', whiteSpace: 'nowrap' }},
                                    format: '{{value:,.0f}}'
                                }},
                                top: (mainPct + 3) + '%',
                                height: volPct + '%',
                                offset: 0,
                                lineWidth: 1,
                                lineColor: '#4a5061',
                                gridLineColor: '#464c5e',
                                gridLineWidth: 1,
                                opposite: true,
                                crosshair: {{
                                    enabled: true,
                                    color: '#7a808e',
                                    dashStyle: 'Dash',
                                    width: 1,
                                    snap: false,
                                    label: {{
                                        enabled: true,
                                        backgroundColor: '#2d3139',
                                        borderColor: '#4a5061',
                                        borderWidth: 1,
                                        style: {{ color: '#ffffff', fontSize: '10.5px' }},
                                        format: '{{value:,.0f}}'
                                    }}
                                }}
                            }});

                            let curTop = mainPct + volPct + 5;
                            subPanes.forEach((sp, idx) => {{
                                const yAxisIdx = 2 + idx;
                                const plotLines = (sp.baselines || []).map(b => ({{
                                    value: b,
                                    color: '#555e70',
                                    width: 1,
                                    dashStyle: 'ShortDash'
                                }}));

                                yAxes.push({{
                                    title: {{
                                        text: sp.title,
                                        rotation: 0,
                                        align: 'high',
                                        offset: -10,
                                        y: -10,
                                        style: {{ color: '#9aa0a6', fontSize: '11px', writingMode: 'vertical-rl' }}
                                    }},
                                    labels: {{
                                        align: 'right',
                                        x: -6,
                                        style: {{ color: '#b0b5c0', fontSize: '11px', whiteSpace: 'nowrap' }}
                                    }},
                                    top: curTop + '%',
                                    height: subEachPct + '%',
                                    offset: 0,
                                    lineWidth: 1,
                                    lineColor: '#4a5061',
                                    gridLineColor: '#464c5e',
                                    gridLineWidth: 1,
                                    plotLines: plotLines,
                                    opposite: true,
                                    crosshair: {{
                                        enabled: true,
                                        color: '#7a808e',
                                        dashStyle: 'Dash',
                                        width: 1,
                                        snap: false,
                                        label: {{
                                            enabled: true,
                                            backgroundColor: '#2d3139',
                                            borderColor: '#4a5061',
                                            borderWidth: 1,
                                            style: {{ color: '#ffffff', fontSize: '10.5px' }},
                                            formatter: function(val) {{
                                                return val.toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
                                            }}
                                        }}
                                    }}
                                }});

                                sp.series.forEach(s => {{
                                    series.push({{
                                        type: s.type || 'line',
                                        name: s.name,
                                        data: s.data,
                                        color: s.color,
                                        lineWidth: s.lineWidth || 1.5,
                                        yAxis: yAxisIdx
                                    }});
                                }});

                                curTop += subEachPct + 2;
                            }});
                        }}
                    }}

                    // ローソク足シリーズ
                    series.unshift({{
                        type: 'candlestick',
                        name: '4本値',
                        data: ohlcData,
                        yAxis: 0,
                        color: '#8b3a3a',
                        lineColor: '#8b3a3a',
                        upColor: '#4a7bb0',
                        upLineColor: '#4a7bb0',
                        id: 'candlestick-series',
                        dataGrouping: {{
                            enabled: false
                        }}
                    }});

                    // 出来高シリーズ (表示時のみ追加)
                    if (showVolume) {{
                        series.push({{
                            type: 'column',
                            name: '出来高',
                            data: volumeData,
                            yAxis: 1,
                            dataGrouping: {{
                                enabled: false
                            }}
                        }});
                    }}

                    // メイン指標シリーズを追加
                    mainSeries.forEach(ms => {{
                        series.push({{
                            type: 'line',
                            name: ms.name,
                            data: ms.data,
                            color: ms.color,
                            lineWidth: ms.lineWidth || 1.5,
                            dashStyle: ms.dashStyle || 'Solid',
                            yAxis: 0,
                            dataGrouping: {{
                                enabled: false
                            }}
                        }});
                    }});

                    // 売買シグナルシリーズ（エントリー・ナンピン・決済）を追加
                    if (signalsSeries && signalsSeries.length > 0) {{
                        signalsSeries.forEach(ss => {{
                            series.push({{
                                type: 'scatter',
                                name: ss.name,
                                data: ss.data,
                                color: ss.color,
                                marker: ss.marker,
                                yAxis: 0,
                                zIndex: ss.zIndex || 10,
                                tooltip: ss.tooltip,
                                dataGrouping: {{
                                    enabled: false
                                }}
                            }});
                        }});
                    }}

                    // Highcharts Stock チャートの生成
                    const chart = Highcharts.stockChart('container', {{
                        time: {{
                            useUTC: true
                        }},
                        chart: {{
                            height: {chart_height},
                            reflow: true,
                            backgroundColor: '#1e1e1e',
                            plotBackgroundColor: '#1e1e1e',
                            zooming: {{
                                type: 'x',
                                mouseWheel: {{
                                    enabled: false
                                }},
                                resetButton: {{
                                    position: {{
                                        align: 'right',
                                        verticalAlign: 'top',
                                        x: -95,
                                        y: 8
                                    }},
                                    theme: {{
                                        fill: '#2a2e39',
                                        stroke: '#4a5061',
                                        r: 4,
                                        style: {{
                                            color: '#d1d4dc',
                                            fontSize: '11px',
                                            fontWeight: 'bold'
                                        }},
                                        states: {{
                                            hover: {{
                                                fill: '#3d4454',
                                                style: {{
                                                    color: '#ffffff'
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }},
                            panning: {{
                                enabled: false
                            }},
                            style: {{
                                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                            }},
                            marginRight: 90,
                            marginLeft: 15,
                            spacingTop: 10,
                            spacingBottom: 5
                        }},

                        credits: {{
                            enabled: false
                        }},

                        exporting: {{
                            enabled: false
                        }},

                        rangeSelector: {{
                            enabled: false
                        }},

                        legend: {{
                            enabled: true,
                            layout: 'horizontal',
                            align: 'center',
                            verticalAlign: 'bottom',
                            y: 0,
                            itemStyle: {{
                                color: '#9aa0a6',
                                fontSize: '12px',
                                fontWeight: 'normal'
                            }},
                            itemHoverStyle: {{
                                color: '#ffffff'
                            }},
                            symbolRadius: 6
                        }},

                        plotOptions: {{
                            series: {{
                                animation: false,
                                dataGrouping: {{
                                    enabled: false
                                }},
                                states: {{
                                    hover: {{
                                        lineWidthPlus: 0
                                    }}
                                }}
                            }},
                            candlestick: {{
                                pointPadding: 0.1,
                                groupPadding: 0.1
                            }},
                            column: {{
                                borderWidth: 0,
                                pointPadding: 0.1,
                                groupPadding: 0.1
                            }}
                        }},

                        tooltip: {{
                            split: false,
                            shared: true,
                            useHTML: true,
                            backgroundColor: 'rgba(28, 28, 30, 0.95)',
                            borderColor: '#555a64',
                            borderRadius: 6,
                            borderWidth: 1,
                            shadow: true,
                            distance: 20,
                            padding: 10,
                            style: {{
                                color: '#e0e0e0',
                                fontSize: '12px'
                            }},
                            formatter: function() {{
                                const date = new Date(this.x);
                                const y = date.getUTCFullYear();
                                const m = ('0' + (date.getUTCMonth() + 1)).slice(-2);
                                const d = ('0' + date.getUTCDate()).slice(-2);
                                const h = ('0' + date.getUTCHours()).slice(-2);
                                const min = ('0' + date.getUTCMinutes()).slice(-2);
                                const dateStr = `${{y}}/${{m}}/${{d}} ${{h}}:${{min}}`;

                                let s = `<div style="font-weight: bold; margin-bottom: 4px; color: #ffffff;">${{dateStr}}</div>`;

                                let ohlcPoint = null;
                                let volPoint = null;
                                let otherPoints = [];

                                if (this.points) {{
                                    this.points.forEach(p => {{
                                        if (p.series.name === '4本値') {{
                                            ohlcPoint = p.point;
                                        }} else if (p.series.name === '出来高') {{
                                            volPoint = p.point;
                                        }} else {{
                                            otherPoints.push(p);
                                        }}
                                    }});
                                }}

                                if (ohlcPoint) {{
                                    const openStr = (ohlcPoint.open != null && !isNaN(ohlcPoint.open)) ? ohlcPoint.open.toLocaleString() : '-';
                                    const highStr = (ohlcPoint.high != null && !isNaN(ohlcPoint.high)) ? ohlcPoint.high.toLocaleString() : '-';
                                    const lowStr = (ohlcPoint.low != null && !isNaN(ohlcPoint.low)) ? ohlcPoint.low.toLocaleString() : '-';
                                    const closeStr = (ohlcPoint.close != null && !isNaN(ohlcPoint.close)) ? ohlcPoint.close.toLocaleString() : '-';
                                    
                                    s += `<div style="display: grid; grid-template-columns: auto auto; column-gap: 8px; font-size: 11.5px; line-height: 1.5;">
                                            <span style="color: #9aa0a6;">始値 :</span><span style="text-align: right; font-weight: 500;">${{openStr}}</span>
                                            <span style="color: #9aa0a6;">高値 :</span><span style="text-align: right; font-weight: 500;">${{highStr}}</span>
                                            <span style="color: #9aa0a6;">安値 :</span><span style="text-align: right; font-weight: 500;">${{lowStr}}</span>
                                            <span style="color: #9aa0a6;">終値 :</span><span style="text-align: right; font-weight: 500;">${{closeStr}}</span>`;
                                    
                                    if (volPoint && showVolume) {{
                                        const volStr = (volPoint.y != null && !isNaN(volPoint.y)) ? volPoint.y.toLocaleString() : '0';
                                        s += `<span style="color: #9aa0a6;">出来高 :</span><span style="text-align: right; font-weight: 500;">${{volStr}}</span>`;
                                    }}
                                    s += `</div>`;
                                }} else if (volPoint && showVolume) {{
                                    const volStr = (volPoint.y != null && !isNaN(volPoint.y)) ? volPoint.y.toLocaleString() : '0';
                                    s += `<div><span style="color: #9aa0a6;">出来高 :</span> <b>${{volStr}}</b></div>`;
                                }}

                                // ★ マウスホバー連動: バックテスト情報の表示
                                const btInfo = btMap ? btMap[this.x] : null;
                                if (btInfo) {{
                                    s += `<div style="margin-top: 6px; padding-top: 5px; border-top: 1px solid #444a55; font-size: 11px;">`;
                                    if (btInfo.lots > 0) {{
                                        const pnlCol = btInfo.unrealized >= 0 ? '#00e676' : '#ff1744';
                                        s += `<div><span style="color: #ffd600;">●</span> 保有: <b>${{btInfo.pos}}</b></div>`;
                                        s += `<div><span style="color: #9aa0a6;">└ 含み損益:</span> <b style="color: ${{pnlCol}};">${{btInfo.pnl}}</b></div>`;
                                    }} else {{
                                        s += `<div><span style="color: #9aa0a6;">● ポジション:</span> <b>ノーポジ</b></div>`;
                                    }}
                                    if (btInfo.signals && btInfo.signals.length > 0) {{
                                        s += `<div style="margin-top: 4px; padding-top: 3px; border-top: 1px dashed #555a64;">`;
                                        btInfo.signals.forEach(sig => {{
                                            s += `<div style="color: #00e5ff; font-weight: bold; margin-top: 2px;">⚡ ${{sig}}</div>`;
                                        }});
                                        s += `</div>`;
                                    }}
                                    s += `</div>`;
                                }}

                                if (otherPoints.length > 0) {{
                                    s += `<div style="margin-top: 5px; padding-top: 4px; border-top: 1px dashed #444a55; font-size: 11px;">`;
                                    otherPoints.forEach(p => {{
                                        const valStr = (p.y != null && !isNaN(p.y)) ? p.y.toLocaleString(undefined, {{ maximumFractionDigits: 2 }}) : '-';
                                        s += `<div><span style="color: ${{p.series.color}};">●</span> ${{p.series.name}}: <b>${{valStr}}</b></div>`;
                                    }});
                                    s += `</div>`;
                                }}

                                return s;
                            }}
                        }},

                        xAxis: {{
                            type: 'datetime',
                            lineColor: '#363a45',
                            lineWidth: 1,
                            gridLineColor: '#2b2f3a',
                            gridLineWidth: 1,
                            labels: {{
                                style: {{ color: '#8e9297', fontSize: '11px' }},
                                formatter: function() {{
                                    const d = new Date(this.value);
                                    const m = ('0' + (d.getUTCMonth() + 1)).slice(-2);
                                    const day = ('0' + d.getUTCDate()).slice(-2);
                                    const h = ('0' + d.getUTCHours()).slice(-2);
                                    const min = ('0' + d.getUTCMinutes()).slice(-2);
                                    if (h === '00' && min === '00') {{
                                        return `${{m}}/${{day}}`;
                                    }}
                                    return `${{m}}/${{day}}<br/>${{h}}:${{min}}`;
                                }}
                            }},
                            crosshair: {{
                                enabled: true,
                                color: '#666a75',
                                dashStyle: 'Solid',
                                width: 1,
                                label: {{
                                    enabled: true,
                                    backgroundColor: '#2d3139',
                                    style: {{ color: '#ffffff' }}
                                }}
                            }}
                        }},

                        yAxis: yAxes,
                        series: series,

                        navigator: {{
                            enabled: true,
                            height: 38,
                            margin: 12,
                            maskFill: 'rgba(255, 255, 255, 0.08)',
                            outlineColor: '#3a3e4b',
                            outlineWidth: 1,
                            handles: {{
                                backgroundColor: '#8a909d',
                                borderColor: '#3a3e4b'
                            }},
                            xAxis: {{
                                gridLineColor: '#2b2f3a',
                                labels: {{
                                    style: {{ color: '#7a808c', fontSize: '10px' }}
                                }}
                            }},
                            series: {{
                                type: 'line',
                                color: '#4a7bb0',
                                lineWidth: 1,
                                fillOpacity: 0.1,
                                dataGrouping: {{
                                    enabled: true
                                }}
                            }}
                        }},

                        scrollbar: {{
                            enabled: true,
                            height: 12,
                            barBackgroundColor: '#3a3e4b',
                            barBorderColor: '#3a3e4b',
                            buttonBackgroundColor: '#2a2e39',
                            buttonBorderColor: '#3a3e4b',
                            buttonArrowColor: '#8a909d',
                            trackBackgroundColor: '#1a1c23',
                            trackBorderColor: '#2a2e39'
                        }}
                    }});

                    // ズーム解除ボタンのイベント連携
                    const resetBtn = document.getElementById('reset-zoom-btn');
                    if (resetBtn && chart && chart.xAxis && chart.xAxis[0]) {{
                        resetBtn.onclick = function() {{
                            const ext = chart.xAxis[0].getExtremes();
                            if (ext && ext.dataMin !== undefined && ext.dataMax !== undefined) {{
                                chart.xAxis[0].setExtremes(ext.dataMin, ext.dataMax, true, false);
                                resetBtn.style.display = 'none';
                            }}
                        }};

                        Highcharts.addEvent(chart.xAxis[0], 'afterSetExtremes', function(e) {{
                            const ext = chart.xAxis[0].getExtremes();
                            if (ext && ext.dataMin !== undefined && ext.dataMax !== undefined) {{
                                // ズーム状態（全期間より表示範囲が狭い）であればボタンを表示
                                const isZoomed = (ext.min > ext.dataMin + 1000 || ext.max < ext.dataMax - 1000);
                                resetBtn.style.display = isZoomed ? 'block' : 'none';
                            }}
                        }});
                    }}
                }} catch(err) {{
                    var errDiv = document.getElementById('error-box');
                    if (errDiv) {{
                        errDiv.style.display = 'block';
                        errDiv.innerText = 'Highcharts Init Error: ' + err.message;
                    }}
                    console.error(err);
                }}
            }}

            // 確実な初期化トリガー
            initChart();
            if (document.readyState !== 'complete') {{
                window.addEventListener('load', function() {{
                    setTimeout(initChart, 20);
                }});
            }}
        </script>
    </body>
    </html>
    """

    total_component_height = chart_height + 20
    components.html(html_code, height=total_component_height, scrolling=False)

def render_lightweight_chart(*args, **kwargs):
    return render_highstock_chart(*args, **kwargs)
