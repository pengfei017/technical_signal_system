from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)
    for period in [5, 10, 20, 60]:
        out[f"ma{period}"] = grouped["adj_close"].transform(lambda s: s.rolling(period, min_periods=period).mean())
    out["vol_ma5"] = grouped["vol"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    out["vol_ma20"] = grouped["vol"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["prev_vol_ma5"] = grouped["vol"].transform(lambda s: s.rolling(5, min_periods=5).mean().shift(1))
    out["prev_vol_ma20"] = grouped["vol"].transform(lambda s: s.rolling(20, min_periods=20).mean().shift(1))
    out["volume_ratio_5"] = out["vol"] / out["prev_vol_ma5"].where(out["prev_vol_ma5"] > 0)
    out["volume_ratio_20"] = out["vol"] / out["prev_vol_ma20"].where(out["prev_vol_ma20"] > 0)
    out["high20"] = grouped["adj_high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    out["low20"] = grouped["adj_low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    out["bias5"] = (out["adj_close"] / out["ma5"] - 1.0) * 100.0

    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100.0 - (100.0 / (1.0 + rs))

    out["rsi14"] = grouped["adj_close"].transform(_rsi)

    def _macd(series: pd.Series) -> pd.DataFrame:
        ema12 = series.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = series.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        hist = macd - signal
        return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist}, index=series.index)

    macd_frames = grouped["adj_close"].apply(_macd).reset_index(level=0, drop=True)
    out[["macd", "macd_signal", "macd_hist"]] = macd_frames[["macd", "macd_signal", "macd_hist"]]
    return out
