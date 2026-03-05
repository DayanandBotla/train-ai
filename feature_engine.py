"""
feature_engine.py — NiftyEdge Pro v2
Calculates all AI features from OHLCV candle data.
Features: VWAP, EMA20, EMA50, RSI, ATR, Volume Ratio, Price Position
"""
import math

def calc_ema(closes, period):
    """Exponential Moving Average."""
    if len(closes) < period:
        return closes[-1] if closes else 0
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return round(ema, 2)

def calc_rsi(closes, period=14):
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calc_atr(highs, lows, closes, period=14):
    """Average True Range — volatility measure."""
    if len(highs) < 2:
        return 0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1])
        )
        trs.append(tr)
    return round(sum(trs[-period:]) / min(len(trs), period), 2)

def calc_vwap(highs, lows, closes, volumes):
    """Volume Weighted Average Price."""
    if not volumes or sum(volumes) == 0:
        return closes[-1] if closes else 0
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    num = sum(t * v for t, v in zip(typical, volumes))
    den = sum(volumes)
    return round(num / den, 2) if den else closes[-1]

def calc_vol_ratio(volumes, period=20):
    """Current volume vs N-period average."""
    if len(volumes) < 2:
        return 1.0
    avg = sum(volumes[-period-1:-1]) / min(len(volumes)-1, period)
    return round(volumes[-1] / avg, 3) if avg > 0 else 1.0

def calc_momentum(closes, period=5):
    """Price momentum — rate of change."""
    if len(closes) <= period:
        return 0
    return round((closes[-1] - closes[-period-1]) / closes[-period-1] * 100, 3)

def calc_bb_position(closes, period=20):
    """
    Bollinger Band position — where is price within the bands?
    Returns 0–1 where 0=lower band, 0.5=middle, 1=upper band
    """
    if len(closes) < period:
        return 0.5
    sma = sum(closes[-period:]) / period
    std = math.sqrt(sum((c - sma)**2 for c in closes[-period:]) / period)
    if std == 0:
        return 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    pos = (closes[-1] - lower) / (upper - lower)
    return round(max(0, min(1, pos)), 3)

def build_features(candles):
    """
    Build feature vector from list of candle dicts.
    Each candle: {t, o, h, l, c, v}
    Returns dict of all features + numpy-ready list.
    """
    if len(candles) < 20:
        return None

    closes  = [c["c"] for c in candles]
    highs   = [c["h"] for c in candles]
    lows    = [c["l"] for c in candles]
    volumes = [c.get("v", 0) for c in candles]
    opens   = [c["o"] for c in candles]

    ema20   = calc_ema(closes, 20)
    ema50   = calc_ema(closes, 50) if len(closes) >= 50 else ema20
    rsi     = calc_rsi(closes)
    atr     = calc_atr(highs, lows, closes)
    vwap    = calc_vwap(highs, lows, closes, volumes)
    vol_r   = calc_vol_ratio(volumes)
    mom5    = calc_momentum(closes, 5)
    mom10   = calc_momentum(closes, 10)
    bb_pos  = calc_bb_position(closes)

    price   = closes[-1]
    atr_pct = round(atr / price * 100, 3) if price else 0

    features = {
        # Price vs indicators
        "price_vs_vwap":    round((price - vwap) / vwap * 100, 3),
        "price_vs_ema20":   round((price - ema20) / ema20 * 100, 3),
        "price_vs_ema50":   round((price - ema50) / ema50 * 100, 3),
        "ema20_vs_ema50":   round((ema20 - ema50) / ema50 * 100, 3),

        # Momentum
        "rsi":              rsi,
        "momentum_5":       mom5,
        "momentum_10":      mom10,

        # Volatility
        "atr_pct":          atr_pct,
        "bb_position":      bb_pos,

        # Volume
        "volume_ratio":     vol_r,

        # Candle structure
        "candle_body":      round((closes[-1] - opens[-1]) / opens[-1] * 100, 3),
        "candle_upper_wick":round((highs[-1] - max(opens[-1], closes[-1])) / opens[-1] * 100, 3),
        "candle_lower_wick":round((min(opens[-1], closes[-1]) - lows[-1]) / opens[-1] * 100, 3),

        # Raw values for display
        "_price": price, "_ema20": ema20, "_ema50": ema50,
        "_vwap": vwap, "_rsi": rsi, "_atr": atr, "_vol_ratio": vol_r,
    }

    # Feature vector (no raw _ values) for model input
    feature_cols = [k for k in features if not k.startswith("_")]
    vector = [features[k] for k in feature_cols]

    return {"features": features, "vector": vector, "cols": feature_cols}

def label_candles(candles, forward=3, min_move_pct=0.3):
    """
    Label each candle for training:
    1 = price went up min_move_pct% in next `forward` candles
    0 = price went down or stayed flat
    -1 = not enough future data (exclude from training)
    """
    labels = []
    closes = [c["c"] for c in candles]
    for i in range(len(closes)):
        if i + forward >= len(closes):
            labels.append(-1)
            continue
        future_high = max(closes[i+1:i+forward+1])
        future_low  = min(closes[i+1:i+forward+1])
        up_move   = (future_high - closes[i]) / closes[i] * 100
        down_move = (closes[i] - future_low)  / closes[i] * 100
        if up_move >= min_move_pct:
            labels.append(1)   # bullish
        elif down_move >= min_move_pct:
            labels.append(0)   # bearish
        else:
            labels.append(-1)  # flat — exclude
    return labels
