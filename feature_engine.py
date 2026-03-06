"""
feature_engine.py — Train-AI
Pure Python. Zero external dependencies.
Calculates: VWAP, EMA20, EMA50, RSI, ATR, Volume Ratio, BB Position, Momentum
"""
import math

def ema(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    k, v = 2.0/(period+1), sum(closes[:period])/period
    for c in closes[period:]: v = c*k + v*(1-k)
    return round(v, 2)

def rsi(closes, period=14):
    if len(closes) < period+1: return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    ag = sum(gains[-period:])/period
    al = sum(losses[-period:])/period
    if al == 0: return 100.0
    return round(100 - 100/(1+ag/al), 2)

def atr(highs, lows, closes, period=14):
    if len(highs) < 2: return 0.0
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(highs))]
    return round(sum(trs[-period:])/min(len(trs),period), 2)

def vwap(highs, lows, closes, volumes):
    if not volumes or sum(volumes)==0: return closes[-1] if closes else 0.0
    tp  = [(h+l+c)/3 for h,l,c in zip(highs,lows,closes)]
    num = sum(t*v for t,v in zip(tp,volumes))
    den = sum(volumes)
    return round(num/den, 2) if den else closes[-1]

def vol_ratio(volumes, period=20):
    if len(volumes) < 2: return 1.0
    avg = sum(volumes[-period-1:-1]) / min(len(volumes)-1, period)
    return round(volumes[-1]/avg, 3) if avg > 0 else 1.0

def momentum(closes, period=5):
    if len(closes) <= period: return 0.0
    return round((closes[-1]-closes[-period-1])/closes[-period-1]*100, 3)

def bb_position(closes, period=20):
    if len(closes) < period: return 0.5
    sma = sum(closes[-period:])/period
    std = math.sqrt(sum((c-sma)**2 for c in closes[-period:])/period)
    if std == 0: return 0.5
    pos = (closes[-1] - (sma-2*std)) / (4*std)
    return round(max(0.0, min(1.0, pos)), 3)

def build_features(candles):
    if len(candles) < 20: return None
    closes  = [c["c"] for c in candles]
    highs   = [c["h"] for c in candles]
    lows    = [c["l"] for c in candles]
    vols    = [c.get("v",0) for c in candles]
    opens   = [c["o"] for c in candles]

    e20  = ema(closes, 20)
    e50  = ema(closes, 50) if len(closes)>=50 else e20
    r    = rsi(closes)
    at   = atr(highs, lows, closes)
    vw   = vwap(highs, lows, closes, vols)
    vr   = vol_ratio(vols)
    m5   = momentum(closes, 5)
    m10  = momentum(closes, 10)
    bb   = bb_position(closes)
    p    = closes[-1]
    body = (closes[-1]-opens[-1])/opens[-1]*100 if opens[-1] else 0

    feat = {
        "price_vs_vwap":  round((p-vw)/vw*100, 3)   if vw  else 0,
        "price_vs_ema20": round((p-e20)/e20*100, 3)  if e20 else 0,
        "price_vs_ema50": round((p-e50)/e50*100, 3)  if e50 else 0,
        "ema20_vs_ema50": round((e20-e50)/e50*100, 3) if e50 else 0,
        "rsi":            r,
        "momentum_5":     m5,
        "momentum_10":    m10,
        "atr_pct":        round(at/p*100, 3) if p else 0,
        "bb_position":    bb,
        "volume_ratio":   vr,
        "candle_body":    round(body, 3),
        # raw for display
        "_price":p, "_ema20":e20, "_ema50":e50,
        "_vwap":vw, "_rsi":r, "_atr":at, "_vol_ratio":vr,
    }
    cols   = [k for k in feat if not k.startswith("_")]
    vector = [feat[k] for k in cols]
    return {"features": feat, "vector": vector, "cols": cols}

def label_candles(candles, forward=3, min_move=0.25):
    closes = [c["c"] for c in candles]
    labels = []
    for i in range(len(closes)):
        if i+forward >= len(closes):
            labels.append(-1); continue
        fh = max(closes[i+1:i+forward+1])
        fl = min(closes[i+1:i+forward+1])
        up   = (fh-closes[i])/closes[i]*100
        down = (closes[i]-fl)/closes[i]*100
        if   up   >= min_move: labels.append(1)
        elif down >= min_move: labels.append(0)
        else:                  labels.append(-1)
    return labels
