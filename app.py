"""
Train-AI — Intelligent Options Trading System
Strategy 1: ORB Liquidity Sweep (8 filters)
Strategy 2: AI XGBoost Confirmation (65% threshold)
Logic: ORB signal fires FIRST → AI confirms → only then execute
Client ID: 1108455416
"""
import os, json, time, threading, struct, requests
from datetime import datetime, time as dtime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from ai_model import ai, generate_sim_candles
from feature_engine import build_features

try:
    import websocket
    WS_OK = True
except ImportError:
    WS_OK = False

app = Flask(__name__)
CORS(app)

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
CONFIG = {
    "CLIENT_ID":      "1108455416",
    "TOKEN":          os.environ.get("DHAN_TOKEN",""),
    "MODE":           os.environ.get("TRADE_MODE","paper"),
    "INDEX":          os.environ.get("INDEX","NIFTY"),
    "LOTS":           int(os.environ.get("LOTS","1")),
    "CAPITAL":        float(os.environ.get("CAPITAL","50000")),
    "RISK_PCT":       float(os.environ.get("RISK_PCT","1")),
    "SL_PCT":         float(os.environ.get("SL_PCT","25")),
    "TGT_PCT":        float(os.environ.get("TGT_PCT","70")),
    "OR_MINUTES":     int(os.environ.get("OR_MINUTES","15")),
    "GAP_MAX_PCT":    float(os.environ.get("GAP_MAX_PCT","0.5")),
    "VIX_MIN":        float(os.environ.get("VIX_MIN","10")),
    "VIX_MAX":        float(os.environ.get("VIX_MAX","22")),
    "TRAIL_EVERY":    float(os.environ.get("TRAIL_EVERY","500")),
    "AI_ENABLED":     os.environ.get("AI_ENABLED","true").lower()=="true",
    "AI_THRESHOLD":   float(os.environ.get("AI_THRESHOLD","0.65")),
    "AI_HIGH_CONF":   float(os.environ.get("AI_HIGH_CONF","0.72")),
    "AI_RETRAIN_HOUR":int(os.environ.get("AI_RETRAIN_HOUR","8")),
}

SECID    = {"NIFTY":"13","BANKNIFTY":"25","FINNIFTY":"27","VIX":"1333"}
LOT      = {"NIFTY":50,"BANKNIFTY":15,"FINNIFTY":40}
DHAN_API = "https://api.dhan.co"
DHAN_WS  = "wss://api-feed.dhan.co"

# ════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════
STATE = {
    "ws_connected":False,"token_valid":False,"last_tick":None,
    "nifty":0.0,"nifty_prev":0.0,"nifty_open":0.0,
    "banknifty":0.0,"vix":0.0,"ema15":0.0,"candles_15m":[],
    "strategy_on":False,"or_high":None,"or_low":None,
    "or_fetched":False,"sweep_dir":None,"sweep_candles":0,"orb_signal":None,
    "ai_signal":None,"ai_prob":0.5,"ai_prob_down":0.5,
    "ai_confidence":"LOW","ai_reason":"","ai_trained":False,
    "ai_training":False,"ai_last_trained":None,"ai_metrics":{},
    "ai_prediction_history":[],
    "signal":None,"signal_score":0.0,
    "position":None,"entry_price":0.0,
    "sl_price":0.0,"target_price":0.0,"trail_high":0.0,
    "trades":[],"alerts":[],
    "gross_win":0.0,"gross_loss":0.0,"daily_loss":0.0,
    "signals_count":0,"blocked_count":0,"skipped_count":0,
    "ai_blocked":0,"ai_confirmed":0,
    # 5-min OR engine
    "or5_high":None,"or5_low":None,"or5_fetched":False,
    "or5_sweep_dir":None,"or5_sweep_count":0,
    "or5_signals":0,"or5_blocked":0,
}

# ════════════════════════════════════════════
# DHAN REST
# ════════════════════════════════════════════
def hdrs():
    return {"access-token":CONFIG["TOKEN"],"client-id":CONFIG["CLIENT_ID"],
            "Content-Type":"application/json","Accept":"application/json"}

def validate_token():
    if not CONFIG["TOKEN"]:
        STATE["token_valid"]=False
        add_alert("warn","No DHAN_TOKEN — simulation mode. Add token to Railway.")
        return False
    try:
        r=requests.get(f"{DHAN_API}/v2/fundlimit",headers=hdrs(),timeout=5)
        STATE["token_valid"]=r.status_code==200
        if STATE["token_valid"]: add_alert("success",f"Dhan connected. Client:{CONFIG['CLIENT_ID']}")
        else: add_alert("danger",f"Token invalid ({r.status_code}). Update DHAN_TOKEN.")
        return STATE["token_valid"]
    except Exception as e:
        STATE["token_valid"]=False
        add_alert("danger",f"Token check: {e}")
        return False

def fetch_prev_close():
    try:
        today=datetime.now().strftime("%Y-%m-%d")
        r=requests.post(f"{DHAN_API}/v2/charts/eod",headers=hdrs(),timeout=5,
            json={"securityId":SECID.get(CONFIG["INDEX"],"13"),"exchangeSegment":"IDX_I",
                  "instrument":"INDEX","expiryCode":0,"oi":False,"fromDate":today,"toDate":today})
        if r.status_code==200:
            closes=r.json().get("close",[])
            if len(closes)>=2: STATE["nifty_prev"]=closes[-2]; add_alert("info",f"Prev close:{closes[-2]}")
    except: pass

def fetch_15min_candles():
    if STATE["token_valid"]:
        try:
            today=datetime.now().strftime("%Y-%m-%d")
            r=requests.post(f"{DHAN_API}/v2/charts/intraday",headers=hdrs(),timeout=5,
                json={"securityId":SECID.get(CONFIG["INDEX"],"13"),"exchangeSegment":"IDX_I",
                      "instrument":"INDEX","interval":"15","oi":False,"fromDate":today,"toDate":today})
            if r.status_code==200:
                d=r.json(); ts=d.get("timestamp",[])
                if ts:
                    STATE["candles_15m"]=[{"t":d["timestamp"][i],"o":d["open"][i],"h":d["high"][i],
                        "l":d["low"][i],"c":d["close"][i],"v":d.get("volume",[0]*999)[i]}
                        for i in range(len(ts))]
                    _calc_ema(); return STATE["candles_15m"]
        except Exception as e: add_alert("warn",f"Candle fetch:{e}")
    if not STATE["candles_15m"]:
        STATE["candles_15m"]=generate_sim_candles(100,STATE["nifty"] or 22350)
        add_alert("info","Using sim candles — add DHAN_TOKEN for real data.")
    return STATE["candles_15m"]

def fetch_training_candles():
    if STATE["token_valid"]:
        try:
            from datetime import timedelta
            end=datetime.now(); start=end-timedelta(days=90)
            r=requests.post(f"{DHAN_API}/v2/charts/intraday",headers=hdrs(),timeout=15,
                json={"securityId":SECID.get(CONFIG["INDEX"],"13"),"exchangeSegment":"IDX_I",
                      "instrument":"INDEX","interval":"15","oi":False,
                      "fromDate":start.strftime("%Y-%m-%d"),"toDate":end.strftime("%Y-%m-%d")})
            if r.status_code==200:
                d=r.json(); ts=d.get("timestamp",[])
                candles=[{"t":d["timestamp"][i],"o":d["open"][i],"h":d["high"][i],
                    "l":d["low"][i],"c":d["close"][i],"v":d.get("volume",[0]*999)[i]}
                    for i in range(len(ts))]
                add_alert("success",f"Fetched {len(candles)} real candles for training.")
                return candles
        except Exception as e: add_alert("warn",f"Training data fetch:{e}")
    add_alert("info","Generating 500 simulated candles for AI training.")
    return generate_sim_candles(500,STATE["nifty"] or 22350)

def _calc_ema():
    closes=[c["c"] for c in STATE["candles_15m"]]
    if len(closes)<2: return
    k,ema=2/21,closes[0]
    for c in closes[1:]: ema=c*k+ema*(1-k)
    STATE["ema15"]=round(ema,2)

def calc_or():
    n=max(1,CONFIG["OR_MINUTES"]//15); cs=STATE["candles_15m"][:n]
    if not cs: return False
    STATE["or_high"]=max(c["h"] for c in cs); STATE["or_low"]=min(c["l"] for c in cs)
    add_alert("success",f"OR — H:{STATE['or_high']} L:{STATE['or_low']} Rng:{round(STATE['or_high']-STATE['or_low'],1)}pts")
    return True

def fetch_5min_candles():
    """Fetch today's 5-min candles from Dhan or simulate."""
    if STATE["token_valid"]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            r = requests.post(f"{DHAN_API}/v2/charts/intraday", headers=hdrs(), timeout=5,
                json={"securityId":SECID.get(CONFIG["INDEX"],"13"),
                      "exchangeSegment":"IDX_I","instrument":"INDEX",
                      "interval":"5","oi":False,"fromDate":today,"toDate":today})
            if r.status_code == 200:
                d = r.json(); ts = d.get("timestamp",[])
                if ts:
                    return [{"t":d["timestamp"][i],"o":d["open"][i],"h":d["high"][i],
                             "l":d["low"][i],"c":d["close"][i],
                             "v":d.get("volume",[0]*9999)[i]}
                            for i in range(len(ts))]
        except Exception as e:
            add_alert("warn", f"5m candle fetch: {e}")
    # Sim fallback
    from ai_model import generate_sim_candles
    return generate_sim_candles(30, STATE["nifty"] or 22350)

def calc_or5():
    """OR from first 5-min candle only (9:15-9:20)."""
    candles = fetch_5min_candles()
    if not candles: return False
    first = candles[0]
    STATE["or5_high"] = first["h"]
    STATE["or5_low"]  = first["l"]
    rng = round(STATE["or5_high"] - STATE["or5_low"], 2)
    add_alert("info", f"5m-OR — H:{STATE['or5_high']} L:{STATE['or5_low']} Rng:{rng}pts")
    return True

def nearest_expiry():
    from datetime import timedelta
    today=datetime.now(); d=(3-today.weekday())%7
    if d==0 and today.hour>=15: d=7
    return (today+timedelta(days=d)).strftime("%Y-%m-%d")

def get_atm_option(sig):
    strike=round((STATE["nifty"] or 22350)/50)*50
    if STATE["token_valid"]:
        try:
            r=requests.get(f"{DHAN_API}/v2/optionchain",headers=hdrs(),
                params={"UnderlyingScrip":SECID.get(CONFIG["INDEX"],"13"),
                        "UnderlyingSeg":"IDX_I","Expiry":nearest_expiry()},timeout=5)
            if r.status_code==200:
                for row in r.json().get("data",[]):
                    if row.get("strikePrice")==strike:
                        det=row.get("callDetail" if sig=="CE" else "putDetail",{})
                        return {"security_id":det.get("securityId"),
                                "ltp":det.get("lastTradedPrice",0),"strike":strike}
        except: pass
    return {"security_id":None,"ltp":100,"strike":strike}

def place_order(security_id,txn,qty):
    if CONFIG["MODE"]=="paper":
        add_alert("warn",f"[PAPER] {txn} qty:{qty}")
        return {"orderId":f"PAPER_{int(time.time())}"}
    if not STATE["token_valid"]: add_alert("danger","Token invalid."); return None
    try:
        r=requests.post(f"{DHAN_API}/v2/orders",headers=hdrs(),timeout=5,
            json={"dhanClientId":CONFIG["CLIENT_ID"],"transactionType":txn,
                  "exchangeSegment":"NSE_FNO","productType":"INTRADAY",
                  "orderType":"MARKET","validity":"DAY",
                  "securityId":str(security_id),"quantity":qty,
                  "price":0,"triggerPrice":0,"disclosedQuantity":0,"afterMarketOrder":False})
        if r.status_code in [200,201]:
            d=r.json(); add_alert("success",f"[LIVE] Order:{d.get('orderId')}"); return d
        add_alert("danger",f"Order error {r.status_code}")
    except Exception as e: add_alert("danger",f"Order:{e}")
    return None

# ════════════════════════════════════════════
# WEBSOCKET
# ════════════════════════════════════════════
class DhanFeed:
    def __init__(self): self.ws=None; self.alive=False
    def on_open(self,ws):
        ws.send(json.dumps({"LoginReq":{"MsgCode":42,"ClientId":CONFIG["CLIENT_ID"],"Token":CONFIG["TOKEN"]}}))
        time.sleep(0.4)
        ws.send(json.dumps({"RequestCode":21,"InstrumentCount":3,"InstrumentList":[
            {"ExchangeSegment":"IDX_I","SecurityId":SECID["NIFTY"]},
            {"ExchangeSegment":"IDX_I","SecurityId":SECID["BANKNIFTY"]},
            {"ExchangeSegment":"IDX_I","SecurityId":SECID["VIX"]},
        ]}))
        STATE["ws_connected"]=True
        add_alert("success","WebSocket live — Nifty/BankNifty/VIX.")
    def on_message(self,ws,msg):
        try:
            if isinstance(msg,bytes) and len(msg)>=13:
                sid=str(struct.unpack_from(">I",msg,1)[0]); ltp=struct.unpack_from(">d",msg,5)[0]
            else:
                d=json.loads(msg); sid=str(d.get("security_id",d.get("SecurityId",""))); ltp=float(d.get("ltp",d.get("LTP",0)))
            if ltp<=0: return
            if sid==SECID["NIFTY"]:
                STATE["nifty"]=round(ltp,2)
                if not STATE["nifty_open"]: STATE["nifty_open"]=ltp
            elif sid==SECID["BANKNIFTY"]: STATE["banknifty"]=round(ltp,2)
            elif sid==SECID["VIX"]: STATE["vix"]=round(ltp,2)
            STATE["last_tick"]=datetime.now().strftime("%H:%M:%S")
        except: pass
    def on_error(self,ws,e): STATE["ws_connected"]=False
    def on_close(self,ws,c,m):
        STATE["ws_connected"]=False
        if self.alive: time.sleep(5); self._run()
    def _run(self):
        if not WS_OK or not CONFIG["TOKEN"]: return
        self.ws=websocket.WebSocketApp(DHAN_WS,
            header={"access-token":CONFIG["TOKEN"],"client-id":CONFIG["CLIENT_ID"]},
            on_open=self.on_open,on_message=self.on_message,
            on_error=self.on_error,on_close=self.on_close)
        self.ws.run_forever(ping_interval=30,ping_timeout=10)
    def start(self): self.alive=True; threading.Thread(target=self._run,daemon=True).start()
    def stop(self): self.alive=False; (self.ws.close() if self.ws else None)

feed=DhanFeed()

# ════════════════════════════════════════════
# FILTERS
# ════════════════════════════════════════════
def gap_ok():
    if not STATE["nifty_prev"] or not STATE["nifty"]: return True
    return abs((STATE["nifty"]-STATE["nifty_prev"])/STATE["nifty_prev"]*100)<=CONFIG["GAP_MAX_PCT"]
def vix_ok():
    if not STATE["vix"]: return True
    return CONFIG["VIX_MIN"]<=STATE["vix"]<=CONFIG["VIX_MAX"]
def trend_ok(sig):
    if not STATE["ema15"]: return True
    return STATE["nifty"]>STATE["ema15"] if sig=="CE" else STATE["nifty"]<STATE["ema15"]
def time_ok():
    n=datetime.now().time()
    return (dtime(9,30)<=n<=dtime(11,30)) or (dtime(14,15)<=n<=dtime(14,45))
def limit_ok():
    return STATE["daily_loss"]<CONFIG["CAPITAL"]*CONFIG["RISK_PCT"]/100*3

# ════════════════════════════════════════════
# AI ENGINE
# ════════════════════════════════════════════
def run_ai_prediction():
    if not CONFIG["AI_ENABLED"] or len(STATE["candles_15m"])<20: return
    result=ai.predict(STATE["candles_15m"])
    STATE["ai_signal"]    =result.get("signal","NEUTRAL")
    STATE["ai_prob"]      =result.get("prob",0.5)
    STATE["ai_prob_down"] =result.get("prob_down",0.5)
    STATE["ai_confidence"]=result.get("confidence","LOW")
    STATE["ai_reason"]    =result.get("reason","")
    STATE["ai_trained"]   =result.get("trained",False)
    STATE["ai_prediction_history"].insert(0,{
        "time":datetime.now().strftime("%H:%M"),
        "prob":STATE["ai_prob"],"signal":STATE["ai_signal"],
        "confidence":STATE["ai_confidence"],
    })
    STATE["ai_prediction_history"]=STATE["ai_prediction_history"][:50]

def ai_approves(orb_sig):
    """
    Returns (approved:bool, score:float, reason:str)
    Higher score = stronger signal = more confidence to trade.
    """
    if not CONFIG["AI_ENABLED"]: return True,50.0,"AI disabled"
    if not STATE["ai_trained"]:  return True,50.0,"AI untrained — allowing"
    prob=STATE["ai_prob"]; prob_dn=STATE["ai_prob_down"]
    wants_up=orb_sig=="CE"
    target_prob=prob if wants_up else prob_dn
    direction="CE" if wants_up else "PE"
    if target_prob>=CONFIG["AI_HIGH_CONF"]:
        return True,round(target_prob*100,1),f"🔥 STRONG AI {direction} — {target_prob:.0%}"
    if target_prob>=CONFIG["AI_THRESHOLD"]:
        return True,round(target_prob*100,1),f"✅ AI confirms {direction} — {target_prob:.0%}"
    return False,round(target_prob*100,1),f"❌ AI {target_prob:.0%} < threshold {CONFIG['AI_THRESHOLD']:.0%}"

def signal_score(orb_sig,ai_prob,ai_ok):
    s=0
    if orb_sig: s+=35
    if ai_ok:   s+=round(ai_prob*35)
    if gap_ok(): s+=5; 
    if vix_ok(): s+=5
    if time_ok(): s+=10
    if limit_ok(): s+=10
    return min(100,s)

def retrain_ai():
    if STATE["ai_training"]: return
    STATE["ai_training"]=True
    add_alert("info","🤖 AI retraining started…")
    def _do():
        candles=fetch_training_candles()
        result=ai.train(candles,log_fn=lambda m:add_alert("info",f"[AI] {m}"))
        STATE["ai_trained"]=result.get("status")=="success"
        STATE["ai_last_trained"]=datetime.now().strftime("%H:%M")
        STATE["ai_metrics"]=ai.summary; STATE["ai_training"]=False
        if STATE["ai_trained"]:
            add_alert("success",
                f"✅ AI trained — Acc:{result.get('accuracy')}% | "
                f"HighConf:{result.get('hc_accuracy')}% | "
                f"Samples:{result.get('samples')} | {result.get('lib')}")
        else:
            add_alert("danger",f"AI training failed: {result.get('error')}")
    threading.Thread(target=_do,daemon=True).start()

# ════════════════════════════════════════════
# STRATEGY LOOP
# ════════════════════════════════════════════
def strategy_loop():
    add_alert("info","🚀 Combined ORB+AI engine started.")
    while STATE["strategy_on"]:
        t=datetime.now().time()
        if not STATE["nifty_prev"] and STATE["token_valid"]: fetch_prev_close()
        if not STATE["or_fetched"] and t>=dtime(9,30):
            if fetch_15min_candles() and calc_or(): STATE["or_fetched"]=True
        run_ai_prediction()
        if not time_ok():
            if t>=dtime(15,15) and STATE["position"]: _eod_close()
            time.sleep(10); continue
        if not gap_ok(): time.sleep(60); continue
        if not vix_ok(): time.sleep(30); continue
        if STATE["position"]: _monitor(); time.sleep(3); continue
        if STATE["or_high"] and not STATE["signal"]:
            price=STATE["nifty"] or 0
            if not price: time.sleep(3); continue
            if price>STATE["or_high"] and STATE["sweep_dir"]!="SWEEP_UP":
                STATE["sweep_dir"]="SWEEP_UP"; STATE["sweep_candles"]=0
                add_alert("warn",f"⚡ UPSIDE SWEEP {price:.0f} > {STATE['or_high']}")
            elif price<STATE["or_low"] and STATE["sweep_dir"]!="SWEEP_DOWN":
                STATE["sweep_dir"]="SWEEP_DOWN"; STATE["sweep_candles"]=0
                add_alert("warn",f"⚡ DOWNSIDE SWEEP {price:.0f} < {STATE['or_low']}")
            if STATE["sweep_dir"]:
                STATE["sweep_candles"]+=1
                orb_sig="PE" if STATE["sweep_dir"]=="SWEEP_UP" else "CE"
                returned=((STATE["sweep_dir"]=="SWEEP_UP" and price<STATE["or_high"]) or
                          (STATE["sweep_dir"]=="SWEEP_DOWN" and price>STATE["or_low"]))
                if returned:
                    cs=STATE["candles_15m"]
                    vol_ok=True
                    if len(cs)>=3:
                        avg=sum(c["v"] for c in cs[-3:])/3
                        vol_ok=cs[-1]["v"]>avg*1.3 if avg>0 else True
                    if not vol_ok:
                        add_alert("warn","Volume low — skip."); STATE["skipped_count"]+=1; _rsweep(); continue
                    if not trend_ok(orb_sig):
                        add_alert("block",f"Trend filter: {orb_sig} blocked."); STATE["blocked_count"]+=1; _rsweep(); continue
                    # ── AI CHECK ──
                    STATE["orb_signal"]=orb_sig
                    ai_ok,ai_sc,ai_reason=ai_approves(orb_sig)
                    sc=signal_score(orb_sig,STATE["ai_prob"],ai_ok)
                    STATE["signal_score"]=sc
                    if not ai_ok:
                        STATE["ai_blocked"]+=1
                        add_alert("block",f"🤖 AI BLOCKED — {ai_reason} | Score:{sc}/100")
                        _rsweep(); continue
                    STATE["ai_confirmed"]+=1
                    add_alert("success",f"🤖 AI CONFIRMED — {ai_reason} | Score:{sc}/100")
                    _execute(orb_sig,sc)
                elif STATE["sweep_candles"]>=3:
                    add_alert("warn","3C rule: no return. Void."); STATE["skipped_count"]+=1; _rsweep()
        # ── 5-MIN OR ENGINE (parallel, 2 lots, AI≥68%) ──
        if not STATE["or5_fetched"] and t >= dtime(9,20):
            if calc_or5(): STATE["or5_fetched"] = True

        if (STATE["or5_high"] and not STATE["signal"]
                and not STATE["position"] and time_ok()):
            _check_or5()

        time.sleep(3)
    add_alert("info","Engine stopped.")

def dynamic_lots(score):
    """
    Fixed 2-lot system — quality over size.
    Score >= 65 → 2 lots (AI confirmed)
    Score <  65 → blocked (never reaches here)
    Hard cap: 2 lots maximum always.
    """
    return 2  # fixed — testing phase, never more than 2

def _check_or5():
    """
    5-min OR sweep engine.
    Runs parallel to 15-min engine.
    Stricter: AI≥68%, SL20%, TGT60%, 2 lots only, window 9:20-10:00 AM only.
    """
    now = datetime.now().time()
    # 5-min OR only valid 9:20–10:00 AM
    if not (dtime(9,20) <= now <= dtime(10,0)):
        return

    price = STATE["nifty"] or 0
    if not price or not STATE["or5_high"]: return

    or5h = STATE["or5_high"]
    or5l = STATE["or5_low"]

    # Detect sweep
    if price > or5h and STATE["or5_sweep_dir"] != "UP":
        STATE["or5_sweep_dir"]   = "UP"
        STATE["or5_sweep_count"] = 0
        add_alert("warn", f"⚡ 5m SWEEP UP {price:.0f} > {or5h:.0f}")

    elif price < or5l and STATE["or5_sweep_dir"] != "DOWN":
        STATE["or5_sweep_dir"]   = "DOWN"
        STATE["or5_sweep_count"] = 0
        add_alert("warn", f"⚡ 5m SWEEP DOWN {price:.0f} < {or5l:.0f}")

    if STATE["or5_sweep_dir"]:
        STATE["or5_sweep_count"] += 1

        returned = (
            (STATE["or5_sweep_dir"] == "UP"   and or5l < price < or5h) or
            (STATE["or5_sweep_dir"] == "DOWN" and or5l < price < or5h)
        )

        if returned and STATE["or5_sweep_count"] <= 6:
            sig = "PE" if STATE["or5_sweep_dir"] == "UP" else "CE"

            # Stricter AI gate for 5-min: ≥68%
            run_ai_prediction()
            ai_ok, ai_sc, ai_reason = ai_approves(sig)
            target_prob = STATE["ai_prob"] if sig=="CE" else STATE["ai_prob_down"]

            if target_prob < 0.68:
                STATE["or5_blocked"] += 1
                add_alert("block",
                    f"🤖 5m AI blocked {sig} — prob:{target_prob:.0%} < 68%")
                _reset_or5(); return

            if not trend_ok(sig):
                STATE["or5_blocked"] += 1
                add_alert("block", f"5m trend filter blocked {sig}")
                _reset_or5(); return

            sc = signal_score(sig, target_prob, True)
            add_alert("success",
                f"5m-OR SIGNAL {sig} | AI:{target_prob:.0%} | Score:{sc}/100")
            _execute_or5(sig, sc)

        elif STATE["or5_sweep_count"] > 6:
            add_alert("warn", "5m: no return in 6 candles. Void.")
            _reset_or5()

def _reset_or5():
    STATE["or5_sweep_dir"]   = None
    STATE["or5_sweep_count"] = 0

def _execute_or5(sig, score=60):
    """Execute 5-min OR trade — fixed 2 lots, SL20%, TGT60%."""
    opt = get_atm_option(sig)
    ltp = opt["ltp"]
    if ltp <= 0:
        add_alert("danger","5m: LTP=0."); _reset_or5(); return

    sl  = round(ltp * 0.80, 2)   # SL 20%
    tgt = round(ltp * 1.60, 2)   # TGT 60%
    qty = 2 * LOT.get(CONFIG["INDEX"], 50)   # always 2 lots

    add_alert("success",
        f"✅ 5m {sig} {opt['strike']} | E:₹{ltp} SL:₹{sl} T:₹{tgt} | "
        f"2L | Score:{score}/100 | {CONFIG['MODE'].upper()}")

    order = place_order(opt["security_id"], "BUY", qty)
    if order:
        feat = build_features(STATE["candles_15m"])
        STATE["or5_signals"] += 1
        STATE.update({
            "signal":sig, "entry_price":ltp, "sl_price":sl,
            "target_price":tgt, "trail_high":ltp,
            "signals_count": STATE["signals_count"]+1,
            "position":{
                "symbol":   f"{CONFIG['INDEX']}{opt['strike']}{sig}",
                "type":     sig,
                "strike":   opt["strike"],
                "qty":      qty,
                "lots":     2,
                "entry":    ltp,
                "sl":       sl,
                "target":   tgt,
                "security_id": opt["security_id"],
                "order_id": order.get("orderId","SIM"),
                "entry_time": datetime.now().strftime("%H:%M"),
                "score":    score,
                "ai_prob":  STATE["ai_prob"],
                "source":   "5m-OR",
                "feature_vector": feat["vector"] if feat else [],
            }
        })
        _reset_or5()

def _execute(sig,score=50):
    opt=get_atm_option(sig); ltp=opt["ltp"]
    if ltp<=0: add_alert("danger","LTP=0."); _rsweep(); return
    boost=1.3 if score>=85 else 1.2 if score>=75 else 1.1 if score>=65 else 1.0
    sl=round(ltp*(1-CONFIG["SL_PCT"]/100),2)
    tgt=round(ltp*(1+CONFIG["TGT_PCT"]*boost/100),2)
    lots=dynamic_lots(score)
    qty=lots*LOT.get(CONFIG["INDEX"],50)
    lot_label=f"{lots}L 🔥"  # fixed 2-lot mode
    add_alert("success",
        f"✅ {sig} {opt['strike']} | E:₹{ltp} SL:₹{sl} T:₹{tgt} | "
        f"Score:{score}/100 | {lot_label} | {CONFIG['MODE'].upper()}")
    order=place_order(opt["security_id"],"BUY",qty)
    if order:
        feat=build_features(STATE["candles_15m"])
        STATE.update({"signal":sig,"entry_price":ltp,"sl_price":sl,
            "target_price":tgt,"trail_high":ltp,
            "signals_count":STATE["signals_count"]+1,
            "position":{"symbol":f"{CONFIG['INDEX']}{opt['strike']}{sig}",
                "type":sig,"strike":opt["strike"],"qty":qty,
                "lots":lots,
                "entry":ltp,"sl":sl,"target":tgt,
                "security_id":opt["security_id"],
                "order_id":order.get("orderId","SIM"),
                "entry_time":datetime.now().strftime("%H:%M"),
                "score":score,"ai_prob":STATE["ai_prob"],
                "feature_vector":feat["vector"] if feat else []}})
        _rsweep()

def _monitor():
    pos=STATE["position"]
    if not pos: return
    import random
    cur=(STATE["entry_price"]*(1+random.uniform(-0.04,0.07))
         if CONFIG["MODE"]=="paper"
         else get_atm_option(pos["type"]).get("ltp",STATE["entry_price"]))
    if cur>STATE["trail_high"]:
        STATE["trail_high"]=cur
        profit=STATE["trail_high"]-STATE["entry_price"]
        if profit*pos["qty"]>=CONFIG["TRAIL_EVERY"]:
            new_sl=round(STATE["entry_price"]+profit*0.5,2)
            if new_sl>STATE["sl_price"]: STATE["sl_price"]=new_sl; add_alert("info",f"Trail SL→₹{new_sl}")
    if cur<=STATE["sl_price"]: _close(cur,"SL ⛔")
    elif cur>=STATE["target_price"]: _close(cur,"Target ✅")

def _close(exit_px,reason):
    pos=STATE["position"]
    if not pos: return
    place_order(pos["security_id"],"SELL",pos["qty"])
    pnl=round((exit_px-pos["entry"])*pos["qty"],2)
    if pos.get("feature_vector"): ai.record_trade_outcome(pos["feature_vector"],1 if pnl>0 else 0,pnl)
    STATE["trades"].insert(0,{"time":pos["entry_time"],"exit_time":datetime.now().strftime("%H:%M"),
        "type":pos["type"],"strike":pos["strike"],"entry":pos["entry"],"exit":exit_px,
        "lots":CONFIG["LOTS"],"pnl":pnl,"reason":reason,"mode":CONFIG["MODE"],
        "score":pos.get("score",0),"ai_prob":pos.get("ai_prob",0)})
    STATE["trades"]=STATE["trades"][:50]
    if pnl>=0: STATE["gross_win"]+=pnl
    else: STATE["gross_loss"]+=abs(pnl); STATE["daily_loss"]+=abs(pnl)
    add_alert("success" if pnl>=0 else "danger",f"{reason} | ₹{exit_px} | P&L:{'+' if pnl>=0 else ''}₹{pnl}")
    STATE.update({"position":None,"signal":None,"orb_signal":None,"entry_price":0})

def _eod_close():
    if STATE["position"]: _close(get_atm_option(STATE["position"]["type"]).get("ltp",STATE["entry_price"]),"EOD ⏰")

def _rsweep(): STATE["sweep_dir"]=None; STATE["sweep_candles"]=0

def add_alert(level,msg):
    STATE["alerts"].insert(0,{"time":datetime.now().strftime("%H:%M:%S"),"level":level,"msg":msg})
    STATE["alerts"]=STATE["alerts"][:40]; print(f"[{level.upper()}] {msg}")

# ════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════
@app.route("/") 
def index(): return render_template("index.html")

@app.route("/api/state")
def get_state():
    pnl=round(STATE["gross_win"]-STATE["gross_loss"],2)
    wins=sum(1 for t in STATE["trades"] if t["pnl"]>=0)
    wr=round(wins/len(STATE["trades"])*100) if STATE["trades"] else 0
    gap=round(abs((STATE["nifty"]-STATE["nifty_prev"])/max(STATE["nifty_prev"],1)*100),2) if STATE["nifty_prev"] else 0
    return jsonify({
        "ws_connected":STATE["ws_connected"],"token_valid":STATE["token_valid"],"last_tick":STATE["last_tick"],
        "nifty":STATE["nifty"],"nifty_prev":STATE["nifty_prev"],"banknifty":STATE["banknifty"],
        "vix":STATE["vix"],"ema15":STATE["ema15"],
        "strategy_on":STATE["strategy_on"],"mode":CONFIG["MODE"],
        "signal":STATE["signal"],"orb_signal":STATE["orb_signal"],
        "or_high":STATE["or_high"],"or_low":STATE["or_low"],
        "sweep_dir":STATE["sweep_dir"],"sweep_candles":STATE["sweep_candles"],
        "signal_score":STATE["signal_score"],
        "ai_enabled":CONFIG["AI_ENABLED"],"ai_trained":STATE["ai_trained"],
        "ai_training":STATE["ai_training"],"ai_signal":STATE["ai_signal"],
        "ai_prob":STATE["ai_prob"],"ai_prob_down":STATE["ai_prob_down"],
        "ai_confidence":STATE["ai_confidence"],"ai_reason":STATE["ai_reason"],
        "ai_last_trained":STATE["ai_last_trained"],"ai_metrics":STATE["ai_metrics"],
        "ai_blocked":STATE["ai_blocked"],"ai_confirmed":STATE["ai_confirmed"],
        "ai_prediction_history":STATE["ai_prediction_history"][:10],
        "position":STATE["position"],"pnl":pnl,
        "gross_win":STATE["gross_win"],"gross_loss":STATE["gross_loss"],
        "win_rate":wr,"trades":STATE["trades"][:20],"trade_count":len(STATE["trades"]),
        "signals_count":STATE["signals_count"],"blocked_count":STATE["blocked_count"],
        "skipped_count":STATE["skipped_count"],
        "filters":{"gap":gap_ok(),"gap_pct":gap,"vix":vix_ok(),"time":time_ok(),"limit":limit_ok()},
        "or5_high":STATE["or5_high"],"or5_low":STATE["or5_low"],
        "or5_sweep_dir":STATE["or5_sweep_dir"],
        "or5_signals":STATE["or5_signals"],"or5_blocked":STATE["or5_blocked"],
        "alerts":STATE["alerts"][:20],
        "config":{"client_id":CONFIG["CLIENT_ID"],"mode":CONFIG["MODE"],"index":CONFIG["INDEX"],
                  "lots":CONFIG["LOTS"],"capital":CONFIG["CAPITAL"],"risk_pct":CONFIG["RISK_PCT"],
                  "ai_threshold":CONFIG["AI_THRESHOLD"],"ai_high_conf":CONFIG["AI_HIGH_CONF"]},
    })

@app.route("/api/strategy/start",methods=["POST"])
def start():
    if STATE["strategy_on"]: return jsonify({"status":"already_running"})
    STATE["strategy_on"]=True
    threading.Thread(target=strategy_loop,daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/strategy/stop",methods=["POST"])
def stop():
    STATE["strategy_on"]=False; return jsonify({"status":"stopped"})

@app.route("/api/ai/train",methods=["POST"])
def train_ai(): retrain_ai(); return jsonify({"status":"training_started"})

@app.route("/api/ai/threshold",methods=["POST"])
def set_threshold():
    t=float((request.json or {}).get("threshold",0.65))
    CONFIG["AI_THRESHOLD"]=max(0.55,min(0.90,t))
    add_alert("info",f"AI threshold → {CONFIG['AI_THRESHOLD']:.0%}")
    return jsonify({"threshold":CONFIG["AI_THRESHOLD"]})

@app.route("/api/square_off",methods=["POST"])
def sq(): _eod_close(); return jsonify({"status":"squared_off"})

@app.route("/api/config",methods=["GET","POST"])
def cfg():
    if request.method=="POST":
        for k in ["LOTS","CAPITAL","RISK_PCT","SL_PCT","TGT_PCT","MODE","INDEX",
                  "OR_MINUTES","AI_ENABLED","AI_THRESHOLD","AI_HIGH_CONF"]:
            if k in (request.json or {}): CONFIG[k]=request.json[k]
        return jsonify({"status":"updated"})
    return jsonify({k:v for k,v in CONFIG.items() if "TOKEN" not in k})

@app.route("/api/reset",methods=["POST"])
def reset():
    for k in ["trades","alerts","ai_prediction_history"]: STATE[k]=[]
    for k in ["gross_win","gross_loss","daily_loss","signals_count","blocked_count",
              "skipped_count","ai_blocked","ai_confirmed","entry_price","signal_score"]: STATE[k]=0
    for k in ["or_high","or_low","sweep_dir","signal","orb_signal","position",
              "or5_high","or5_low","or5_sweep_dir"]: STATE[k]=None
    STATE["or_fetched"]=False; STATE["sweep_candles"]=0
    STATE["or5_fetched"]=False; STATE["or5_sweep_count"]=0
    STATE["or5_signals"]=0; STATE["or5_blocked"]=0
    add_alert("info","Session reset."); return jsonify({"status":"reset"})

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","ws":STATE["ws_connected"],"token":STATE["token_valid"],
                    "nifty":STATE["nifty"],"ai_trained":STATE["ai_trained"],"mode":CONFIG["MODE"]})

# ════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════
def startup():
    print(f"\n{'='*50}\n  Train-AI | ORB + AI Trading\n"
          f"  Client:{CONFIG['CLIENT_ID']} Mode:{CONFIG['MODE']}\n"
          f"  Token:{'SET ✅' if CONFIG['TOKEN'] else 'MISSING — sim mode'}\n"
          f"  AI:{ai.lib_type} Trained:{ai.is_ready}\n{'='*50}\n")
    if CONFIG["TOKEN"]:
        if validate_token(): feed.start(); time.sleep(2); fetch_prev_close()
    else:
        STATE["nifty"]=22350.0; STATE["nifty_prev"]=22250.0
        add_alert("warn","No token — simulation mode.")
    if not ai.is_ready:
        add_alert("info","First run — training AI…"); retrain_ai()
    def _daily():
        while True:
            now=datetime.now()
            if now.hour==CONFIG["AI_RETRAIN_HOUR"] and now.minute==30:
                add_alert("info","⏰ Daily AI retrain."); retrain_ai(); time.sleep(70)
            time.sleep(30)
    threading.Thread(target=_daily,daemon=True).start()

if __name__=="__main__":
    port=int(os.environ.get("PORT",8080))
    threading.Thread(target=startup,daemon=True).start()
    app.run(host="0.0.0.0",port=port,debug=False)
