# market_snapshot_app.py
# ---------------------------------------
# Tabs: FX / Commods / Stocks 
# FX/Commods/Stocks via Yahoo (yfinance)
# Includes: Snapshot Time + As-of time per asset/table
# Uses: st.rerun() (no experimental_rerun)
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
import contextlib
import os
import sys
import requests

# -----------------------------
# Global settings
# -----------------------------
LONDON = ZoneInfo("Europe/London")

# -----------------------------
# Asset ordering (exactly as you sent)
# -----------------------------
ORDER = [
    # FX - Dollar
    "DXY", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "NZD/USD",
    "USD/CAD", "USD/CHF", "USD/CNH", "USD/CNY",
    # FX - Euro
    "EUR/CHF", "EUR/GBP", "EUR/SEK", "EUR/NOK", "EUR/PLN",
    # FX - Crosses
    "AUD/NZD", "CNH/JPY",
    # Commodities
    "WTI", "Brent", "XAU (Gold)", "XAG (Silver)", "PLD (Palladium)", "PLT (Platinum)",
    "HG (Copper)", "Iron Ore", "Aluminum", "Nickel",
    # Equities (Cash)
    "Euro Stoxx 50", "Euro Stoxx 600", "DAX 40", "FTSE 100", "CAC 40", "FTSE MIB",
    "IBEX 35", "PSI", "SMI", "AEX", "S&P 500", "NDX", "DJI", "RUT",
    # Equities (Futures)
    "S&P 500 Fut", "Nasdaq 100 Fut", "Russell 2000 Fut", "Dow Jones Fut"
]

TENOR_ORDER = {"1M": 0, "3M": 1, "2Y": 2, "5Y": 3, "10Y": 4, "30Y": 5}


# -----------------------------
# Universe
# -----------------------------
FX = {
    # Dollar
    "DXY": ["DX-Y.NYB"],
    "EUR/USD": ["EURUSD=X"],
    "USD/JPY": ["JPY=X"],
    "GBP/USD": ["GBPUSD=X"],
    "AUD/USD": ["AUDUSD=X"],
    "NZD/USD": ["NZDUSD=X"],
    "USD/CAD": ["CAD=X"],
    "USD/CHF": ["CHF=X"],
    "USD/CNH": ["USDCNH=X"],
    "USD/CNY": ["USDCNY=X"],

    # Euro
    "EUR/CHF": ["EURCHF=X"],
    "EUR/GBP": ["EURGBP=X"],
    "EUR/SEK": ["EURSEK=X"],
    "EUR/NOK": ["EURNOK=X"],
    "EUR/PLN": ["EURPLN=X"],

    # Crosses
    "AUD/NZD": ["AUDNZD=X"],
    "CNH/JPY": ["CNHJPY=X"],
}

COMMODITIES = {
    "WTI": ["CL=F"],
    "Brent": ["BZ=F"],
    "XAU (Gold)": ["GC=F", "XAUUSD=X"],
    "XAG (Silver)": ["SI=F", "XAGUSD=X"],
    "PLD (Palladium)": ["PA=F"],
    "PLT (Platinum)": ["PL=F"],
    "HG (Copper)": ["HG=F"],
    "Iron Ore": ["TIO=F"],                 # may be patchy on Yahoo
    "Aluminum": ["ALI=F", "AL=F"],         # may be patchy on Yahoo
    "Nickel": ["^SPGSIK", "NICK.L"],       # proxy tickers (Yahoo coverage varies)
}

EQUITIES_CASH = {
    "Euro Stoxx 50": ["^STOXX50E"],
    "Euro Stoxx 600": ["^STOXX"],
    "DAX 40": ["^GDAXI"],
    "FTSE 100": ["^FTSE"],
    "CAC 40": ["^FCHI"],
    "FTSE MIB": ["FTSEMIB.MI"],
    "IBEX 35": ["^IBEX"],
    "PSI": ["PSI20.LS"],
    "SMI": ["^SSMI"],
    "AEX": ["^AEX"],
    "S&P 500": ["^GSPC"],
    "NDX": ["^NDX"],
    "DJI": ["^DJI"],
    "RUT": ["^RUT"],
}

EQUITIES_FUTURES = {
    "S&P 500 Fut": ["ES=F"],
    "Nasdaq 100 Fut": ["NQ=F"],
    "Russell 2000 Fut": ["RTY=F"],
    "Dow Jones Fut": ["YM=F"],
}

UST_YIELDS = {
    "UST 3M": "^IRX",
    "UST 5Y": "^FVX",
    "UST 10Y": "^TNX",
    "UST 30Y": "^TYX",
}


# -----------------------------
# YFinance fetch
# -----------------------------

def yahoo_quote(symbol: str) -> dict:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = requests.get(url, params={"symbols": symbol}, timeout=10)
    r.raise_for_status()
    result = r.json()["quoteResponse"]["result"]
    return result[0] if result else {}




def fetch_one(asset_name: str, tickers: list[str], group: str, snapshot_time_london: str) -> dict:
    last_err = "Unknown error"

    for tkr in tickers:
        try:
            # ---------- Intraday ----------
            intraday = yf.download(
                tkr,
                period="2d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if intraday is None or intraday.empty:
                raise ValueError("No intraday data")

            # Timezone handling
            if getattr(intraday.index, "tz", None) is None:
                intraday = intraday.copy()
                intraday.index = intraday.index.tz_localize("UTC")

            intraday.index = intraday.index.tz_convert("Europe/London")

            last_bar_ts = intraday.index[-1]
            asof_london = last_bar_ts.strftime("%Y-%m-%d %H:%M")

            last = float(intraday["Close"].iloc[-1])
            high = float(intraday["High"].max())
            low = float(intraday["Low"].min())

            # ---------- Previous close ----------
            prev_close = np.nan

            # 1) Yahoo quote fields
            try:
                t = yf.Ticker(tkr)
                fi = getattr(t, "fast_info", None)

                if fi is not None and hasattr(fi, "get"):
                    prev_close = fi.get("previous_close", np.nan)

                if pd.isna(prev_close):
                    info = t.info
                    prev_close = info.get("previousClose", np.nan)

            except Exception:
                prev_close = np.nan

            # 2) Fallback: daily history
            if pd.isna(prev_close):
                try:
                    daily = yf.download(
                        tkr,
                        period="10d",
                        interval="1d",
                        auto_adjust=False,
                        progress=False,
                        threads=False
                    )

                    if daily is not None and not daily.empty and "Close" in daily:
                        closes = daily["Close"].dropna()
                        if len(closes) >= 2:
                            prev_close = float(closes.iloc[-2])
                        elif len(closes) == 1:
                            prev_close = float(closes.iloc[-1])

                except Exception:
                    pass

            # ---------- Change ----------
            if pd.isna(prev_close) or float(prev_close) == 0.0:
                chg = np.nan
                chg_pct = np.nan
            else:
                prev_close = float(prev_close)
                chg = last - prev_close
                chg_pct = (chg / prev_close) * 100.0

            return {
                "Snapshot Time (London)": snapshot_time_london,
                "As of (London)": asof_london,
                "Group": group,
                "Asset": asset_name,
                "Ticker": tkr,
                "Last": last,
                "High": high,
                "Low": low,
                "Chg": chg,
                "Chg%": chg_pct,
                "Status": "OK",
            }

        except Exception as e:
            last_err = str(e)

    return {
        "Snapshot Time (London)": snapshot_time_london,
        "As of (London)": np.nan,
        "Group": group,
        "Asset": asset_name,
        "Ticker": tickers[0] if tickers else "",
        "Last": np.nan,
        "High": np.nan,
        "Low": np.nan,
        "Chg": np.nan,
        "Chg%": np.nan,
        "Status": f"FAILED: {last_err[:120]}",
    }




def snapshot(universe: dict, group: str, snapshot_time_london: str) -> pd.DataFrame:
    return pd.DataFrame([fetch_one(asset, tickers, group, snapshot_time_london) for asset, tickers in universe.items()])

    
def fetch_yield(name: str, ticker: str, snapshot_time_london: str) -> dict:
    try:
        y = yf.Ticker(ticker)

        last = np.nan
        asof_london = snapshot_time_london

        # --- Try quote-style first (usually most current) ---
        try:
            fi = getattr(y, "fast_info", None)
            if fi is not None and hasattr(fi, "get"):
                last = fi.get("last_price", np.nan)
                if pd.isna(last):
                    last = fi.get("lastPrice", np.nan)
        except Exception:
            pass

        # --- Fallback to info ---
        if pd.isna(last):
            try:
                inf = y.info
                last = inf.get("regularMarketPrice", np.nan)
                ts = inf.get("regularMarketTime", None)
                if ts is not None:
                    asof_london = (
                        datetime.fromtimestamp(int(ts), tz=ZoneInfo("UTC"))
                        .astimezone(LONDON)
                        .strftime("%Y-%m-%d %H:%M")
                    )
            except Exception:
                pass

        # --- Fallback to history ---
        if pd.isna(last):
            try:
                hist = y.history(period="1d", interval="1m")
                if hist is None or hist.empty:
                    hist = y.history(period="5d", interval="5m")
                if hist is None or hist.empty:
                    raise ValueError("No recent quote data")

                if getattr(hist.index, "tz", None) is None:
                    hist = hist.copy()
                    hist.index = hist.index.tz_localize("UTC")
                hist.index = hist.index.tz_convert("Europe/London")

                last = float(hist["Close"].iloc[-1])
                asof_london = hist.index[-1].strftime("%Y-%m-%d %H:%M")

            except Exception:
                pass

        # --- Yahoo previous close (for daily change) ---
        prev_close = np.nan

        try:
            fi = getattr(y, "fast_info", None)
            if fi is not None and hasattr(fi, "get"):
                prev_close = fi.get("previous_close", np.nan)
        except Exception:
            prev_close = np.nan

        if pd.isna(prev_close):
            try:
                inf = y.info
                prev_close = inf.get("previousClose", np.nan)
            except Exception:
                prev_close = np.nan

        # --- Compute daily change in bps ---
        if pd.isna(prev_close) or float(prev_close) == 0.0:
            chg_bps = np.nan
        else:
            chg_bps = (float(last) - float(prev_close)) * 100.0

        return {
            "As of (London)": asof_london,
            "Asset": name,
            "Ticker": ticker,
            "Yield (%)": round(float(last), 3),
            "Chg (bps)": round(float(chg_bps), 1) if not pd.isna(chg_bps) else np.nan,
            "Status": "OK",
        }

    except Exception as e:
        return {
            "As of (London)": snapshot_time_london,
            "Asset": name,
            "Ticker": ticker,
            "Yield (%)": np.nan,
            "Chg (bps)": np.nan,
            "Status": f"FAILED: {str(e)[:80]}",
        }





def build_ust_yields(snapshot_time_london: str) -> pd.DataFrame:
    rows = [fetch_yield(name, tkr, snapshot_time_london) for name, tkr in UST_YIELDS.items()]
    return pd.DataFrame(rows)




# -----------------------------
# Build all dataframes
# -----------------------------
def build_all():
    snapshot_time = datetime.now(LONDON).strftime("%Y-%m-%d %H:%M")

    df = pd.concat(
        [
            snapshot(FX, "FX", snapshot_time),
            snapshot(COMMODITIES, "Commodities", snapshot_time),
            snapshot(EQUITIES_CASH, "Equities (Cash)", snapshot_time),
            snapshot(EQUITIES_FUTURES, "Equities (Futures)", snapshot_time),
        ],
        ignore_index=True,
    )

    # Numeric formatting
    for c in ["Last", "High", "Low", "Chg", "Chg%"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Last"] = df["Last"].round(4)
    df["High"] = df["High"].round(4)
    df["Low"]  = df["Low"].round(4)
    df["Chg"]  = df["Chg"].round(4)
    df["Chg%"] = df["Chg%"].round(2)

    # Order assets
    df["order"] = df["Asset"].apply(lambda x: ORDER.index(x) if x in ORDER else 999)
    df = df.sort_values("order").drop(columns="order").reset_index(drop=True)

    # Drop columns no longer needed in UI
    df = df.drop(columns=["Group", "Snapshot Time (London)"], errors="ignore")

    # Move As of (London) to the front
    cols = df.columns.tolist()
    if "As of (London)" in cols:
        cols.insert(0, cols.pop(cols.index("As of (London)")))
        df = df[cols]

    return df


def style_changes(df: pd.DataFrame, decimals: int):
    def colour(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: green;"
        if val < 0:
            return "color: red;"
        return ""

    sty = df.style.map(colour, subset=["Chg", "Chg%"])

    fmt = {
        "Last": f"{{:.{decimals}f}}",
        "High": f"{{:.{decimals}f}}",
        "Low":  f"{{:.{decimals}f}}",
        "Chg":  f"{{:.{decimals}f}}",
        "Chg%": "{:.2f}",
    }

    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return sty.format(fmt, na_rep="")

def style_bond_changes(df: pd.DataFrame):
    def colour(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: green;"
        if val < 0:
            return "color: red;"
        return ""

    sty = df.style.map(colour, subset=["Chg (bps)"])

    fmt = {
        "Yield (%)": "{:.3f}",
        "Chg (bps)": "{:+.1f}",
    }

    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return sty.format(fmt, na_rep="")



# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Market Snapshot", layout="wide")
st.title("Market Snapshot")

now_london = datetime.now(LONDON).strftime("%Y-%m-%d %H:%M")
st.caption(f"Market snapshot as of {now_london} London")


col1, col2 = st.columns([1, 3])
with col1:
    refresh = st.button("Refresh now")

if refresh:
    st.cache_data.clear()
    st.rerun()

df = build_all()

tab_fx, tab_cmd, tab_stocks, tab_bonds = st.tabs(
    ["FX", "Commods", "Stocks", "Bonds"]
)


with tab_fx:
    st.subheader("FX")
    fx_df = df[df["Asset"].isin(FX.keys())]
    st.dataframe(style_changes(fx_df, 4), use_container_width=True)



with tab_cmd:
    st.subheader("Commodities")
    cmd_df = df[df["Asset"].isin(COMMODITIES.keys())]
    st.dataframe(style_changes(cmd_df, 3), use_container_width=True)



with tab_stocks:
    st.subheader("Equities")
    sub_cash, sub_fut = st.tabs(["Cash", "Futures"])

    with sub_cash:
        cash_df = df[df["Asset"].isin(EQUITIES_CASH.keys())]
        st.dataframe(style_changes(cash_df, 4), use_container_width=True)

    with sub_fut:
        fut_df = df[df["Asset"].isin(EQUITIES_FUTURES.keys())]
        st.dataframe(style_changes(fut_df, 4), use_container_width=True)


with tab_bonds:
    st.subheader("US Treasury Yields")

    snapshot_time = datetime.now(LONDON).strftime("%Y-%m-%d %H:%M")
    bonds_df = build_ust_yields(snapshot_time)

    st.dataframe(
        style_bond_changes(bonds_df),
        use_container_width=True,
        hide_index=True
    )



        





