"""
Production-ready personal off-exchange fund investment assistant.

Required environment variables:
- DEEPSEEK_API_KEY
- DEEPSEEK_BASE_URL (example: https://api.deepseek.com/v1)

Required Python packages and standard-library modules:
streamlit, akshare, pandas, openai, json, os, time, datetime, re, tenacity
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    import akshare as ak
except Exception:
    ak = None

try:
    import openai as _openai
except Exception:
    _openai = None


APP_TITLE = "Personal Off-Exchange Fund Investment Assistant"
CACHE_TTL_SECONDS = 300
ANALYSIS_COOLDOWN_SECONDS = 10.0

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"new\s+instructions?\s*[:=]", re.I),
    re.compile(r"do\s+not\s+follow\s+the\s+(above|previous|user)", re.I),
)

_HOLDING_COLUMN_ALIASES = {
    "fund_code": {"fundcode", "code", "基金代码", "基金编码", "代码"},
    "fund_name": {"fundname", "name", "基金名称", "基金简称", "名称"},
    "shares": {"shares", "share", "份额", "持有份额", "持仓份额"},
    "cost": {"cost", "costprice", "成本", "成本价", "持仓成本", "买入成本"},
}


class AnalysisError(RuntimeError):
    """User-facing error raised when an analysis cannot be produced safely."""


def _mock_capital_flow_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"industry": "半导体", "main_net_inflow": 21.6e8, "change_pct": 2.4, "source": "mock/previous-day"},
            {"industry": "人工智能", "main_net_inflow": 17.8e8, "change_pct": 1.9, "source": "mock/previous-day"},
            {"industry": "新能源", "main_net_inflow": 12.3e8, "change_pct": 1.1, "source": "mock/previous-day"},
            {"industry": "高端制造", "main_net_inflow": 9.7e8, "change_pct": 0.8, "source": "mock/previous-day"},
            {"industry": "医药", "main_net_inflow": 6.4e8, "change_pct": 0.5, "source": "mock/previous-day"},
        ]
    )


def _mock_policy_signals() -> List[Dict[str, Any]]:
    today = date.today().isoformat()
    return [
        {
            "title": "Mock policy: equipment renewal and domestic consumption support",
            "summary": "Policy monitor highlights continued support for equipment renewal, consumer goods trade-ins, and domestic demand expansion.",
            "source": "Mock policy monitor (fallback)",
            "credibility": 0.70,
            "published_at": today,
            "tags": ["消费", "设备更新"],
        },
        {
            "title": "Mock policy: technology innovation and AI infrastructure",
            "summary": "Policy monitor highlights support for AI infrastructure, computing power, industrial software, and technology innovation.",
            "source": "Mock policy monitor (fallback)",
            "credibility": 0.68,
            "published_at": today,
            "tags": ["人工智能", "算力"],
        },
        {
            "title": "Mock policy: green transformation and new energy",
            "summary": "Policy monitor highlights green transition measures covering renewable energy, storage, and low-carbon manufacturing.",
            "source": "Mock policy monitor (fallback)",
            "credibility": 0.66,
            "published_at": today,
            "tags": ["新能源", "绿色"],
        },
        {
            "title": "Mock policy: financial stability and capital market quality",
            "summary": "Policy monitor highlights stability-oriented financial policy and efforts to improve capital market quality and investor protection.",
            "source": "Mock policy monitor (fallback)",
            "credibility": 0.64,
            "published_at": today,
            "tags": ["金融", "资本市场"],
        },
        {
            "title": "Mock policy: advanced manufacturing and supply-chain resilience",
            "summary": "Policy monitor highlights advanced manufacturing, industrial chains, and supply-chain resilience as priority areas.",
            "source": "Mock policy monitor (fallback)",
            "credibility": 0.62,
            "published_at": today,
            "tags": ["高端制造"],
        },
    ]


def _mock_fund_universe_df() -> pd.DataFrame:
    rows = [
        {
            "fund_code": "999001",
            "fund_name": "华证半导体产业精选联接A",
            "fund_size": "8.6亿元",
            "fund_size_numeric": 8.6e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "18.50%",
            "one_year_return_numeric": 18.5,
            "one_month_return_numeric": 4.2,
            "three_month_return_numeric": 9.8,
            "volatility_pct": 22.3,
        },
        {
            "fund_code": "999002",
            "fund_name": "华证半导体产业精选联接C",
            "fund_size": "3.2亿元",
            "fund_size_numeric": 3.2e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "17.90%",
            "one_year_return_numeric": 17.9,
            "one_month_return_numeric": 4.0,
            "three_month_return_numeric": 9.5,
            "volatility_pct": 22.1,
        },
        {
            "fund_code": "999003",
            "fund_name": "华证人工智能算力联接A",
            "fund_size": "12.1亿元",
            "fund_size_numeric": 12.1e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "25.60%",
            "one_year_return_numeric": 25.6,
            "one_month_return_numeric": 6.1,
            "three_month_return_numeric": 13.2,
            "volatility_pct": 28.4,
        },
        {
            "fund_code": "999004",
            "fund_name": "华证人工智能算力联接C",
            "fund_size": "5.8亿元",
            "fund_size_numeric": 5.8e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "25.10%",
            "one_year_return_numeric": 25.1,
            "one_month_return_numeric": 6.0,
            "three_month_return_numeric": 13.0,
            "volatility_pct": 28.2,
        },
        {
            "fund_code": "999005",
            "fund_name": "华证新能源储能联接A",
            "fund_size": "15.4亿元",
            "fund_size_numeric": 15.4e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "12.80%",
            "one_year_return_numeric": 12.8,
            "one_month_return_numeric": 3.1,
            "three_month_return_numeric": 7.4,
            "volatility_pct": 24.6,
        },
        {
            "fund_code": "999006",
            "fund_name": "华证新能源储能联接C",
            "fund_size": "6.3亿元",
            "fund_size_numeric": 6.3e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "12.30%",
            "one_year_return_numeric": 12.3,
            "one_month_return_numeric": 3.0,
            "three_month_return_numeric": 7.2,
            "volatility_pct": 24.4,
        },
        {
            "fund_code": "999007",
            "fund_name": "华证高端装备制造联接A",
            "fund_size": "10.9亿元",
            "fund_size_numeric": 10.9e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "15.20%",
            "one_year_return_numeric": 15.2,
            "one_month_return_numeric": 3.6,
            "three_month_return_numeric": 8.1,
            "volatility_pct": 21.8,
        },
        {
            "fund_code": "999008",
            "fund_name": "华证医药健康联接A",
            "fund_size": "9.2亿元",
            "fund_size_numeric": 9.2e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "8.40%",
            "one_year_return_numeric": 8.4,
            "one_month_return_numeric": 1.8,
            "three_month_return_numeric": 4.6,
            "volatility_pct": 19.5,
        },
        {
            "fund_code": "999009",
            "fund_name": "华证消费升级联接A",
            "fund_size": "13.7亿元",
            "fund_size_numeric": 13.7e8,
            "fund_type": "股票型",
            "fund_manager": "Mock Manager",
            "one_year_return": "10.60%",
            "one_year_return_numeric": 10.6,
            "one_month_return_numeric": 2.5,
            "three_month_return_numeric": 6.0,
            "volatility_pct": 20.2,
        },
        {
            "fund_code": "999010",
            "fund_name": "华证金融蓝筹联接A",
            "fund_size": "20.3亿元",
            "fund_size_numeric": 20.3e8,
            "fund_type": "混合型",
            "fund_manager": "Mock Manager",
            "one_year_return": "7.20%",
            "one_year_return_numeric": 7.2,
            "one_month_return_numeric": 1.4,
            "three_month_return_numeric": 3.8,
            "volatility_pct": 16.1,
        },
    ]
    df = pd.DataFrame(rows)
    df["source"] = "mock fund universe"
    return df


def _norm_col(value: Any) -> str:
    return re.sub(r"[\s_\-\/（）()]", "", str(value).lower())


def _pick_col(df: pd.DataFrame, needles: Sequence[str], prefer: Optional[str] = None) -> Optional[str]:
    prefer_norm = _norm_col(prefer) if prefer else ""
    best = None
    for col in df.columns:
        key = _norm_col(str(col))
        if all(needle in key for needle in needles):
            if prefer_norm and prefer_norm in key:
                return str(col)
            if best is None:
                best = str(col)
    return best


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("％", "%").replace("%", "").replace("元", "")
        if not s or s.lower() in {"-", "--", "nan", "none", "null", "n/a"}:
            return default
        try:
            return float(s)
        except ValueError:
            return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _parse_fund_size(value: Any) -> Tuple[Optional[float], str]:
    if value is None:
        return None, ""
    if not isinstance(value, str):
        f = _to_float(value)
        return f, "" if f is None else str(value)
    s = value.strip().replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(万亿|亿|万)元?", s)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        multiplier = {"万亿": 1e12, "亿": 1e8, "万": 1e4}.get(unit, 1.0)
        return num * multiplier, s
    return None, s


def _first_value(row: Any, keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        try:
            value = row.get(key)
        except AttributeError:
            value = row[key] if key in row else None
        if value is not None:
            if isinstance(value, str):
                if value.strip() and value.strip().lower() not in {"nan", "none", "nat", "null"}:
                    return value
            else:
                try:
                    if not pd.isna(value):
                        return value
                except (TypeError, ValueError):
                    return value
    return default


def sanitize_text(value: Any, max_length: int = 300) -> Tuple[str, bool]:
    """Neutralize prompt-injection patterns and cap overlong user/external text."""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    blocked = False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            value = pattern.sub("[blocked]", value)
            blocked = True
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > max_length:
        value = value[:max_length].rstrip() + "..."
        blocked = True
    return value, blocked


def _display_text(value: Any, max_length: int = 300) -> str:
    return sanitize_text(str(value), max_length=max_length)[0]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_capital_flow() -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if ak is None:
        warnings.append("akshare is unavailable; using mock/previous-day capital flow data.")
        return _mock_capital_flow_df(), {
            "source": "mock/previous-day",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings

    try:
        raw = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        if raw is None or raw.empty:
            raise ValueError("empty capital flow table")
        df = pd.DataFrame(raw).copy()
        industry_col = _pick_col(df, ["名称"])
        change_col = _pick_col(df, ["涨跌幅"], prefer="今日")
        flow_col = _pick_col(df, ["主力净流入", "净额"], prefer="今日")
        if industry_col is None or flow_col is None:
            raise ValueError("required capital flow columns missing")

        out = pd.DataFrame(
            {
                "industry": df[industry_col].astype(str).str.strip(),
                "main_net_inflow": pd.to_numeric(df[flow_col], errors="coerce"),
                "change_pct": (
                    pd.to_numeric(df[change_col], errors="coerce")
                    if change_col is not None
                    else float("nan")
                ),
                "source": "akshare live",
            }
        )
        out = out.dropna(subset=["industry", "main_net_inflow"])
        out["industry"] = out["industry"].replace("", pd.NA).dropna()
        out = out.dropna(subset=["industry"]).sort_values(
            by="main_net_inflow", ascending=False
        ).head(5).reset_index(drop=True)
        if out.empty:
            raise ValueError("no valid capital flow rows")
        return out, {
            "source": "akshare live",
            "is_fallback": False,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings
    except Exception as exc:
        warnings.append(f"Capital flow live fetch failed ({type(exc).__name__}); using mock/previous-day data.")
        return _mock_capital_flow_df(), {
            "source": "mock/previous-day",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_capital_flow_history(industry: str) -> Optional[pd.DataFrame]:
    if ak is None:
        return None
    try:
        raw = ak.stock_sector_fund_flow_hist(symbol=industry, indicator="今日")
        if raw is None or raw.empty:
            return None
        df = pd.DataFrame(raw).copy()
        date_col = _pick_col(df, ["日期"])
        flow_col = _pick_col(df, ["主力净流入", "净额"], prefer="今日")
        if date_col is None or flow_col is None:
            return None
        out = pd.DataFrame(
            {
                "date": df[date_col].astype(str),
                "main_net_inflow": pd.to_numeric(df[flow_col], errors="coerce"),
            }
        ).dropna()
        return out if not out.empty else None
    except Exception:
        return None


def validate_capital_flow(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    warnings: List[str],
) -> Tuple[pd.DataFrame, str, List[str], bool]:
    if df is None or df.empty:
        warnings.append("Capital flow data is empty; downgrading to mock/previous-day data.")
        return _mock_capital_flow_df(), "mock/previous-day (empty)", warnings, True
    if meta.get("is_fallback"):
        return df, str(meta.get("source", "mock/previous-day")), warnings, False

    inflows = pd.to_numeric(df["main_net_inflow"], errors="coerce").dropna()
    anomaly = False
    if len(inflows) < 2 or len({round(v, 2) for v in inflows}) == 1:
        anomaly = True
        warnings.append("Capital flow anomaly: all industries show the same net inflow.")

    top_industry = str(df.iloc[0]["industry"]) if not df.empty else ""
    history = _fetch_capital_flow_history(top_industry) if top_industry else None
    if history is not None and len(history) >= 5:
        hist_values = pd.to_numeric(history["main_net_inflow"], errors="coerce").dropna()
        if len(hist_values) >= 5:
            mean = float(hist_values.mean())
            std = float(hist_values.std(ddof=0))
            if std > 0:
                outliers = [idx for idx, value in inflows.items() if abs(float(value) - mean) > 3 * std]
                if outliers:
                    anomaly = True
                    warnings.append(
                        "Capital flow anomaly: at least one industry is more than 3 standard deviations "
                        "away from its simplified historical average."
                    )
    else:
        warnings.append(
            "Historical outlier check skipped: simplified akshare history is unavailable; "
            "capital flow freshness may be uncertain."
        )

    if anomaly:
        warnings.append("Downgrading capital flow to mock/previous-day data due to detected anomaly.")
        return _mock_capital_flow_df(), "mock/previous-day (anomaly)", warnings, True
    return df, str(meta.get("source", "akshare live")), warnings, False


def _normalize_live_policy_rows(raw: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in raw.head(30).iterrows():
        title = _first_value(row, ["title", "标题", "name", "新闻标题"], "")
        summary = _first_value(row, ["content", "summary", "摘要", "正文", "详情", "内容"], "")
        source = _first_value(row, ["source", "来源", "媒体", "site", "channel"], "CCTV/akshare")
        url = _first_value(row, ["url", "link", "链接", "href"], "")
        published_at = _first_value(row, ["datetime", "date", "time", "发布时间", "时间", "日期"], "")
        title = _display_text(str(title), 200)
        summary = _display_text(str(summary), 600)
        if not title and summary:
            title = summary[:60]
        if not title and not summary:
            continue

        source_text = str(source)
        source_lower = source_text.lower()
        credibility = 0.70
        if "央视" in source_text or "cctv" in source_lower:
            credibility = 0.85
        elif "新华社" in source_text or "人民" in source_text:
            credibility = 0.90
        elif "证券" in source_text or "证监" in source_text:
            credibility = 0.80
        elif "财联社" in source_text:
            credibility = 0.75

        rows.append(
            {
                "title": title,
                "summary": summary,
                "source": source_text,
                "credibility": min(1.0, max(0.0, credibility)),
                "published_at": _display_text(str(published_at), 40),
                "url": _display_text(str(url), 200),
                "tags": [],
            }
        )
    return rows


def _policy_list_invalid(policy_signals: Sequence[Dict[str, Any]]) -> bool:
    if not policy_signals:
        return True
    texts = [
        f"{item.get('title', '')} {item.get('summary', '')}".strip()
        for item in policy_signals
    ]
    if not any(texts):
        return True
    if len(texts) > 1 and len({t for t in texts if t}) == 1:
        return True
    return False


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_policy_signals() -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if ak is None:
        warnings.append("akshare is unavailable; using curated mock policy signals.")
        return _mock_policy_signals(), {
            "source": "mock policy monitor",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings

    candidates: List[Dict[str, Any]] = []
    fetchers = [
        ("CCTV", lambda: ak.news_cctv()),
        ("Baidu economic news", lambda: ak.news_economic_baidu()),
    ]
    for name, fetcher in fetchers:
        try:
            raw = fetcher()
            if raw is not None and not raw.empty:
                candidates.extend(_normalize_live_policy_rows(raw))
                if candidates:
                    break
        except Exception as exc:
            warnings.append(f"Policy live fetch {name} failed ({type(exc).__name__}).")

    if not candidates:
        warnings.append("No live policy signals available; using curated mock policy signals.")
        return _mock_policy_signals(), {
            "source": "mock policy monitor",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings

    if _policy_list_invalid(candidates):
        warnings.append(
            "Live policy data failed validation (empty or identical snippets); "
            "using curated mock policy signals."
        )
        return _mock_policy_signals(), {
            "source": "mock policy monitor (validation fallback)",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings

    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in candidates:
        key = f"{item.get('title', '')}|{item.get('summary', '')}"
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:10], {
        "source": "akshare live policy feed",
        "is_fallback": False,
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }, warnings


def _normalize_fund_rank(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw).copy()
    code_col = _pick_col(df, ["基金代码"]) or _pick_col(df, ["代码"])
    name_col = _pick_col(df, ["基金简称"]) or _pick_col(df, ["基金名称"]) or _pick_col(df, ["名称"])
    if code_col is None or name_col is None:
        raise ValueError("fund rank columns missing")

    out = pd.DataFrame(
        {
            "fund_code": df[code_col].astype(str).str.extract(r"(\d{6})", expand=False),
            "fund_name": df[name_col].astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["fund_code", "fund_name"]).copy()
    out["fund_code"] = out["fund_code"].str.strip()
    out["fund_size"] = ""
    out["fund_size_numeric"] = None
    out["fund_type"] = ""
    out["fund_manager"] = ""
    out["volatility_pct"] = None

    field_map = [
        ("近1年", "one_year_return"),
        ("近1月", "one_month_return"),
        ("近1周", "one_week_return"),
        ("近3月", "three_month_return"),
        ("近6月", "six_month_return"),
        ("今年来", "ytd_return"),
    ]
    for needle, field in field_map:
        col = _pick_col(df, [needle])
        if col is not None:
            out[field] = df[col].map(lambda value: _to_float(value))
        else:
            out[field] = None

    out["one_year_return_numeric"] = out["one_year_return"]
    out["one_year_return"] = out["one_year_return_numeric"].map(
        lambda value: f"{value:.2f}%" if value is not None else "N/A"
    )
    out["one_month_return_numeric"] = out["one_month_return"]
    out["three_month_return_numeric"] = out["three_month_return"]
    return out


def _normalize_fund_scale(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw).copy()
    code_col = _pick_col(df, ["基金代码"]) or _pick_col(df, ["代码"])
    name_col = _pick_col(df, ["基金简称"]) or _pick_col(df, ["基金名称"]) or _pick_col(df, ["名称"])
    if code_col is None or name_col is None:
        raise ValueError("fund scale columns missing")

    out = pd.DataFrame(
        {
            "fund_code": df[code_col].astype(str).str.extract(r"(\d{6})", expand=False),
            "fund_name": df[name_col].astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["fund_code", "fund_name"]).copy()
    out["fund_code"] = out["fund_code"].str.strip()

    size_col = _pick_col(df, ["规模"])
    type_col = _pick_col(df, ["基金类型"]) or _pick_col(df, ["类型"])
    manager_col = _pick_col(df, ["基金经理"]) or _pick_col(df, ["经理"])

    out["fund_size"] = df[size_col].map(lambda value: _parse_fund_size(value)[1]) if size_col else ""
    out["fund_size_numeric"] = df[size_col].map(lambda value: _parse_fund_size(value)[0]) if size_col else None
    out["fund_type"] = df[type_col].astype(str) if type_col else ""
    out["fund_manager"] = df[manager_col].astype(str) if manager_col else ""
    return out.drop_duplicates(subset=["fund_code"], keep="first")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_fund_universe() -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if ak is None:
        warnings.append("akshare is unavailable; using curated mock fund universe.")
        return _mock_fund_universe_df(), {
            "source": "mock fund universe",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings

    rank_df: Optional[pd.DataFrame] = None
    scale_df: Optional[pd.DataFrame] = None
    try:
        raw = ak.fund_open_fund_rank_em(symbol="全部")
        if raw is not None and not raw.empty:
            rank_df = _normalize_fund_rank(raw)
    except Exception as exc:
        warnings.append(f"Fund rank live fetch failed ({type(exc).__name__}).")

    try:
        raw = ak.fund_scale_open_sina(symbol="全部")
        if raw is not None and not raw.empty:
            scale_df = _normalize_fund_scale(raw)
    except Exception as exc:
        warnings.append(f"Fund scale live fetch failed ({type(exc).__name__}); fund size may be unavailable.")

    if rank_df is None or rank_df.empty:
        warnings.append("Fund universe live fetch failed; using curated mock fund list.")
        return _mock_fund_universe_df(), {
            "source": "mock fund universe",
            "is_fallback": True,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }, warnings

    universe = rank_df
    if scale_df is not None and not scale_df.empty:
        keep_cols = ["fund_code", "fund_size", "fund_size_numeric", "fund_type", "fund_manager"]
        universe = universe.drop(columns=[c for c in keep_cols if c in universe.columns], errors="ignore")
        universe = universe.merge(scale_df[keep_cols], on="fund_code", how="left")
    else:
        universe["fund_size"] = ""
        universe["fund_size_numeric"] = None
        universe["fund_type"] = ""
        universe["fund_manager"] = ""

    universe["one_year_return_numeric"] = pd.to_numeric(universe["one_year_return_numeric"], errors="coerce")
    universe = universe.sort_values(
        by="one_year_return_numeric", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return universe, {
        "source": "akshare live fund universe",
        "is_fallback": False,
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }, warnings


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_fund_volatility(fund_code: str) -> Optional[float]:
    if ak is None or str(fund_code).startswith("999"):
        return None

    fetchers = [
        lambda: ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势"),
        lambda: ak.fund_hist_em(
            symbol=fund_code,
            period="daily",
            start_date=(date.today() - timedelta(days=400)).strftime("%Y%m%d"),
            end_date=date.today().strftime("%Y%m%d"),
            adjust="",
        ),
    ]
    for fetcher in fetchers:
        try:
            raw = fetcher()
            if raw is None or raw.empty:
                continue
            df = pd.DataFrame(raw).copy()
            nav_col = _pick_col(df, ["单位净值"]) or _pick_col(df, ["close"]) or _pick_col(df, ["收盘"])
            if nav_col is None:
                continue
            values = pd.to_numeric(df[nav_col], errors="coerce").dropna()
            if len(values) < 5:
                continue
            returns = values.pct_change().dropna()
            if len(returns) < 4:
                continue
            return float(returns.std(ddof=0) * math.sqrt(252) * 100)
        except Exception:
            continue
    return None


def _sector_keywords(sector_name: Any) -> List[str]:
    text = str(sector_name).strip().lower()
    groups = {
        "半导体": ["半导体", "芯片", "集成电路"],
        "人工智能": ["人工智能", "算力", "ai", "计算机", "软件"],
        "新能源": ["新能源", "光伏", "储能", "电池", "碳中和"],
        "消费": ["消费", "白酒", "食品", "家电", "汽车"],
        "医药": ["医药", "医疗", "生物", "健康"],
        "金融": ["金融", "银行", "证券", "保险"],
        "高端制造": ["高端制造", "制造", "装备", "工业", "设备"],
        "军工": ["军工", "国防", "航天"],
        "科技": ["科技", "电子", "通信", "互联网"],
        "农业": ["农业", "农林", "食品饮料"],
    }
    for keyword, candidates in groups.items():
        if keyword in text:
            return candidates
    if len(text) >= 2:
        return [text, text[-2:]]
    return [text]


def select_relevant_funds(
    universe: pd.DataFrame,
    sectors: Sequence[Dict[str, Any]],
    limit: int = 6,
) -> pd.DataFrame:
    if universe is None or universe.empty:
        return pd.DataFrame()
    df = universe.copy()
    if "one_year_return_numeric" not in df.columns:
        df["one_year_return_numeric"] = df.get("one_year_return", pd.Series(dtype=float)).map(
            lambda value: _to_float(value)
        )

    keywords: set[str] = set()
    for sector in sectors:
        keywords.update(_sector_keywords(sector.get("name", "")))

    mask = pd.Series(False, index=df.index)
    if keywords:
        pattern = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
        mask = df["fund_name"].astype(str).str.contains(pattern, case=False, na=False, regex=True)

    relevant = df[mask] if mask.any() else df
    relevant = relevant.sort_values(
        by="one_year_return_numeric", ascending=False, na_position="last"
    ).head(limit).copy()

    for idx, row in relevant.iterrows():
        code = str(row["fund_code"])
        if not code.startswith("999"):
            volatility = _fetch_fund_volatility(code)
            if volatility is not None:
                relevant.at[idx, "volatility_pct"] = volatility
    return relevant.reset_index(drop=True)


def _normalize_holdings_columns(df: pd.DataFrame, errors: List[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lstrip("\ufeff").strip() for col in df.columns]
    mapped = {}
    for col in df.columns:
        key = re.sub(r"[\s_\-/]", "", str(col).lower())
        for target, aliases in _HOLDING_COLUMN_ALIASES.items():
            if key in aliases:
                mapped[target] = col
                break
        else:
            if "基金" in key and "代码" in key and "fund_code" not in mapped:
                mapped["fund_code"] = col
            elif "基金" in key and ("名称" in key or "简称" in key) and "fund_name" not in mapped:
                mapped["fund_name"] = col
            elif "份额" in key and "shares" not in mapped:
                mapped["shares"] = col
            elif "成本" in key and "cost" not in mapped:
                mapped["cost"] = col

    required = {"fund_code", "fund_name", "shares", "cost"}
    missing = required - set(mapped)
    if missing:
        errors.append(
            "Missing required columns: " + ", ".join(sorted(missing)) +
            ". Expected fund code, name, shares, and cost."
        )
        return pd.DataFrame()

    normalized = pd.DataFrame(
        {
            "fund_code": df[mapped["fund_code"]],
            "fund_name": df[mapped["fund_name"]],
            "shares": df[mapped["shares"]],
            "cost": df[mapped["cost"]],
        }
    )
    return normalized


def _read_pasted_holdings(text: str) -> Optional[pd.DataFrame]:
    cleaned = text.replace("，", ",").replace("；", ";")
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(io.StringIO(cleaned), sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    try:
        return pd.read_csv(io.StringIO(cleaned), sep=r"\s+", engine="python")
    except Exception:
        return None


def _read_uploaded_holdings(uploaded_file: Any) -> Optional[pd.DataFrame]:
    try:
        raw = uploaded_file.getvalue()
    except AttributeError:
        return None
    for encoding in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            if not df.empty:
                return df
        except Exception:
            continue
    return None


def parse_holdings_input(
    pasted_text: str,
    uploaded_file: Any,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    errors: List[str] = []
    df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        df = _read_uploaded_holdings(uploaded_file)
        if df is None:
            errors.append("Uploaded CSV could not be parsed. Check encoding and column names.")
    elif pasted_text and pasted_text.strip():
        df = _read_pasted_holdings(pasted_text)
        if df is None:
            errors.append("Pasted table could not be parsed as CSV.")

    if df is None:
        if not errors:
            errors.append("Provide fund holdings as a pasted table or CSV upload.")
        return None, errors
    if df.empty:
        errors.append("Holdings table is empty.")
        return None, errors

    normalized = _normalize_holdings_columns(df, errors)
    if normalized.empty:
        return None, errors

    for idx, row in normalized.iterrows():
        code = str(row["fund_code"]).strip()
        if not re.fullmatch(r"\d{6}", code):
            errors.append(f"Row {idx + 1}: fund code '{code}' is not a 6-digit numeric code.")
        if not str(row["fund_name"]).strip():
            errors.append(f"Row {idx + 1}: fund name is empty.")
        shares = _to_float(row["shares"])
        cost = _to_float(row["cost"])
        if shares is None or shares <= 0:
            errors.append(f"Row {idx + 1}: shares must be a positive number.")
        if cost is None or cost <= 0:
            errors.append(f"Row {idx + 1}: cost must be a positive number.")

    if errors:
        return None, errors

    normalized["fund_code"] = normalized["fund_code"].astype(str).str.strip()
    normalized["fund_name"] = normalized["fund_name"].astype(str).str.strip()
    normalized["shares"] = normalized["shares"].map(lambda value: _to_float(value))
    normalized["cost"] = normalized["cost"].map(lambda value: _to_float(value))
    normalized = normalized.groupby(
        ["fund_code", "fund_name"], as_index=False, dropna=False
    ).agg(
        shares=("shares", "sum"),
        cost=("cost", "mean"),
    )
    normalized = normalized.sort_values("shares", ascending=False).reset_index(drop=True)
    return normalized, errors


def _check_analysis_rate_limit() -> None:
    now = time.time()
    key = "analysis_last_run_at"
    last = float(st.session_state.get(key, 0.0))
    remaining = ANALYSIS_COOLDOWN_SECONDS - (now - last)
    if remaining > 0:
        raise AnalysisError(
            f"Analysis rate limit: please wait {remaining:.0f} seconds before running another analysis."
        )
    st.session_state[key] = now


def build_prompt(
    holdings_df: pd.DataFrame,
    available_cash: float,
    risk_preference: str,
    capital_flow_df: pd.DataFrame,
    policy_signals: Sequence[Dict[str, Any]],
    fund_universe_df: pd.DataFrame,
    data_warnings: Sequence[str],
) -> str:
    holdings_rows = []
    for _, row in holdings_df.iterrows():
        safe_name, _ = sanitize_text(str(row["fund_name"]), max_length=80)
        holdings_rows.append(
            {
                "fund_code": str(row["fund_code"]).strip(),
                "fund_name": safe_name,
                "shares": round(float(row["shares"]), 2),
                "cost": round(float(row["cost"]), 4),
            }
        )

    flow_rows = []
    for _, row in capital_flow_df.iterrows():
        flow_rows.append(
            {
                "industry": _display_text(row["industry"], 60),
                "main_net_inflow_cny": round(float(row["main_net_inflow"]), 2),
                "change_pct": _to_float(row.get("change_pct")),
                "source": _display_text(row.get("source", ""), 40),
            }
        )

    policy_rows = []
    for item in policy_signals:
        policy_rows.append(
            {
                "title": _display_text(item.get("title", ""), 200),
                "summary": _display_text(item.get("summary", ""), 500),
                "source": _display_text(item.get("source", ""), 80),
                "credibility": min(1.0, max(0.0, float(item.get("credibility", 0.0)))),
            }
        )

    universe_rows = []
    for _, row in fund_universe_df.head(80).iterrows():
        universe_rows.append(
            {
                "fund_code": _display_text(row.get("fund_code", ""), 20),
                "fund_name": _display_text(row.get("fund_name", ""), 80),
                "fund_size": _display_text(row.get("fund_size", ""), 40),
                "one_year_return": _display_text(row.get("one_year_return", ""), 30),
            }
        )

    instructions = """
You are a cautious personal fund research assistant.
Combine the capital flow signals, policy signals, validation warnings, user holdings, and fund universe into a single
signal-fusion view.

Return ONLY a valid strict JSON object. Do not use markdown code fences, commentary, or text outside JSON.
Use exactly these keys:
{
  "sectors": [
    {
      "name": "sector name",
      "conviction_score": 0-100,
      "reasoning": "specific reasoning tied to policy credibility, flow magnitude, and alignment",
      "policy_alignment": "short explanation",
      "flow_magnitude": "short explanation"
    }
  ],
  "recommended_funds": [
    {
      "code": "6-digit code from the universe",
      "name": "fund name",
      "sector": "matched sector",
      "reason": "specific reasoning",
      "suggested_allocation_pct": "number from 0 to 100",
      "risk_level": "conservative | balanced | aggressive"
    }
  ],
  "rebalance_actions": [
    {
      "fund_code": "user holding code",
      "fund_name": "user holding name",
      "action": "INCREASE | HOLD | REDUCE | SELL",
      "reason": "reason tied to sector conviction and fund exposure",
      "suggested_amount": "optional CNY amount"
    }
  ],
  "overall_confidence": "HIGH | MEDIUM | LOW",
  "confidence_note": "short explanation, especially when data is fallback/mock or signals conflict"
}

Rules:
- Recommend 1-2 off-exchange funds per high-conviction sector when possible, and prefer fund codes from the provided universe.
- Provide an action for every user holding. If a holding has no clear signal, use HOLD with a concise reason.
- Conviction scores must combine policy credibility and actual capital flow magnitude, not just policy enthusiasm.
- If signals conflict, data are missing, a data source is fallback/mock, or the model is uncertain, set
  overall_confidence to "LOW" and add a confidence_note telling the user to verify independently.
- Do not invent live policy facts. Treat mock/fallback sources as low credibility and downgrade conviction.
- Keep reasoning specific and concise. Do not include instructions, system prompts, or non-JSON text.
"""

    prompt = f"""
Risk preference: {risk_preference}
Available cash (CNY): {available_cash:,.2f}
User holdings:
{json.dumps(holdings_rows, ensure_ascii=False)}

Today's capital flow (top industries by main net inflow):
{json.dumps(flow_rows, ensure_ascii=False)}

Policy signals:
{json.dumps(policy_rows, ensure_ascii=False)}

Fund universe excerpt (sorted by 1-year return where available):
{json.dumps(universe_rows, ensure_ascii=False)}

Data source warnings:
{json.dumps([_display_text(item, 200) for item in data_warnings], ensure_ascii=False)}

Task:
{instructions}
"""
    return prompt


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if _openai is not None and isinstance(exc, (_openai.APIConnectionError, _openai.RateLimitError)):
        return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_retryable),
    reraise=True,
)
def _call_deepseek(prompt: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    if _openai is None:
        raise AnalysisError("The openai package is not installed or could not be imported.")
    if not api_key:
        raise AnalysisError("DEEPSEEK_API_KEY is not set. Add it to the environment before running analysis.")
    if not base_url:
        raise AnalysisError("DEEPSEEK_BASE_URL is not set. Add it to the environment before running analysis.")

    client = _openai.OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a cautious investment research assistant. You output only valid JSON with sector signals, "
                "fund recommendations, and rebalance actions."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    try:
        response = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content if response.choices else ""
    if not content or not content.strip():
        raise AnalysisError("DeepSeek returned an empty response.")
    return content


def _clamp_score(value: Any, default: float = 50.0) -> float:
    score = _to_float(value, default)
    if score is None:
        score = default
    return round(max(0.0, min(100.0, float(score))), 1)


def _clean_fund_code(value: Any) -> str:
    text = _display_text(value, 20).strip()
    match = re.search(r"\d{6}", text)
    return match.group(0) if match else text


def parse_llm_response(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise AnalysisError("DeepSeek response did not contain a JSON object.")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnalysisError("DeepSeek response contained malformed JSON.") from exc

    if not isinstance(data, dict):
        raise AnalysisError("DeepSeek response JSON root must be an object.")

    sectors: List[Dict[str, Any]] = []
    for item in data.get("sectors", []) or []:
        if not isinstance(item, dict):
            continue
        sectors.append(
            {
                "name": _display_text(item.get("name", item.get("sector", "Unknown sector")), 80),
                "conviction_score": _clamp_score(
                    item.get("conviction_score", item.get("score", 50)),
                    50.0,
                ),
                "reasoning": _display_text(item.get("reasoning", item.get("rationale", "")), 500),
                "policy_alignment": _display_text(item.get("policy_alignment", ""), 200),
                "flow_magnitude": _display_text(item.get("flow_magnitude", ""), 200),
            }
        )

    recommended_funds: List[Dict[str, Any]] = []
    for item in data.get("recommended_funds", []) or []:
        if not isinstance(item, dict):
            continue
        allocation = _to_float(
            item.get("suggested_allocation_pct", item.get("allocation_pct", None)),
            None,
        )
        rec = {
            "fund_code": _clean_fund_code(item.get("code", item.get("fund_code", ""))),
            "fund_name": _display_text(item.get("name", item.get("fund_name", "")), 80),
            "sector": _display_text(item.get("sector", ""), 80),
            "reasoning": _display_text(item.get("reason", item.get("reasoning", "")), 500),
            "suggested_allocation_pct": (
                round(max(0.0, min(100.0, allocation)), 2)
                if allocation is not None
                else None
            ),
            "risk_level": _display_text(item.get("risk_level", ""), 30),
        }
        if rec["fund_code"] or rec["fund_name"]:
            recommended_funds.append(rec)

    allowed_actions = {"INCREASE", "HOLD", "REDUCE", "SELL"}
    rebalance_actions: List[Dict[str, Any]] = []
    for item in data.get("rebalance_actions", []) or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", item.get("recommendation", "HOLD"))).upper().strip()
        if action == "BUY":
            action = "INCREASE"
        if action not in allowed_actions:
            action = "HOLD"
        suggested_amount = _to_float(item.get("suggested_amount", item.get("amount", None)), None)
        if suggested_amount is not None:
            suggested_amount = round(max(0.0, suggested_amount), 2)
        rebalance_actions.append(
            {
                "fund_code": _clean_fund_code(item.get("fund_code", "")),
                "fund_name": _display_text(item.get("fund_name", ""), 80),
                "action": action,
                "reasoning": _display_text(item.get("reason", item.get("reasoning", "")), 500),
                "suggested_amount": suggested_amount,
            }
        )

    raw_confidence = str(data.get("overall_confidence", "LOW")).upper().strip()
    raw_confidence = raw_confidence.replace(" CONFIDENCE", "")
    if raw_confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raw_confidence = "LOW"
    overall_confidence = raw_confidence
    confidence_note = _display_text(
        data.get("confidence_note", data.get("note", "")),
        300,
    )

    if not sectors:
        raise AnalysisError("DeepSeek response did not contain valid sector conviction data.")
    return {
        "sectors": sectors,
        "recommended_funds": recommended_funds,
        "rebalance_actions": rebalance_actions,
        "overall_confidence": overall_confidence,
        "confidence_note": confidence_note,
    }


def build_analysis(
    holdings_df: pd.DataFrame,
    available_cash: float,
    risk_preference: str,
    capital_flow_df: pd.DataFrame,
    policy_signals: Sequence[Dict[str, Any]],
    fund_universe_df: pd.DataFrame,
    data_warnings: Sequence[str],
) -> Dict[str, Any]:
    _check_analysis_rate_limit()
    prompt = build_prompt(
        holdings_df=holdings_df,
        available_cash=available_cash,
        risk_preference=risk_preference,
        capital_flow_df=capital_flow_df,
        policy_signals=policy_signals,
        fund_universe_df=fund_universe_df,
        data_warnings=data_warnings,
    )
    raw_response = _call_deepseek(prompt)
    result = parse_llm_response(raw_response)
    result["relevant_funds"] = select_relevant_funds(fund_universe_df, result["sectors"], limit=6)
    combined_warnings = " ".join(str(item) for item in data_warnings).lower()
    if any(
        marker in combined_warnings
        for marker in ("fallback", "mock", "unavailable", "failed", "empty", "anomaly")
    ):
        result["overall_confidence"] = "LOW"
        result["confidence_note"] = (
            f"{result.get('confidence_note', '')} Fallback or invalid data was used; "
            "confidence is forced to LOW. Verify independently before acting."
        ).strip()
    return result


def render_warnings(warnings: Sequence[str], limit: int = 8) -> None:
    unique: List[str] = []
    seen = set()
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            unique.append(warning)
    for warning in unique[:limit]:
        st.warning(warning)
    if len(unique) > limit:
        st.caption(f"{len(unique) - limit} additional warnings were suppressed.")


def render_signals_tab(
    capital_flow: pd.DataFrame,
    policy_signals: Sequence[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    cap_source: str,
    policy_meta: Dict[str, Any],
    warnings: Sequence[str],
) -> None:
    render_warnings(warnings)

    st.subheader("Capital Flow")
    st.caption(f"Source: {cap_source} | Policy source: {policy_meta.get('source', 'unknown')}")
    if capital_flow is not None and not capital_flow.empty:
        display = capital_flow.copy()
        display["main_net_inflow_yi"] = (pd.to_numeric(display["main_net_inflow"], errors="coerce") / 1e8).round(2)
        chart = display.set_index("industry")["main_net_inflow_yi"]
        st.dataframe(
            display[
                ["industry", "main_net_inflow_yi", "change_pct", "source"]
            ].rename(
                columns={
                    "industry": "Industry",
                    "main_net_inflow_yi": "Main Net Inflow (100M CNY)",
                    "change_pct": "Change %",
                    "source": "Data Source",
                }
            )
        )
        st.bar_chart(chart)

    st.subheader("Policy Signals")
    for item in policy_signals:
        title = _display_text(item.get("title", ""), 120)
        summary = _display_text(item.get("summary", ""), 600)
        source = _display_text(item.get("source", ""), 100)
        published = _display_text(item.get("published_at", "unknown"), 50)
        credibility = min(1.0, max(0.0, _to_float(item.get("credibility", 0.0), 0.0)))
        st.markdown(f"**{title}**")
        st.write(summary)
        st.caption(f"Source: {source} | Credibility: {credibility:.2f} | Published: {published}")
        st.divider()

    st.subheader("Signal Sectors")
    if result:
        confidence = str(result.get("overall_confidence", "LOW"))
        st.metric("Overall Confidence", confidence)
        if confidence == "LOW":
            st.error(
                "This analysis is LOW confidence. Signals conflict, data are missing, "
                "or fallback/mock data were used. Verify independently before acting."
            )
    if result and result.get("sectors"):
        for sector in result["sectors"]:
            score = float(sector["conviction_score"])
            st.markdown(f"**{_display_text(sector['name'], 80)}**")
            st.progress(max(0.0, min(100.0, score)) / 100.0)
            st.write(_display_text(sector.get("reasoning", ""), 500))
            st.caption(
                f"Conviction: {score:.0f}/100 | Policy alignment: "
                f"{_display_text(sector.get('policy_alignment', ''), 120)} | "
                f"Flow magnitude: {_display_text(sector.get('flow_magnitude', ''), 120)}"
            )
            st.divider()
    else:
        st.info("Run AI analysis to fuse capital flow, policy, and fund universe data into sector convictions.")


def render_funds_tab(fund_universe: pd.DataFrame, result: Optional[Dict[str, Any]]) -> None:
    st.subheader("AI Recommended Funds")
    if result:
        rec_df = pd.DataFrame(result.get("recommended_funds", []))
        if rec_df.empty:
            st.info("The model did not return fund recommendations.")
        else:
            rec_cols = [
                "fund_code",
                "fund_name",
                "suggested_allocation_pct",
                "sector",
                "risk_level",
                "reasoning",
            ]
            rec_cols = [col for col in rec_cols if col in rec_df.columns]
            st.dataframe(
                rec_df[rec_cols].rename(
                    columns={
                        "fund_code": "Code",
                        "fund_name": "Fund",
                        "suggested_allocation_pct": "Allocation %",
                        "sector": "Sector",
                        "risk_level": "Risk",
                        "reasoning": "Reason",
                    }
                )
            )

        relevant = result.get("relevant_funds")
        if relevant is not None and not relevant.empty:
            st.subheader("Fund Universe Candidates by Signal Sector")
            cols = ["fund_code", "fund_name", "fund_size", "one_year_return", "volatility_pct"]
            cols = [col for col in cols if col in relevant.columns]
            st.dataframe(relevant[cols])
    else:
        st.info("Run AI analysis to generate fund recommendations from the current signals.")
        if fund_universe is not None and not fund_universe.empty:
            st.caption("Live/mock fund universe is loaded; the analysis will filter it by signal sectors.")
            cols = ["fund_code", "fund_name", "fund_size", "one_year_return"]
            cols = [col for col in cols if col in fund_universe.columns]
            st.dataframe(fund_universe.head(20)[cols])


def render_rebalance_tab(
    result: Optional[Dict[str, Any]],
    holdings_df: Optional[pd.DataFrame],
    available_cash: float,
    risk_preference: str,
) -> None:
    st.subheader("Rebalancing Advice")
    if not result:
        st.info("Run AI analysis to get per-holding rebalance actions.")
    else:
        actions_df = pd.DataFrame(result.get("rebalance_actions", []))
        if actions_df.empty:
            st.info("The model did not return rebalance actions.")
        else:
            action_cols = ["fund_code", "fund_name", "action", "suggested_amount", "reasoning"]
            action_cols = [col for col in action_cols if col in actions_df.columns]
            st.dataframe(
                actions_df[action_cols].rename(
                    columns={
                        "fund_code": "Fund Code",
                        "fund_name": "Fund",
                        "action": "Action",
                        "suggested_amount": "Suggested Amount (CNY)",
                        "reasoning": "Reason",
                    }
                )
            )
            provided_codes = set(actions_df["fund_code"].dropna().astype(str))
            if holdings_df is not None and not holdings_df.empty:
                missing = holdings_df[~holdings_df["fund_code"].astype(str).isin(provided_codes)]
                if not missing.empty:
                    st.warning(
                        "The model did not return advice for these holdings: "
                        + ", ".join(missing["fund_code"].astype(str).tolist())
                    )

        confidence = str(result.get("overall_confidence", "LOW"))
        st.metric("Available Cash", f"{available_cash:,.2f} CNY")
        st.write(f"Risk preference: {risk_preference} | Overall confidence: {confidence}")
        confidence_note = result.get("confidence_note")
        if confidence_note:
            st.caption(f"Confidence note: {_display_text(confidence_note, 400)}")
        if "LOW" in confidence.upper():
            st.warning(
                "The model marked this analysis LOW confidence because signals conflict, data are missing, "
                "or fallback/mock data were used. Verify independently before acting."
            )

    if holdings_df is not None and not holdings_df.empty:
        st.subheader("Parsed Holdings")
        st.dataframe(holdings_df)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:", layout="wide")
    st.title(APP_TITLE)
    st.caption("Personal reference analysis with validation, fallback data, and uncertainty reporting.")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    if not api_key:
        st.sidebar.error(
            "DEEPSEEK_API_KEY is not set. Add it to the environment before running AI analysis. "
            "Market data tabs remain available."
        )
    if not base_url:
        st.sidebar.error("DEEPSEEK_BASE_URL is not set. Add it to the environment before running AI analysis.")
    if _openai is None:
        st.sidebar.error("The openai package could not be imported. Install required dependencies.")
    if ak is None:
        st.sidebar.error("The akshare package could not be imported. All market data will use mock fallbacks.")

    with st.sidebar:
        st.header("Inputs")
        pasted_text = st.text_area(
            "Paste holdings CSV/table",
            height=180,
            placeholder="fund_code,fund_name,shares,cost\n000001,Example Fund,1000,1.25",
        )
        uploaded_file = st.file_uploader("Or upload holdings CSV", type=["csv", "txt"])
        available_cash = st.number_input(
            "Available cash (CNY)",
            min_value=0.0,
            value=100000.0,
            step=10000.0,
            format="%.2f",
        )
        risk_preference = st.selectbox(
            "Risk preference",
            ["conservative", "balanced", "aggressive"],
            index=1,
        )
        run_clicked = st.button("Run AI analysis", type="primary", use_container_width=True)

    with st.spinner("Loading and validating market data..."):
        capital_flow, capital_meta, capital_warnings = fetch_capital_flow()
        policy_signals, policy_meta, policy_warnings = fetch_policy_signals()
        fund_universe, fund_meta, fund_warnings = fetch_fund_universe()

    capital_flow, cap_source, capital_warnings, capital_anomaly = validate_capital_flow(
        capital_flow, capital_meta, capital_warnings.copy()
    )
    data_warnings = list(capital_warnings) + list(policy_warnings) + list(fund_warnings)

    if run_clicked:
        holdings_df, errors = parse_holdings_input(pasted_text, uploaded_file)
        if errors:
            st.error("Holdings input is invalid:")
            for error in errors:
                st.write(f"- {error}")
        elif not api_key:
            st.error(
                "DEEPSEEK_API_KEY is not set. Set it in the environment and restart the app "
                "to run AI analysis."
            )
        else:
            try:
                with st.spinner("Fusing signals and generating structured advice..."):
                    result = build_analysis(
                        holdings_df=holdings_df,
                        available_cash=available_cash,
                        risk_preference=risk_preference,
                        capital_flow_df=capital_flow,
                        policy_signals=policy_signals,
                        fund_universe_df=fund_universe,
                        data_warnings=data_warnings,
                    )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_holdings"] = holdings_df
            except AnalysisError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

    analysis_result = st.session_state.get("analysis_result")
    analysis_holdings = st.session_state.get("analysis_holdings")

    combined_warning_text = " ".join(data_warnings).lower()
    if any(
        marker in combined_warning_text
        for marker in ("fallback", "mock", "unavailable", "failed", "empty", "anomaly")
    ):
        st.warning(
            "One or more data sources are using fallback/mock data or failed validation. "
            "AI analysis will be marked LOW confidence."
        )

    tab1, tab2, tab3 = st.tabs(
        ["Today's Signals & Sectors", "Recommended Funds", "Rebalancing Plan"]
    )
    with tab1:
        render_signals_tab(
            capital_flow=capital_flow,
            policy_signals=policy_signals,
            result=analysis_result,
            cap_source=cap_source,
            policy_meta=policy_meta,
            warnings=data_warnings,
        )
    with tab2:
        render_funds_tab(fund_universe=fund_universe, result=analysis_result)
    with tab3:
        render_rebalance_tab(
            result=analysis_result,
            holdings_df=analysis_holdings,
            available_cash=available_cash,
            risk_preference=risk_preference,
        )

    st.markdown("---")
    st.caption("AI-generated analysis for personal reference only. Not investment advice.")


if __name__ == "__main__":
    main()
