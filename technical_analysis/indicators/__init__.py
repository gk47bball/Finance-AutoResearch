"""
JK Indicator Library — Python translations of TradeStation EasyLanguage indicators.

All indicators take a pandas DataFrame with columns: Open, High, Low, Close, Volume
and return a pandas Series (or DataFrame of multiple signals).

Naming convention:
  - jk_ prefix for Jonathan Kornblatt's custom indicators
  - standard_ prefix for well-known indicators used as components (RSI, EMA, etc.)
"""
