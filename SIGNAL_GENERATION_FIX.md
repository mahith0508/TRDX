# Signal Generation Fix - Pairs Trading Strategy

## Problem Summary

The pairs trading strategy was failing to generate signals due to a critical bug in column naming when merging dataframes. The error was:
```
Error generating signals for strategy pairs_trading: 'AAPL_1'
```

## Root Cause

When merging two pandas Series using `pd.merge()` with `suffixes=('_1', '_2')`, the resulting column names are `close_1` and `close_2`, NOT `{symbol}_1` and `{symbol}_2`.

The code was attempting to access:
```python
aligned_data[f'{symbol1}_1']  # Tries to access 'AAPL_1' column - WRONG!
aligned_data[f'{symbol2}_2']  # Tries to access 'GOOGL_2' column - WRONG!
```

But the actual columns after merge were:
```python
aligned_data['close_1']  # Correct column name
aligned_data['close_2']  # Correct column name
```

## Solution

Fixed all occurrences in `trading_algorithm/strategies/pairs_trading_strategy.py`:

### 1. `_calculate_pair_metrics()` method (Lines 144-147)
```python
# Before:
correlation = aligned_data[f'{symbol1}_1'].corr(aligned_data[f'{symbol2}_2'])
spread = aligned_data[f'{symbol1}_1'] - aligned_data[f'{symbol2}_2']

# After:
correlation = aligned_data['close_1'].corr(aligned_data['close_2'])
spread = aligned_data['close_1'] - aligned_data['close_2']
```

### 2. `_calculate_current_spread_metrics()` method (Lines 403-404)
```python
# Before:
spread = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]

# After:
spread = aligned_data['close_1'] - aligned_data['close_2']
```

### 3. Added error handling in `calculate_signals()` (Lines 56-72)
Wrapped the signal generation logic in try-except block with detailed error logging for better debugging.

### 4. Updated `_calculate_cointegration_score()` (Lines 174-178)
Made column references more flexible by using dynamic column names, while maintaining correctness.

## Verification

All tests pass successfully:

✓ **Paper Trading Mode**: No signal generation errors
✓ **Analysis Mode**: All strategies generate signals correctly
✓ **Unit Test**: Pairs trading strategy initializes with 10 active pairs
✓ **Signal Generation**: 
  - Momentum: 5 signals
  - Mean Reversion: 1 signal
  - RSI: 0 signals
  - Moving Average: 0 signals  
  - Bollinger Bands: 5 signals
  - **Pairs Trading: 2 signals (no errors!)**

## Changes Made

- Modified `trading_algorithm/strategies/pairs_trading_strategy.py`
- 25 insertions, 20 deletions
- Commit: `da37fe3` - "fix(pairs_trading): fix signal generation column naming errors"

## Testing

Run the test suite:
```bash
python test_signal_generation.py
```

Or run the main application:
```bash
python trading_algorithm/main.py paper --initial-capital 50000
python trading_algorithm/main.py analysis
```

## Status

✅ **FIXED** - Signal generation for pairs trading strategy now works without errors.
