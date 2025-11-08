# CSV Import Fallback Documentation

The trading platform now supports CSV data import as a fallback mechanism, allowing the system to work offline or when primary data sources are unavailable.

## Features

### 1. Automatic CSV Fallback
- When the primary data source (Yahoo Finance) fails, the system automatically tries to load data from CSV files
- Works seamlessly with existing signal generation and backtesting functionality
- No configuration changes required

### 2. CSV Data Provider
- Supports standard OHLCV format (Open, High, Low, Close, Volume)
- Flexible column name mapping
- Automatic date parsing and filtering
- Returns calculation included

### 3. Data Download Utility
- Command-line tool to download market data and save to CSV
- Support for custom date ranges and symbol lists
- Batch processing for multiple symbols

## Usage

### Downloading Data to CSV

```bash
# Download data for all symbols in configuration
python download_data.py

# Download data for specific symbols
python download_data.py --symbols AAPL MSFT GOOGL

# Custom date range
python download_data.py --start-date 2024-01-01 --end-date 2024-11-01

# Custom output directory
python download_data.py --output-dir my_csv_data

# List available CSV files
python download_data.py --list-csv
```

### CSV File Format

CSV files should have the following columns (case-insensitive):
- `date` or `Date` - Trading date
- `open` or `Open` - Opening price
- `high` or `High` - Highest price
- `low` or `Low` - Lowest price
- `close` or `Close` - Closing price
- `volume` or `Volume` - Trading volume

Example CSV format:
```csv
date,open,high,low,close,volume
2024-01-02,185.58,186.86,182.35,184.08,82488700
2024-01-03,182.67,184.32,181.89,182.70,58414500
```

### File Organization

- CSV files are stored in the `csv_data/` directory by default
- Each symbol has its own file: `{symbol}.csv` (e.g., `AAPL.csv`)
- The directory is created automatically if it doesn't exist

### Testing CSV Functionality

```bash
# Test CSV fallback with downloaded data
python test_csv_fallback.py

# Run comprehensive signal generation test
python test_signal_generation.py
```

## How It Works

1. **Primary Data Source**: The system first tries to fetch data from the configured primary source (Yahoo Finance by default)

2. **Automatic Fallback**: If the primary source fails or returns incomplete data, the system automatically:
   - Checks for CSV files in the configured directory
   - Loads data for any missing symbols
   - Logs the fallback activity

3. **Seamless Integration**: CSV data works exactly like live data:
   - Technical indicators are calculated automatically
   - Signal generation works normally
   - Backtesting and analysis functions unchanged

## Configuration

The CSV provider is automatically initialized when the data manager loads. No additional configuration is required.

## Error Handling

- Missing CSV files are logged but don't cause system failures
- Invalid CSV formats are skipped with warnings
- The system continues to work with whatever data is available

## Benefits

1. **Offline Operation**: Work without internet connection
2. **Data Persistence**: Keep historical data locally
3. **Speed**: Faster data loading from local files
4. **Reliability**: Backup when external APIs are down
5. **Testing**: Use consistent historical data for testing

## Troubleshooting

### CSV Files Not Found
- Ensure files are in the correct directory (`csv_data/` by default)
- Check file naming: `{symbol}.csv` (e.g., `AAPL.csv`)
- Use `python download_data.py --list-csv` to verify available files

### Invalid CSV Format
- Ensure required columns are present: date, open, high, low, close, volume
- Check that dates are in a recognizable format (YYYY-MM-DD recommended)
- Verify there are no missing values in critical columns

### No Signals Generated
- Check that CSV data covers the requested date range
- Ensure sufficient historical data for technical indicators (typically 50+ periods)
- Verify data quality with the test scripts

## Integration with Existing Workflows

The CSV fallback integrates seamlessly with all existing functionality:

- **Signal Generation**: `python test_signal_generation.py`
- **Backtesting**: `python trading_algorithm/main.py backtest`
- **Live Trading**: Uses CSV as fallback when real-time data fails
- **Analysis Tools**: All analysis functions work with CSV data

The system automatically chooses the best data source available, ensuring reliable operation under all conditions.