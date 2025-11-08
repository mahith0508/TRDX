#!/usr/bin/env python3
"""
Utility script to download market data and save to CSV for offline use
"""
import sys
import logging
from datetime import datetime, timedelta
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Download market data and save to CSV files')
    parser.add_argument('--symbols', nargs='+', default=None, 
                       help='Symbols to download (default: use config universe)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD, default: 1 year ago)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--output-dir', type=str, default='csv_data',
                       help='Output directory for CSV files')
    parser.add_argument('--list-csv', action='store_true',
                       help='List available CSV files')
    
    args = parser.parse_args()
    
    try:
        from trading_algorithm.config.config import Config
        from trading_algorithm.data.data_manager import DataManager
        
        # Load configuration
        config = Config()
        
        # Initialize data manager
        data_manager = DataManager(config)
        
        # List CSV files if requested
        if args.list_csv:
            csv_symbols = data_manager.list_csv_symbols()
            if csv_symbols:
                logger.info(f"Available CSV symbols ({len(csv_symbols)}):")
                for symbol in sorted(csv_symbols):
                    logger.info(f"  - {symbol}")
            else:
                logger.info("No CSV files found")
            return
        
        # Determine symbols to download
        symbols = args.symbols if args.symbols else config.universe
        
        # Determine date range
        if args.end_date:
            end_date = args.end_date
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        if args.start_date:
            start_date = args.start_date
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        logger.info(f"Downloading data for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Save data to CSV
        data_manager.save_data_to_csv(symbols, start_date, end_date, args.output_dir)
        
        logger.info("✓ Data download complete!")
        logger.info(f"You can now run signal generation using the CSV files as fallback")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()