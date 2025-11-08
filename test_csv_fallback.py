#!/usr/bin/env python3
"""
Test script to verify CSV fallback functionality
"""
import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        from trading_algorithm.config.config import Config
        from trading_algorithm.data.data_manager import DataManager
        from trading_algorithm.strategies.strategy_manager import StrategyManager
        
        logger.info("Testing CSV fallback functionality...")
        
        # Load configuration
        config = Config()
        
        # Initialize data manager
        data_manager = DataManager(config)
        
        # List available CSV symbols
        csv_symbols = data_manager.list_csv_symbols()
        logger.info(f"Available CSV symbols: {csv_symbols}")
        
        if not csv_symbols:
            logger.error("No CSV symbols found. Run download_data.py first.")
            sys.exit(1)
        
        # Test CSV data loading
        logger.info("\n" + "="*60)
        logger.info("TESTING CSV DATA LOADING")
        logger.info("="*60)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Test with CSV symbols
        csv_data = data_manager.get_data(csv_symbols, start_date, end_date)
        
        logger.info(f"Successfully loaded data for {len(csv_data)} symbols from CSV/cache")
        for symbol, df in csv_data.items():
            logger.info(f"  {symbol}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
        
        # Test signal generation with CSV data
        logger.info("\n" + "="*60)
        logger.info("TESTING SIGNAL GENERATION WITH CSV DATA")
        logger.info("="*60)
        
        # Initialize strategy manager
        strategy_manager = StrategyManager(config)
        
        # Get data with indicators
        data_with_indicators = data_manager.get_data_with_indicators(csv_symbols, start_date, end_date)
        
        # Initialize strategies
        strategy_manager.initialize_strategies(data_with_indicators)
        
        # Generate signals
        signals = strategy_manager.generate_signals(data_with_indicators)
        
        logger.info(f"Generated {len(signals)} signals from CSV data")
        
        # Display signals
        for i, signal in enumerate(signals[:5]):  # Show first 5 signals
            logger.info(f"  Signal {i+1}: {signal.symbol} {signal.action} strength={signal.strength:.3f} price={signal.price:.2f}")
        
        logger.info("\n✓ CSV fallback test PASSED!")
        logger.info("The platform can now work offline using CSV data files.")
        
    except Exception as e:
        logger.error(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()