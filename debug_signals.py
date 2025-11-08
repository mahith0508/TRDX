#!/usr/bin/env python3
"""
Debug script to examine individual signal values
"""
import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from trading_algorithm.config.config import Config
    from trading_algorithm.data.data_manager import DataManager
    from trading_algorithm.strategies.strategy_manager import StrategyManager
    
    logger.info("Debugging signal generation...")
    
    # Load configuration
    config = Config()
    logger.info(f"Signal combination method: {config.strategy.signal_combination_method}")
    
    # Initialize components
    data_manager = DataManager(config)
    strategy_manager = StrategyManager(config)
    
    # Get historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = data_manager.get_data_with_indicators(
        config.universe,
        start_date,
        end_date
    )
    
    # Initialize strategies
    strategy_manager.initialize_strategies(data)
    
    # Generate signals and examine individual strategy signals
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL STRATEGY SIGNALS")
    logger.info("="*60)
    
    all_strategy_signals = {}
    total_signals = 0
    
    for strategy_name, strategy in strategy_manager.strategies.items():
        if hasattr(strategy, 'calculate_signals'):
            try:
                signals = strategy.calculate_signals(data)
                all_strategy_signals[strategy_name] = signals
                total_signals += len(signals)
                
                logger.info(f"\n{strategy_name.upper()} Strategy:")
                logger.info(f"  Signals generated: {len(signals)}")
                
                for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                    logger.info(f"    Signal {i+1}: {signal.symbol} {signal.action} strength={signal.strength:.3f} price={signal.price:.2f}")
                    
            except Exception as e:
                logger.error(f"Error in {strategy_name}: {e}")
    
    logger.info(f"\nTotal individual signals: {total_signals}")
    
    # Test the combination logic with debug info
    logger.info("\n" + "="*60)
    logger.info("SIGNAL COMBINATION DEBUG")
    logger.info("="*60)
    
    combined = strategy_manager._combine_signals_weighted_average(all_strategy_signals)
    logger.info(f"Combined signals: {len(combined)}")
    
    for signal in combined[:5]:
        logger.info(f"  Combined: {signal.symbol} {signal.action} strength={signal.strength:.3f}")
    
    logger.info("\n✓ DEBUG COMPLETE")
    
except Exception as e:
    logger.error(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)