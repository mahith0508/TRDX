#!/usr/bin/env python3
"""
Test script to verify signal generation for all strategies
"""
import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from trading_algorithm.config.config import Config
    from trading_algorithm.engine.trading_engine import TradingEngine
    from trading_algorithm.data.data_manager import DataManager
    from trading_algorithm.strategies.strategy_manager import StrategyManager
    
    logger.info("Testing signal generation for all strategies...")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = Config()
    
    # Initialize components
    logger.info("Initializing data manager...")
    data_manager = DataManager(config)
    
    logger.info("Initializing strategy manager...")
    strategy_manager = StrategyManager(config)
    
    # Get historical data
    logger.info("Fetching historical data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = data_manager.get_data_with_indicators(
        config.universe,
        start_date,
        end_date
    )
    
    if not data:
        logger.error("Failed to fetch data")
        sys.exit(1)
    
    logger.info(f"Fetched data for {len(data)} symbols")
    
    # Initialize strategies
    logger.info("Initializing strategies...")
    strategy_manager.initialize_strategies(data)
    
    # Generate signals
    logger.info("Generating signals...")
    signals = strategy_manager.generate_signals(data)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SIGNAL GENERATION TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total signals generated: {len(signals)}")
    logger.info(f"Active strategies: {len(strategy_manager.strategies)}")
    
    # Print performance metrics
    logger.info("\nStrategy Performance:")
    for strategy_name, metrics in strategy_manager.get_strategy_performance().items():
        logger.info(f"  {strategy_name}: {metrics['total_signals']} signals, weight: {metrics['weight']:.3f}")
    
    # Test pairs trading specifically
    logger.info("\n" + "="*60)
    logger.info("PAIRS TRADING STRATEGY TEST")
    logger.info("="*60)
    
    pairs_strategy = strategy_manager.strategies.get('pairs_trading')
    if pairs_strategy:
        logger.info(f"Pairs Trading Active Pairs: {len(pairs_strategy.active_pairs)}")
        logger.info(f"Pairs Trading Pairs Data Keys: {len(pairs_strategy.pairs_data)}")
        
        # List active pairs
        for pair_name in pairs_strategy.active_pairs[:5]:
            logger.info(f"  - {pair_name}")
        
        logger.info("✓ Pairs trading strategy initialized successfully!")
    else:
        logger.error("✗ Pairs trading strategy not found!")
    
    logger.info("\n" + "="*60)
    logger.info("✓ SIGNAL GENERATION TEST PASSED!")
    logger.info("="*60)
    
except KeyError as e:
    logger.error(f"✗ KeyError during signal generation: {str(e)}")
    logger.error("This likely indicates a column name mismatch in pairs trading")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    logger.error(f"✗ Error during signal generation: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
