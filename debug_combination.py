#!/usr/bin/env python3
"""
Detailed debug script to examine signal combination weighting
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
    
    logger.info("Debugging signal combination weighting...")
    
    # Load configuration
    config = Config()
    
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
    
    # Generate individual strategy signals
    all_strategy_signals = {}
    
    for strategy_name, strategy in strategy_manager.strategies.items():
        if hasattr(strategy, 'calculate_signals'):
            try:
                signals = strategy.calculate_signals(data)
                all_strategy_signals[strategy_name] = signals
            except Exception as e:
                logger.error(f"Error in {strategy_name}: {e}")
    
    # Manual combination debug
    logger.info("\n" + "="*60)
    logger.info("MANUAL SIGNAL COMBINATION DEBUG")
    logger.info("="*60)
    
    combined_signals = {}
    
    for strategy_name, signals in all_strategy_signals.items():
        weight = strategy_manager.strategy_weights.get(strategy_name, 1.0)
        logger.info(f"\nStrategy: {strategy_name} (weight: {weight})")
        
        for signal in signals:
            symbol = signal.symbol
            
            if symbol not in combined_signals:
                combined_signals[symbol] = {
                    'buy_strength': 0.0,
                    'sell_strength': 0.0,
                    'signals': []
                }
            
            # Weight the signal strength
            weighted_strength = signal.strength * weight
            logger.info(f"  {signal.symbol}: {signal.action} {signal.strength:.3f} -> weighted: {weighted_strength:.3f}")
            
            if signal.action == 'buy':
                combined_signals[symbol]['buy_strength'] += weighted_strength
            elif signal.action == 'sell':
                combined_signals[symbol]['sell_strength'] += weighted_strength
            
            combined_signals[symbol]['signals'].append(signal)
    
    # Check final combined signals
    logger.info(f"\n" + "="*60)
    logger.info("FINAL COMBINED SIGNALS")
    logger.info("="*60)
    
    final_signals = []
    for symbol, data in combined_signals.items():
        buy_strength = data['buy_strength']
        sell_strength = data['sell_strength']
        signals = data['signals']
        
        logger.info(f"\n{symbol}:")
        logger.info(f"  Buy strength: {buy_strength:.3f}")
        logger.info(f"  Sell strength: {sell_strength:.3f}")
        logger.info(f"  Signal count: {len(signals)}")
        
        # Determine final action
        if buy_strength > sell_strength and buy_strength > 0.5:
            action = 'buy'
            final_strength = buy_strength
            logger.info(f"  -> FINAL: {action} {final_strength:.3f} ✓")
            final_signals.append(symbol)
        elif sell_strength > buy_strength and sell_strength > 0.5:
            action = 'sell'
            final_strength = sell_strength
            logger.info(f"  -> FINAL: {action} {final_strength:.3f} ✓")
            final_signals.append(symbol)
        else:
            logger.info(f"  -> REJECTED (below 0.5 threshold)")
    
    logger.info(f"\nFinal signals that passed: {len(final_signals)}")
    for symbol in final_signals:
        logger.info(f"  - {symbol}")
    
    logger.info("\n✓ DEBUG COMPLETE")
    
except Exception as e:
    logger.error(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)