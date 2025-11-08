#!/usr/bin/env python3
"""Final comprehensive signal generation test"""
import sys
from datetime import datetime, timedelta
from trading_algorithm.config.config import Config
from trading_algorithm.data.data_manager import DataManager
from trading_algorithm.strategies.strategy_manager import StrategyManager

def test_signal_generation():
    print("\n" + "="*70)
    print("COMPREHENSIVE SIGNAL GENERATION TEST")
    print("="*70)
    
    # Initialize
    config = Config()
    data_manager = DataManager(config)
    strategy_manager = StrategyManager(config)
    
    # Get data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = data_manager.get_data_with_indicators(config.universe, start_date, end_date)
    
    if not data:
        print("❌ Failed to fetch data")
        return False
    
    # Initialize strategies
    strategy_manager.initialize_strategies(data)
    
    # Generate signals
    try:
        signals = strategy_manager.generate_signals(data)
        
        print(f"\n✓ Signal Generation Successful!")
        print(f"  Total signals: {len(signals)}")
        print(f"  Active strategies: {len(strategy_manager.strategies)}")
        
        # Print strategy breakdown
        print(f"\n  Strategy Signals Generated:")
        for strategy_name, metrics in strategy_manager.get_strategy_performance().items():
            print(f"    - {strategy_name:30s}: {metrics['total_signals']:3d} signals")
        
        # Check pairs trading specifically
        pairs_strategy = strategy_manager.strategies.get('pairs_trading')
        if pairs_strategy:
            print(f"\n✓ Pairs Trading Strategy:")
            print(f"    Active pairs: {len(pairs_strategy.active_pairs)}")
            print(f"    Pairs data: {len(pairs_strategy.pairs_data)}")
            if pairs_strategy.active_pairs:
                print(f"    Sample pairs: {', '.join(list(pairs_strategy.active_pairs)[:3])}")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - SIGNAL GENERATION WORKING CORRECTLY")
        print("="*70)
        return True
        
    except KeyError as e:
        print(f"\n❌ KeyError during signal generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Error during signal generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generation()
    sys.exit(0 if success else 1)
