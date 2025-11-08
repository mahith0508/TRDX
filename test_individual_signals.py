#!/usr/bin/env python3
"""
Test script to compare combined vs individual signal trading performance
Runs backtests with signals generated individually without combination
"""
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from trading_algorithm.config.config import Config
from trading_algorithm.engine.trading_engine import TradingEngine
from trading_algorithm.data.data_manager import DataManager
from trading_algorithm.strategies.strategy_manager import StrategyManager

config = Config()

class IndividualSignalStrategyManager(StrategyManager):
    """
    Custom strategy manager that returns individual signals without combining them
    """
    def generate_signals(self, data):
        """Generate signals from all strategies WITHOUT combining them"""
        all_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Filter data for strategy
                filtered_data = strategy.filter_universe(data)
                
                if filtered_data:
                    # Generate signals
                    signals = strategy.calculate_signals(filtered_data)
                    
                    # Limit number of signals per strategy
                    signals = signals[:self.max_signals_per_strategy]
                    
                    # Add strategy name to signals
                    for signal in signals:
                        signal.metadata['strategy'] = strategy_name
                        signal.metadata['is_individual'] = True
                    
                    all_signals.extend(signals)
                    
                    logger.info(f"Strategy {strategy_name} generated {len(signals)} individual signals")
                
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_name}: {str(e)}")
        
        logger.info(f"Total individual signals from all strategies: {len(all_signals)}")
        
        # Store signal history
        strategy_signals = {strategy: [] for strategy in self.strategies}
        for signal in all_signals:
            strategy = signal.metadata.get('strategy')
            if strategy:
                strategy_signals[strategy].append(signal)
        
        self.signal_history.append({
            'timestamp': datetime.now(),
            'strategy_signals': strategy_signals,
            'combined_signals': all_signals  # Store as "combined" for compatibility
        })
        
        return all_signals


def run_individual_signal_backtest():
    """Run backtest using individual signals"""
    logger.info("=" * 60)
    logger.info("INDIVIDUAL SIGNALS BACKTEST")
    logger.info("=" * 60)
    
    # Create trading engine
    engine = TradingEngine(config)
    
    # Replace strategy manager with individual signal version
    engine.strategy_manager = IndividualSignalStrategyManager(config)
    
    # Initialize
    if not engine.initialize():
        logger.error("Failed to initialize trading engine")
        return None
    
    # Get backtest parameters
    start_date = config.backtest.start_date
    end_date = config.backtest.end_date
    initial_capital = config.trading.initial_capital
    
    logger.info(f"Backtest Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Universe: {', '.join(config.universe)}")
    logger.info(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    logger.info(f"Signal Trading Mode: INDIVIDUAL (NO COMBINATION)")
    logger.info("-" * 60)
    
    # Run backtest
    engine.start_trading()
    
    # Get results
    summary = engine.get_portfolio_summary()
    strategy_perf = engine.get_strategy_performance()
    
    results = {
        'mode': 'individual_signals',
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'final_portfolio_value': summary['portfolio_value'],
        'total_return': summary['total_return'],
        'total_return_pct': f"{summary['total_return']:.2%}",
        'total_trades': summary['total_trades'],
        'sharpe_ratio': summary['sharpe_ratio'],
        'max_drawdown': summary['max_drawdown'],
        'max_drawdown_pct': f"{summary['max_drawdown']:.2%}",
        'winning_trades': summary.get('winning_trades', 0),
        'losing_trades': summary.get('losing_trades', 0),
        'num_positions': summary['num_positions'],
        'timestamp': datetime.now().isoformat(),
        'strategy_performance': strategy_perf
    }
    
    return results, engine


def run_combined_signal_backtest():
    """Run backtest using combined signals (original method)"""
    logger.info("=" * 60)
    logger.info("COMBINED SIGNALS BACKTEST")
    logger.info("=" * 60)
    
    # Create trading engine with default strategy manager
    engine = TradingEngine(config)
    
    # Initialize
    if not engine.initialize():
        logger.error("Failed to initialize trading engine")
        return None
    
    # Get backtest parameters
    start_date = config.backtest.start_date
    end_date = config.backtest.end_date
    initial_capital = config.trading.initial_capital
    
    logger.info(f"Backtest Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Universe: {', '.join(config.universe)}")
    logger.info(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    logger.info(f"Signal Trading Mode: COMBINED (Weighted Average)")
    logger.info("-" * 60)
    
    # Run backtest
    engine.start_trading()
    
    # Get results
    summary = engine.get_portfolio_summary()
    strategy_perf = engine.get_strategy_performance()
    
    results = {
        'mode': 'combined_signals',
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'final_portfolio_value': summary['portfolio_value'],
        'total_return': summary['total_return'],
        'total_return_pct': f"{summary['total_return']:.2%}",
        'total_trades': summary['total_trades'],
        'sharpe_ratio': summary['sharpe_ratio'],
        'max_drawdown': summary['max_drawdown'],
        'max_drawdown_pct': f"{summary['max_drawdown']:.2%}",
        'winning_trades': summary.get('winning_trades', 0),
        'losing_trades': summary.get('losing_trades', 0),
        'num_positions': summary['num_positions'],
        'timestamp': datetime.now().isoformat(),
        'strategy_performance': strategy_perf
    }
    
    return results, engine


def print_results(individual_results, combined_results):
    """Print comparison of results"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 70)
    
    # Individual signals results
    print("\n📊 INDIVIDUAL SIGNALS BACKTEST RESULTS")
    print("-" * 70)
    print(f"Final Portfolio Value:  ${individual_results['final_portfolio_value']:,.2f}")
    print(f"Total Return:           {individual_results['total_return_pct']}")
    print(f"Total Trades:           {individual_results['total_trades']}")
    print(f"Sharpe Ratio:           {individual_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {individual_results['max_drawdown_pct']}")
    
    # Combined signals results
    print("\n📊 COMBINED SIGNALS BACKTEST RESULTS")
    print("-" * 70)
    print(f"Final Portfolio Value:  ${combined_results['final_portfolio_value']:,.2f}")
    print(f"Total Return:           {combined_results['total_return_pct']}")
    print(f"Total Trades:           {combined_results['total_trades']}")
    print(f"Sharpe Ratio:           {combined_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {combined_results['max_drawdown_pct']}")
    
    # Comparison
    print("\n📈 COMPARISON")
    print("-" * 70)
    
    return_diff = individual_results['total_return'] - combined_results['total_return']
    return_diff_pct = (return_diff / combined_results['total_return'] * 100) if combined_results['total_return'] != 0 else 0
    
    portfolio_diff = individual_results['final_portfolio_value'] - combined_results['final_portfolio_value']
    
    trades_diff = individual_results['total_trades'] - combined_results['total_trades']
    
    sharpe_diff = individual_results['sharpe_ratio'] - combined_results['sharpe_ratio']
    
    drawdown_diff = individual_results['max_drawdown'] - combined_results['max_drawdown']
    
    print(f"Portfolio Value Difference:     ${portfolio_diff:+,.2f}")
    print(f"Return Difference:              {return_diff:+.4f} ({return_diff_pct:+.2f}%)")
    print(f"Trades Difference:              {trades_diff:+d}")
    print(f"Sharpe Ratio Difference:        {sharpe_diff:+.2f}")
    print(f"Max Drawdown Difference:        {drawdown_diff:+.4f}")
    
    # Determine winner
    print("\n🏆 PERFORMANCE WINNER")
    print("-" * 70)
    if return_diff > 0:
        print(f"✓ Individual Signals trading is {abs(return_diff_pct):.2f}% better!")
    elif return_diff < 0:
        print(f"✓ Combined Signals trading is {abs(return_diff_pct):.2f}% better!")
    else:
        print("✓ Both strategies performed equally")
    
    # Strategy breakdown
    print("\n📋 STRATEGY BREAKDOWN - INDIVIDUAL SIGNALS")
    print("-" * 70)
    for strategy, metrics in individual_results['strategy_performance'].items():
        print(f"{strategy}:")
        print(f"  - Total Signals: {metrics['total_signals']}")
        print(f"  - Success Rate:  {metrics['success_rate']:.1%}")
        print(f"  - Weight:        {metrics['weight']:.1%}")
    
    print("\n📋 STRATEGY BREAKDOWN - COMBINED SIGNALS")
    print("-" * 70)
    for strategy, metrics in combined_results['strategy_performance'].items():
        print(f"{strategy}:")
        print(f"  - Total Signals: {metrics['total_signals']}")
        print(f"  - Success Rate:  {metrics['success_rate']:.1%}")
        print(f"  - Weight:        {metrics['weight']:.1%}")
    
    print("\n" + "=" * 70)


def main():
    """Main function"""
    try:
        logger.info("Starting Individual vs Combined Signals Backtest Comparison")
        logger.info("=" * 60)
        
        # Run both backtests
        logger.info("\nPhase 1: Running Combined Signals Backtest...")
        combined_result = run_combined_signal_backtest()
        if combined_result is None:
            logger.error("Combined signals backtest failed")
            return
        
        combined_results, combined_engine = combined_result
        
        logger.info("\nPhase 2: Running Individual Signals Backtest...")
        individual_result = run_individual_signal_backtest()
        if individual_result is None:
            logger.error("Individual signals backtest failed")
            return
        
        individual_results, individual_engine = individual_result
        
        # Print comparison
        print_results(individual_results, combined_results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"backtest_comparison_{timestamp}.json"
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'individual_signals': individual_results,
            'combined_signals': combined_results,
            'comparison': {
                'return_diff': float(individual_results['total_return'] - combined_results['total_return']),
                'portfolio_value_diff': float(individual_results['final_portfolio_value'] - combined_results['final_portfolio_value']),
                'trades_diff': int(individual_results['total_trades'] - combined_results['total_trades']),
                'sharpe_diff': float(individual_results['sharpe_ratio'] - combined_results['sharpe_ratio']),
                'winner': 'individual' if individual_results['total_return'] > combined_results['total_return'] else 'combined'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
        logger.info("✓ Individual vs Combined Signals Backtest Comparison Complete!")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
