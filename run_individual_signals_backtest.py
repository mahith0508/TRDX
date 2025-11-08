#!/usr/bin/env python3
"""
Run backtesting with individual signals (not combined)

This script demonstrates trading on individual signals from all strategies
without combining them. Each signal is traded independently.
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from trading_algorithm.config.config import Config
from trading_algorithm.engine.trading_engine import TradingEngine


def run_backtest_with_mode(trade_individual: bool, period: tuple = ('2023-01-01', '2023-12-31')):
    """Run backtest with specified signal trading mode"""
    
    config = Config()
    config.strategy.trade_individual_signals = trade_individual
    
    start_date, end_date = period
    config.backtest.start_date = start_date
    config.backtest.end_date = end_date
    
    mode_name = "INDIVIDUAL SIGNALS" if trade_individual else "COMBINED SIGNALS"
    
    logger.info("=" * 80)
    logger.info(f"BACKTEST: {mode_name}")
    logger.info("=" * 80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    logger.info(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    logger.info(f"Signal Mode: {'Individual (No Combination)' if trade_individual else 'Combined (' + config.strategy.signal_combination_method + ')'}")
    logger.info("-" * 80)
    
    engine = TradingEngine(config)
    
    if not engine.initialize():
        logger.error("Failed to initialize trading engine")
        return None
    
    engine.start_trading()
    
    summary = engine.get_portfolio_summary()
    strategy_perf = engine.get_strategy_performance()
    
    results = {
        'mode': 'individual' if trade_individual else 'combined',
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': config.trading.initial_capital,
        'final_value': summary['portfolio_value'],
        'total_return': summary['total_return'],
        'total_trades': summary['total_trades'],
        'sharpe_ratio': summary['sharpe_ratio'],
        'max_drawdown': summary['max_drawdown'],
        'strategy_performance': strategy_perf
    }
    
    return results


def print_results(results):
    """Print backtest results"""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nMode: {results['mode'].upper()}")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    print("\n" + "-" * 80)
    print("Strategy Performance:")
    print("-" * 80)
    for strategy, metrics in results['strategy_performance'].items():
        print(f"{strategy}:")
        print(f"  Signals: {metrics['total_signals']}, Success Rate: {metrics['success_rate']:.1%}, Weight: {metrics['weight']:.1%}")


def main():
    """Main function"""
    try:
        print("\n" + "=" * 80)
        print("INDIVIDUAL vs COMBINED SIGNALS BACKTEST COMPARISON")
        print("=" * 80)
        
        period = ('2023-01-01', '2023-12-31')
        
        # Run backtest with INDIVIDUAL signals
        individual_results = run_backtest_with_mode(trade_individual=True, period=period)
        if individual_results:
            print_results(individual_results)
        else:
            logger.error("Individual signals backtest failed")
            return
        
        # Run backtest with COMBINED signals
        combined_results = run_backtest_with_mode(trade_individual=False, period=period)
        if combined_results:
            print_results(combined_results)
        else:
            logger.error("Combined signals backtest failed")
            return
        
        # Comparison
        print("\n" + "=" * 80)
        print("COMPARISON: INDIVIDUAL vs COMBINED SIGNALS")
        print("=" * 80)
        
        return_diff = individual_results['total_return'] - combined_results['total_return']
        portfolio_diff = individual_results['final_value'] - combined_results['final_value']
        trades_diff = individual_results['total_trades'] - combined_results['total_trades']
        
        print(f"\nReturn Difference: {return_diff:+.2%}")
        print(f"Portfolio Value Difference: ${portfolio_diff:+,.2f}")
        print(f"Trades Difference: {trades_diff:+d}")
        
        if return_diff > 0:
            print(f"\n✓ INDIVIDUAL signals outperformed by {abs(return_diff):.2%}")
        elif return_diff < 0:
            print(f"\n✓ COMBINED signals outperformed by {abs(return_diff):.2%}")
        else:
            print("\n✓ Both strategies performed equally")
        
        print("\n" + "=" * 80)
        print("✓ COMPARISON COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
