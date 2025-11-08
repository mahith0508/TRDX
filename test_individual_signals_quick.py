#!/usr/bin/env python3
"""
Quick test script for trading on individual signals (not combined)
Runs backtest with shorter timeframe for faster results
"""
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from trading_algorithm.config.config import Config
from trading_algorithm.engine.trading_engine import TradingEngine
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
                filtered_data = strategy.filter_universe(data)
                
                if filtered_data:
                    signals = strategy.calculate_signals(filtered_data)
                    signals = signals[:self.max_signals_per_strategy]
                    
                    for signal in signals:
                        signal.metadata['strategy'] = strategy_name
                        signal.metadata['is_individual'] = True
                    
                    all_signals.extend(signals)
                    
                    logger.info(f"Strategy {strategy_name} generated {len(signals)} individual signals")
                
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_name}: {str(e)}")
        
        logger.info(f"Total individual signals from all strategies: {len(all_signals)}")
        
        strategy_signals = {strategy: [] for strategy in self.strategies}
        for signal in all_signals:
            strategy = signal.metadata.get('strategy')
            if strategy:
                strategy_signals[strategy].append(signal)
        
        self.signal_history.append({
            'timestamp': datetime.now(),
            'strategy_signals': strategy_signals,
            'combined_signals': all_signals
        })
        
        return all_signals


def run_individual_signal_backtest():
    """Run backtest using individual signals"""
    logger.info("=" * 70)
    logger.info("INDIVIDUAL SIGNALS BACKTEST (2023 Period)")
    logger.info("=" * 70)
    
    engine = TradingEngine(config)
    engine.strategy_manager = IndividualSignalStrategyManager(config)
    
    # Override with shorter timeframe for quick test
    config.backtest.start_date = '2023-01-01'
    config.backtest.end_date = '2023-12-31'
    
    if not engine.initialize():
        logger.error("Failed to initialize trading engine")
        return None
    
    start_date = config.backtest.start_date
    end_date = config.backtest.end_date
    initial_capital = config.trading.initial_capital
    
    logger.info(f"Backtest Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Universe: {', '.join(config.universe)}")
    logger.info(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    logger.info(f"Signal Mode: INDIVIDUAL (NO COMBINATION)")
    logger.info("-" * 70)
    
    engine.start_trading()
    
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
        'num_positions': summary['num_positions'],
        'strategy_performance': strategy_perf
    }
    
    return results, engine


def main():
    """Main function"""
    try:
        logger.info("Starting Individual Signals Backtest")
        logger.info("=" * 70 + "\n")
        
        individual_result = run_individual_signal_backtest()
        if individual_result is None:
            logger.error("Individual signals backtest failed")
            return
        
        individual_results, engine = individual_result
        
        # Print results
        print("\n" + "=" * 70)
        print("INDIVIDUAL SIGNALS BACKTEST RESULTS")
        print("=" * 70)
        print(f"\nFinal Portfolio Value:  ${individual_results['final_portfolio_value']:,.2f}")
        print(f"Total Return:           {individual_results['total_return_pct']}")
        print(f"Total Trades:           {individual_results['total_trades']}")
        print(f"Sharpe Ratio:           {individual_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:           {individual_results['max_drawdown_pct']}")
        
        print("\n" + "-" * 70)
        print("Strategy Performance (Individual Signals):")
        print("-" * 70)
        for strategy, metrics in individual_results['strategy_performance'].items():
            print(f"{strategy}:")
            print(f"  - Total Signals: {metrics['total_signals']}")
            print(f"  - Success Rate:  {metrics['success_rate']:.1%}")
            print(f"  - Weight:        {metrics['weight']:.1%}")
        
        print("\n" + "=" * 70)
        print("✓ INDIVIDUAL SIGNALS BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Signal history summary
        if engine.strategy_manager.signal_history:
            total_individual_signals = sum(
                len(entry['combined_signals']) 
                for entry in engine.strategy_manager.signal_history 
            )
            logger.info(f"\nTotal individual signals generated: {total_individual_signals}")
            logger.info(f"Trading days with signals: {len(engine.strategy_manager.signal_history)}")
        
        print("\n✓ Test Complete!")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
