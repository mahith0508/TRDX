"""
Main entry point for the trading algorithm
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure the project root is available when executed directly
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_algorithm.config.config import config
from trading_algorithm.engine.trading_engine import TradingEngine

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_backtest(engine: TradingEngine, args):
    """Run backtesting mode"""
    print("=" * 60)
    print("           TRADING ALGORITHM - BACKTEST MODE")
    print("=" * 60)
    
    # Override config with command line args
    if args.start_date:
        config.backtest.start_date = args.start_date
    if args.end_date:
        config.backtest.end_date = args.end_date
    if args.initial_capital:
        config.trading.initial_capital = args.initial_capital
    
    print(f"Backtest Period: {config.backtest.start_date} to {config.backtest.end_date}")
    print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    print(f"Universe: {', '.join(config.universe)}")
    print(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    print("-" * 60)
    
    # Initialize and run backtest
    if engine.initialize():
        engine.start_trading()
        
        # Print results
        print("\n" + "=" * 60)
        print("           BACKTEST RESULTS")
        print("=" * 60)
        
        summary = engine.get_portfolio_summary()
        print(f"Final Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"Total Return: {summary['total_return']:.2%}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        
        # Strategy performance
        print("\n" + "-" * 40)
        print("Strategy Performance:")
        print("-" * 40)
        
        strategy_perf = engine.get_strategy_performance()
        for strategy, metrics in strategy_perf.items():
            print(f"{strategy}:")
            print(f"  - Success Rate: {metrics['success_rate']:.1%}")
            print(f"  - Total Signals: {metrics['total_signals']}")
            print(f"  - Weight: {metrics['weight']:.1%}")
        
        # Save results
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
            engine.save_state(filename)
            print(f"\nResults saved to: {filename}")
    
    else:
        print("Failed to initialize trading engine")

def run_live_trading(engine: TradingEngine, args):
    """Run live trading mode"""
    print("=" * 60)
    print("           TRADING ALGORITHM - LIVE MODE")
    print("=" * 60)
    print("⚠️  WARNING: This is live trading mode!")
    print("⚠️  Real money will be at risk!")
    print("=" * 60)
    
    # Confirmation
    if not args.no_confirm:
        confirm = input("Are you sure you want to start live trading? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Live trading cancelled.")
            return
    
    # Override config
    if args.initial_capital:
        config.trading.initial_capital = args.initial_capital
    
    print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    print(f"Universe: {', '.join(config.universe)}")
    print(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    print("-" * 60)
    
    # Initialize and start
    if engine.initialize():
        print("Starting live trading...")
        print("Press Ctrl+C to stop safely")
        
        try:
            engine.start_trading()
            
            # Keep running until interrupted
            while engine.state.is_running:
                import time
                time.sleep(10)
                
                # Print periodic updates
                summary = engine.get_portfolio_summary()
                print(f"\r[{summary['last_update'].strftime('%H:%M:%S')}] "
                      f"Portfolio: ${summary['portfolio_value']:,.2f} "
                      f"P&L: ${summary['total_pnl']:,.2f} "
                      f"Positions: {summary['num_positions']}", end='')
                
        except KeyboardInterrupt:
            print("\n\nShutting down safely...")
            engine.stop_trading()
            
            # Option to close all positions
            if engine.state.positions:
                close_all = input("Close all positions? (yes/no): ")
                if close_all.lower() == 'yes':
                    engine.force_close_all_positions()
            
            print("Trading stopped.")
    else:
        print("Failed to initialize trading engine")

def run_paper_trading(engine: TradingEngine, args):
    """Run paper trading mode"""
    print("=" * 60)
    print("           TRADING ALGORITHM - PAPER MODE")
    print("=" * 60)
    
    # Override config
    if args.initial_capital:
        config.trading.initial_capital = args.initial_capital
    
    print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    print(f"Universe: {', '.join(config.universe)}")
    print(f"Strategies: {', '.join(config.strategy.enabled_strategies)}")
    print("-" * 60)
    
    # Initialize and start
    if engine.initialize():
        print("Starting paper trading...")
        print("Press Ctrl+C to stop")
        
        try:
            engine.start_trading()
            
            # Keep running until interrupted
            while engine.state.is_running:
                import time
                time.sleep(10)
                
                # Print periodic updates
                summary = engine.get_portfolio_summary()
                print(f"\r[{summary['last_update'].strftime('%H:%M:%S')}] "
                      f"Portfolio: ${summary['portfolio_value']:,.2f} "
                      f"P&L: ${summary['total_pnl']:,.2f} "
                      f"Positions: {summary['num_positions']}", end='')
                
        except KeyboardInterrupt:
            print("\n\nStopping paper trading...")
            engine.stop_trading()
            
            # Print final summary
            summary = engine.get_portfolio_summary()
            print(f"\nFinal Portfolio Value: ${summary['portfolio_value']:,.2f}")
            print(f"Total Return: {summary['total_return']:.2%}")
            print(f"Total Trades: {summary['total_trades']}")
            
            print("Paper trading stopped.")
    else:
        print("Failed to initialize trading engine")

def run_analysis(engine: TradingEngine, args):
    """Run analysis mode"""
    print("=" * 60)
    print("           TRADING ALGORITHM - ANALYSIS MODE")
    print("=" * 60)
    
    # Initialize
    if engine.initialize():
        # Get recent data for analysis
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = engine.data_manager.get_data_with_indicators(
            config.universe,
            start_date,
            end_date
        )
        
        # Generate and display signals
        signals = engine.strategy_manager.generate_signals(data)
        
        print(f"Generated {len(signals)} signals:")
        print("-" * 60)
        
        for signal in signals[:10]:  # Show top 10
            print(f"{signal.symbol}: {signal.action.upper()} "
                  f"(strength: {signal.strength:.2f}, "
                  f"strategy: {signal.metadata.get('strategy', 'unknown')})")
        
        # Strategy analysis
        print("\n" + "-" * 40)
        print("Strategy Analysis:")
        print("-" * 40)
        
        strategy_analysis = engine.strategy_manager.get_strategy_analysis(data)
        
        for strategy, analysis in strategy_analysis.get('strategy_analysis', {}).items():
            print(f"\n{strategy}:")
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        # Market overview
        print("\n" + "-" * 40)
        print("Market Overview:")
        print("-" * 40)
        
        for symbol in config.universe[:5]:  # Show top 5
            if symbol in data:
                df = data[symbol]
                current_price = df['close'].iloc[-1]
                change = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
                volume = df['volume'].iloc[-1]
                
                print(f"{symbol}: ${current_price:.2f} "
                      f"({change:+.2%}) "
                      f"Vol: {volume:,.0f}")
    
    else:
        print("Failed to initialize trading engine")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced Trading Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest
  python main.py backtest --start-date 2023-01-01 --end-date 2023-12-31

  # Run paper trading
  python main.py paper --initial-capital 50000

  # Run analysis
  python main.py analysis

  # Run live trading (use with caution!)
  python main.py live --initial-capital 10000
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Trading mode')
    
    # Backtest mode
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-capital', type=float, help='Initial capital')
    backtest_parser.add_argument('--save-results', action='store_true', help='Save results to file')
    
    # Paper trading mode
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--initial-capital', type=float, help='Initial capital')
    
    # Live trading mode
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--initial-capital', type=float, help='Initial capital')
    live_parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation')
    
    # Analysis mode
    analysis_parser = subparsers.add_parser('analysis', help='Run analysis')
    
    # Global options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--config-file', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load custom config if provided
    if args.config_file:
        # TODO: Implement config file loading
        pass
    
    # Print welcome message
    print("🚀 Advanced Trading Algorithm")
    print(f"   Version: 1.0.0")
    print(f"   Mode: {args.mode}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Validate mode
    if not args.mode:
        parser.print_help()
        return
    
    # Set trading mode in config
    config.trading.mode = args.mode
    
    # Create and run trading engine
    engine = TradingEngine(config)
    
    try:
        if args.mode == 'backtest':
            run_backtest(engine, args)
        elif args.mode == 'paper':
            run_paper_trading(engine, args)
        elif args.mode == 'live':
            run_live_trading(engine, args)
        elif args.mode == 'analysis':
            run_analysis(engine, args)
        else:
            print(f"Unknown mode: {args.mode}")
            parser.print_help()
            
    except Exception as e:
        logging.error(f"Error running trading algorithm: {str(e)}")
        raise
    
    print("\nGoodbye! 👋")

if __name__ == "__main__":
    main()