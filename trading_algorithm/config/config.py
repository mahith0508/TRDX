"""
Configuration file for the trading algorithm
"""
import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class DataConfig(BaseModel):
    """Data source configuration"""
    default_source: str = "yahoo"
    sources: Dict[str, Dict[str, Any]] = {
        "yahoo": {
            "enabled": True,
            "rate_limit": 2000,  # requests per hour
        },
        "alpaca": {
            "enabled": bool(os.getenv("ALPACA_API_KEY")),
            "api_key": os.getenv("ALPACA_API_KEY"),
            "secret_key": os.getenv("ALPACA_SECRET_KEY"),
            "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        },
        "interactive_brokers": {
            "enabled": bool(os.getenv("IB_HOST")),
            "host": os.getenv("IB_HOST", "127.0.0.1"),
            "port": int(os.getenv("IB_PORT", "7497")),
            "client_id": int(os.getenv("IB_CLIENT_ID", "1")),
        },
        "binance": {
            "enabled": bool(os.getenv("BINANCE_API_KEY")),
            "api_key": os.getenv("BINANCE_API_KEY"),
            "secret_key": os.getenv("BINANCE_SECRET_KEY"),
            "testnet": bool(os.getenv("BINANCE_TESTNET", "true")),
        }
    }

class TradingConfig(BaseModel):
    """Trading configuration"""
    mode: str = "backtest"  # backtest, paper, live
    initial_capital: float = 100000.0
    max_positions: int = 10
    position_size: float = 0.1  # 10% of portfolio per position
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.001  # 0.1% slippage
    
    # Risk management
    max_portfolio_risk: float = 0.02  # 2% portfolio risk per trade
    max_drawdown: float = 0.15  # 15% max drawdown
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.10  # 10% take profit
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    
class StrategyConfig(BaseModel):
    """Strategy configuration"""
    enabled_strategies: List[str] = [
        "momentum",
        "mean_reversion",
        "rsi_strategy",
        "moving_average_crossover",
        "bollinger_bands",
        "pairs_trading"
    ]
    
    # Signal combination settings
    signal_combination_method: str = "weighted_average"
    max_signals_per_strategy: int = 5
    signal_decay_factor: float = 0.1
    
    # Individual signals trading (if False, uses signal combination)
    trade_individual_signals: bool = False
    
    # Strategy parameters
    momentum: Dict[str, Any] = {
        "lookback_period": 20,
        "threshold": 0.02,
        "min_volume": 1000000
    }
    
    mean_reversion: Dict[str, Any] = {
        "lookback_period": 20,
        "z_score_threshold": 2.0,
        "min_volume": 1000000
    }
    
    rsi_strategy: Dict[str, Any] = {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "min_volume": 1000000
    }
    
    moving_average_crossover: Dict[str, Any] = {
        "fast_ma": 10,
        "slow_ma": 30,
        "min_volume": 1000000
    }
    
    bollinger_bands: Dict[str, Any] = {
        "period": 20,
        "std_dev": 2.0,
        "min_volume": 1000000
    }
    
    pairs_trading: Dict[str, Any] = {
        "lookback_period": 60,
        "z_score_threshold": 2.0,
        "correlation_threshold": 0.8,
        "min_volume": 1000000
    }

class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    benchmark: str = "SPY"
    
    # Performance metrics
    risk_free_rate: float = 0.02  # 2% risk-free rate
    
class Config(BaseModel):
    """Main configuration class"""
    data: DataConfig = DataConfig()
    trading: TradingConfig = TradingConfig()
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()
    
    # Universe of assets
    universe: List[str] = [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX",
        "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "VXX", "EEM"
    ]
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_algorithm.log"
    
    # Database
    database_url: str = "sqlite:///trading_data.db"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Notifications
    enable_notifications: bool = True
    notification_channels: List[str] = ["console", "file"]
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    performance_update_frequency: int = 3600  # seconds

# Global configuration instance
config = Config()