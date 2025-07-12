"""
Advanced Trading Algorithm
A comprehensive algorithmic trading system with multiple strategies, 
risk management, and backtesting capabilities.
"""

__version__ = "1.0.0"
__author__ = "Trading Algorithm Team"
__email__ = "contact@tradingalgorithm.com"

from .config.config import config
from .engine.trading_engine import TradingEngine
from .data.data_manager import DataManager
from .strategies.strategy_manager import StrategyManager

__all__ = [
    'config',
    'TradingEngine', 
    'DataManager',
    'StrategyManager'
]