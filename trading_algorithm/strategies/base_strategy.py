"""
Base Strategy Class
All trading strategies inherit from this base class
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal data class"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # Signal strength (0-1)
    price: float
    timestamp: datetime
    quantity: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position data class"""
    symbol: str
    side: str  # 'long', 'short'
    entry_price: float
    current_price: float
    quantity: int
    entry_date: datetime
    current_date: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def update_price(self, price: float, date: datetime):
        """Update current price and calculate PnL"""
        self.current_price = price
        self.current_date = date
        
        if self.side == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.performance_metrics = {}
        self.is_initialized = False
        
        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.min_volume = config.get('min_volume', 1000000)
        
        logger.info(f"Initialized {self.name} strategy with config: {config}")
    
    @abstractmethod
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate trading signals based on market data"""
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        pass
    
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """Initialize strategy with historical data"""
        if not self._validate_data(data):
            raise ValueError("Invalid data provided for strategy initialization")
        
        self.is_initialized = True
        logger.info(f"Strategy {self.name} initialized successfully")
    
    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate input data"""
        for symbol, df in data.items():
            # Check minimum data requirements
            if len(df) < self.lookback_period:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.lookback_period}")
                return False
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol}")
                return False
            
            # Check for minimum volume
            if df['volume'].median() < self.min_volume:
                logger.warning(f"Low volume for {symbol}: {df['volume'].median()}")
                return False
        
        return True
    
    def filter_universe(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter universe based on strategy requirements"""
        filtered_data = {}
        
        for symbol, df in data.items():
            # Apply volume filter
            if df['volume'].median() >= self.min_volume:
                # Apply additional filters
                if self._passes_additional_filters(symbol, df):
                    filtered_data[symbol] = df
        
        logger.info(f"Filtered universe from {len(data)} to {len(filtered_data)} symbols")
        return filtered_data
    
    def _passes_additional_filters(self, symbol: str, data: pd.DataFrame) -> bool:
        """Apply additional symbol-specific filters"""
        # Check for sufficient price movement
        if data['close'].std() / data['close'].mean() < 0.01:  # Less than 1% volatility
            return False
        
        # For backtesting, check if data is recent relative to the data itself
        # Skip the age filter for backtesting as it's not relevant
        # The age filter is mainly for live trading to ensure fresh data
        
        return True
    
    def calculate_position_size(self, symbol: str, signal: Signal, 
                               portfolio_value: float, risk_per_trade: float) -> int:
        """Calculate position size based on risk management"""
        # Basic position sizing using fixed percentage
        if signal.action in ['buy', 'sell']:
            position_value = portfolio_value * risk_per_trade
            quantity = int(position_value / signal.price)
            return max(1, quantity)  # Minimum 1 share
        
        return 0
    
    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if position.stop_loss is None:
            return False
        
        if position.side == 'long':
            return current_price <= position.stop_loss
        else:  # short
            return current_price >= position.stop_loss
    
    def check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if position.take_profit is None:
            return False
        
        if position.side == 'long':
            return current_price >= position.take_profit
        else:  # short
            return current_price <= position.take_profit
    
    def update_positions(self, current_prices: Dict[str, float], current_date: datetime):
        """Update all positions with current prices"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol], current_date)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0
        for position in self.positions.values():
            total_value += position.current_price * position.quantity
        return total_value
    
    def get_open_positions(self) -> Dict[str, Position]:
        """Get all open positions"""
        return self.positions.copy()
    
    def close_position(self, symbol: str, price: float, date: datetime) -> float:
        """Close a position and return realized PnL"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.update_price(price, date)
            realized_pnl = position.unrealized_pnl
            del self.positions[symbol]
            
            logger.info(f"Closed position in {symbol} with PnL: {realized_pnl:.2f}")
            return realized_pnl
        
        return 0.0
    
    def open_position(self, signal: Signal, quantity: int, date: datetime):
        """Open a new position"""
        side = 'long' if signal.action == 'buy' else 'short'
        
        position = Position(
            symbol=signal.symbol,
            side=side,
            entry_price=signal.price,
            current_price=signal.price,
            quantity=quantity,
            entry_date=date,
            current_date=date,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        self.positions[signal.symbol] = position
        logger.info(f"Opened {side} position in {signal.symbol}: {quantity} shares at {signal.price}")
    
    def get_signal_strength_threshold(self) -> float:
        """Get minimum signal strength threshold"""
        return self.config.get('signal_threshold', 0.5)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns / returns.std() * np.sqrt(252)  # Annualized
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        return drawdown.min()
    
    def calculate_win_rate(self, trades: List[float]) -> float:
        """Calculate win rate from list of trade returns"""
        if len(trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade > 0)
        return winning_trades / len(trades)
    
    def update_performance_metrics(self, returns: pd.Series, trades: List[float]):
        """Update strategy performance metrics"""
        self.performance_metrics.update({
            'total_returns': returns.sum(),
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'win_rate': self.calculate_win_rate(trades),
            'total_trades': len(trades),
            'avg_trade_return': np.mean(trades) if trades else 0.0,
            'best_trade': max(trades) if trades else 0.0,
            'worst_trade': min(trades) if trades else 0.0
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'strategy_name': self.name,
            'performance_metrics': self.performance_metrics,
            'current_positions': len(self.positions),
            'total_signals': len(self.signals)
        }
    
    def reset(self):
        """Reset strategy state"""
        self.positions.clear()
        self.signals.clear()
        self.performance_metrics.clear()
        self.is_initialized = False
        logger.info(f"Strategy {self.name} reset")
    
    def __str__(self):
        return f"{self.name} Strategy (Positions: {len(self.positions)}, Signals: {len(self.signals)})"
    
    def __repr__(self):
        return self.__str__()