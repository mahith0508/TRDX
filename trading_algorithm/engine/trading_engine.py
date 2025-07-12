"""
Main Trading Engine
Coordinates all components of the trading algorithm
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import asyncio
import threading
import time
from dataclasses import dataclass
import json

# Import components
from ..data.data_manager import DataManager
from ..strategies.strategy_manager import StrategyManager
from ..strategies.base_strategy import Signal, Position

logger = logging.getLogger(__name__)

@dataclass
class TradingState:
    """Current state of the trading system"""
    mode: str  # 'backtest', 'paper', 'live'
    is_running: bool
    last_update: datetime
    portfolio_value: float
    cash: float
    positions: Dict[str, Position]
    daily_pnl: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades_today: int
    total_trades: int

class TradingEngine:
    """
    Main trading engine that coordinates all components
    """
    
    def __init__(self, config):
        self.config = config
        self.state = TradingState(
            mode=config.trading.mode,
            is_running=False,
            last_update=datetime.now(),
            portfolio_value=config.trading.initial_capital,
            cash=config.trading.initial_capital,
            positions={},
            daily_pnl=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            trades_today=0,
            total_trades=0
        )
        
        # Initialize components
        self.data_manager = DataManager(config)
        self.strategy_manager = StrategyManager(config)
        
        # Trading execution
        self.pending_orders = {}
        self.order_history = []
        self.trade_history = []
        self.performance_history = []
        
        # Risk management
        self.max_portfolio_risk = config.trading.max_portfolio_risk
        self.max_drawdown_limit = config.trading.max_drawdown
        self.position_size_limit = config.trading.position_size
        
        # Threading for live trading
        self.trading_thread = None
        self.update_interval = 60  # seconds
        
        logger.info(f"Trading Engine initialized in {self.state.mode} mode")
    
    def initialize(self):
        """Initialize the trading engine"""
        try:
            # Get historical data for strategy initialization
            logger.info("Fetching historical data for initialization...")
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Get data with indicators
            data = self.data_manager.get_data_with_indicators(
                self.config.universe,
                start_date,
                end_date
            )
            
            if not data:
                raise ValueError("No data available for initialization")
            
            # Initialize strategies
            logger.info("Initializing strategies...")
            self.strategy_manager.initialize_strategies(data)
            
            # Initialize portfolio
            self._initialize_portfolio()
            
            logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {str(e)}")
            return False
    
    def _initialize_portfolio(self):
        """Initialize portfolio with starting capital"""
        self.state.portfolio_value = self.config.trading.initial_capital
        self.state.cash = self.config.trading.initial_capital
        self.state.positions = {}
        
        logger.info(f"Portfolio initialized with ${self.state.portfolio_value:,.2f}")
    
    def start_trading(self):
        """Start the trading engine"""
        if self.state.is_running:
            logger.warning("Trading engine is already running")
            return
        
        self.state.is_running = True
        
        if self.state.mode == 'backtest':
            self._run_backtest()
        elif self.state.mode in ['paper', 'live']:
            self._start_live_trading()
        
        logger.info(f"Trading engine started in {self.state.mode} mode")
    
    def stop_trading(self):
        """Stop the trading engine"""
        self.state.is_running = False
        
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join()
        
        logger.info("Trading engine stopped")
    
    def _run_backtest(self):
        """Run backtesting"""
        logger.info("Starting backtest...")
        
        # Get backtest data
        start_date = self.config.backtest.start_date
        end_date = self.config.backtest.end_date
        
        data = self.data_manager.get_data_with_indicators(
            self.config.universe,
            start_date,
            end_date
        )
        
        if not data:
            logger.error("No data available for backtesting")
            return
        
        # Get all unique dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        
        all_dates = sorted(all_dates)
        
        # Run backtest day by day
        for date in all_dates:
            if not self.state.is_running:
                break
            
            try:
                # Get data up to current date
                current_data = {}
                for symbol, df in data.items():
                    current_df = df[df.index <= date]
                    if not current_df.empty:
                        current_data[symbol] = current_df
                
                if current_data:
                    self._process_trading_day(current_data, date)
                
            except Exception as e:
                logger.error(f"Error processing date {date}: {str(e)}")
                continue
        
        # Calculate final performance
        self._calculate_final_performance()
        
        logger.info("Backtest completed")
    
    def _start_live_trading(self):
        """Start live trading in a separate thread"""
        def trading_loop():
            while self.state.is_running:
                try:
                    self._run_trading_cycle()
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    time.sleep(self.update_interval)
        
        self.trading_thread = threading.Thread(target=trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
    
    def _run_trading_cycle(self):
        """Run a single trading cycle"""
        current_date = datetime.now()
        
        # Get current market data
        try:
            # For live trading, get recent data
            end_date = current_date.strftime('%Y-%m-%d')
            start_date = (current_date - timedelta(days=100)).strftime('%Y-%m-%d')
            
            data = self.data_manager.get_data_with_indicators(
                self.config.universe,
                start_date,
                end_date
            )
            
            if data:
                self._process_trading_day(data, current_date)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
    
    def _process_trading_day(self, data: Dict[str, pd.DataFrame], current_date: datetime):
        """Process a single trading day"""
        logger.info(f"Processing trading day: {current_date.strftime('%Y-%m-%d')}")
        
        # Update positions with current prices
        self._update_positions(data, current_date)
        
        # Generate signals
        signals = self.strategy_manager.generate_signals(data)
        
        # Filter signals through risk management
        filtered_signals = self._apply_risk_management(signals)
        
        # Execute trades
        for signal in filtered_signals:
            self._execute_signal(signal, current_date)
        
        # Update portfolio metrics
        self._update_portfolio_metrics(current_date)
        
        # Check for exit conditions
        self._check_exit_conditions(data, current_date)
        
        self.state.last_update = current_date
    
    def _update_positions(self, data: Dict[str, pd.DataFrame], current_date: datetime):
        """Update existing positions with current prices"""
        for symbol, position in self.state.positions.items():
            if symbol in data and not data[symbol].empty:
                current_price = data[symbol]['close'].iloc[-1]
                position.update_price(current_price, current_date)
    
    def _apply_risk_management(self, signals: List[Signal]) -> List[Signal]:
        """Apply risk management filters to signals"""
        filtered_signals = []
        
        for signal in signals:
            # Check if we can afford the position
            if not self._can_afford_position(signal):
                continue
            
            # Check position limits
            if len(self.state.positions) >= self.config.trading.max_positions:
                continue
            
            # Check portfolio risk
            if self._calculate_portfolio_risk() >= self.max_portfolio_risk:
                continue
            
            # Check drawdown limits
            if self.state.max_drawdown >= self.max_drawdown_limit:
                logger.warning("Maximum drawdown reached, stopping new positions")
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def _can_afford_position(self, signal: Signal) -> bool:
        """Check if we can afford to take a position"""
        position_value = self.state.portfolio_value * self.position_size_limit
        required_cash = position_value if signal.action == 'buy' else 0
        
        return self.state.cash >= required_cash
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk"""
        if not self.state.positions:
            return 0.0
        
        total_risk = 0.0
        for position in self.state.positions.values():
            position_value = position.current_price * position.quantity
            position_risk = position_value / self.state.portfolio_value
            total_risk += position_risk
        
        return total_risk
    
    def _execute_signal(self, signal: Signal, current_date: datetime):
        """Execute a trading signal"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                return
            
            # Create order
            order = {
                'symbol': signal.symbol,
                'action': signal.action,
                'quantity': position_size,
                'price': signal.price,
                'timestamp': current_date,
                'signal_strength': signal.strength,
                'strategy': signal.metadata.get('strategy', 'unknown'),
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            # Execute order
            if self._execute_order(order):
                self.state.trades_today += 1
                self.state.total_trades += 1
                
                # Record trade
                self.trade_history.append(order)
                
                logger.info(f"Executed {signal.action} order for {signal.symbol}: "
                           f"{position_size} shares at ${signal.price:.2f}")
        
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {str(e)}")
    
    def _calculate_position_size(self, signal: Signal) -> int:
        """Calculate position size based on risk management"""
        # Use fixed percentage of portfolio
        position_value = self.state.portfolio_value * self.position_size_limit
        
        # Adjust based on signal strength
        adjusted_value = position_value * signal.strength
        
        # Calculate number of shares
        shares = int(adjusted_value / signal.price)
        
        return max(1, shares)
    
    def _execute_order(self, order: Dict[str, Any]) -> bool:
        """Execute a trading order"""
        try:
            symbol = order['symbol']
            action = order['action']
            quantity = order['quantity']
            price = order['price']
            
            # Calculate costs
            trade_value = quantity * price
            commission = trade_value * self.config.trading.commission
            slippage = trade_value * self.config.trading.slippage
            total_cost = commission + slippage
            
            if action == 'buy':
                total_required = trade_value + total_cost
                
                if self.state.cash >= total_required:
                    # Create position
                    position = Position(
                        symbol=symbol,
                        side='long',
                        entry_price=price,
                        current_price=price,
                        quantity=quantity,
                        entry_date=order['timestamp'],
                        current_date=order['timestamp'],
                        stop_loss=order.get('stop_loss'),
                        take_profit=order.get('take_profit')
                    )
                    
                    self.state.positions[symbol] = position
                    self.state.cash -= total_required
                    
                    return True
                else:
                    logger.warning(f"Insufficient cash for {symbol} purchase")
                    return False
                    
            elif action == 'sell':
                if symbol in self.state.positions:
                    position = self.state.positions[symbol]
                    
                    # Calculate PnL
                    pnl = (price - position.entry_price) * position.quantity - total_cost
                    
                    # Update cash
                    self.state.cash += trade_value - total_cost
                    
                    # Remove position
                    del self.state.positions[symbol]
                    
                    # Update PnL
                    self.state.total_pnl += pnl
                    
                    return True
                else:
                    logger.warning(f"No position to sell for {symbol}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return False
    
    def _check_exit_conditions(self, data: Dict[str, pd.DataFrame], current_date: datetime):
        """Check exit conditions for existing positions"""
        positions_to_close = []
        
        for symbol, position in self.state.positions.items():
            if symbol in data and not data[symbol].empty:
                current_price = data[symbol]['close'].iloc[-1]
                
                # Check stop loss
                if position.stop_loss and current_price <= position.stop_loss:
                    positions_to_close.append((symbol, current_price, 'stop_loss'))
                
                # Check take profit
                elif position.take_profit and current_price >= position.take_profit:
                    positions_to_close.append((symbol, current_price, 'take_profit'))
                
                # Check strategy exit conditions
                else:
                    # This would require access to strategy-specific logic
                    pass
        
        # Close positions
        for symbol, price, reason in positions_to_close:
            self._close_position(symbol, price, current_date, reason)
    
    def _close_position(self, symbol: str, price: float, current_date: datetime, reason: str):
        """Close a position"""
        if symbol in self.state.positions:
            position = self.state.positions[symbol]
            
            # Calculate trade value and costs
            trade_value = position.quantity * price
            commission = trade_value * self.config.trading.commission
            slippage = trade_value * self.config.trading.slippage
            total_cost = commission + slippage
            
            # Calculate PnL
            pnl = (price - position.entry_price) * position.quantity - total_cost
            
            # Update cash
            self.state.cash += trade_value - total_cost
            
            # Remove position
            del self.state.positions[symbol]
            
            # Update PnL
            self.state.total_pnl += pnl
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'action': 'sell',
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'exit_price': price,
                'entry_date': position.entry_date,
                'exit_date': current_date,
                'pnl': pnl,
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            
            logger.info(f"Closed position {symbol} at ${price:.2f} ({reason}): PnL = ${pnl:.2f}")
    
    def _update_portfolio_metrics(self, current_date: datetime):
        """Update portfolio performance metrics"""
        # Calculate portfolio value
        positions_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.state.positions.values()
        )
        
        self.state.portfolio_value = self.state.cash + positions_value
        
        # Calculate daily PnL
        initial_value = self.config.trading.initial_capital
        self.state.daily_pnl = self.state.portfolio_value - initial_value
        
        # Update max drawdown
        peak_value = max(self.state.portfolio_value, initial_value)
        current_drawdown = (peak_value - self.state.portfolio_value) / peak_value
        self.state.max_drawdown = max(self.state.max_drawdown, current_drawdown)
        
        # Store performance history
        self.performance_history.append({
            'date': current_date,
            'portfolio_value': self.state.portfolio_value,
            'cash': self.state.cash,
            'positions_value': positions_value,
            'daily_pnl': self.state.daily_pnl,
            'total_pnl': self.state.total_pnl,
            'max_drawdown': self.state.max_drawdown,
            'num_positions': len(self.state.positions)
        })
    
    def _calculate_final_performance(self):
        """Calculate final performance metrics"""
        if not self.performance_history:
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.performance_history)
        
        # Calculate returns
        df['returns'] = df['portfolio_value'].pct_change()
        
        # Calculate Sharpe ratio
        if len(df) > 1:
            mean_return = df['returns'].mean()
            std_return = df['returns'].std()
            
            if std_return > 0:
                risk_free_rate = self.config.backtest.risk_free_rate / 252  # Daily
                self.state.sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        # Log final results
        total_return = (self.state.portfolio_value - self.config.trading.initial_capital) / self.config.trading.initial_capital
        
        logger.info(f"Final Portfolio Value: ${self.state.portfolio_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Total Trades: {self.state.total_trades}")
        logger.info(f"Sharpe Ratio: {self.state.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {self.state.max_drawdown:.2%}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        positions_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.state.positions.values()
        )
        
        return {
            'mode': self.state.mode,
            'is_running': self.state.is_running,
            'last_update': self.state.last_update,
            'portfolio_value': self.state.portfolio_value,
            'cash': self.state.cash,
            'positions_value': positions_value,
            'num_positions': len(self.state.positions),
            'daily_pnl': self.state.daily_pnl,
            'total_pnl': self.state.total_pnl,
            'total_return': (self.state.portfolio_value - self.config.trading.initial_capital) / self.config.trading.initial_capital,
            'max_drawdown': self.state.max_drawdown,
            'sharpe_ratio': self.state.sharpe_ratio,
            'trades_today': self.state.trades_today,
            'total_trades': self.state.total_trades
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        return {
            symbol: {
                'side': pos.side,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'entry_date': pos.entry_date,
                'unrealized_pnl': pos.unrealized_pnl,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit
            }
            for symbol, pos in self.state.positions.items()
        }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history[-limit:]
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history"""
        return self.performance_history
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance"""
        return self.strategy_manager.get_strategy_performance()
    
    def force_close_all_positions(self):
        """Force close all positions (emergency stop)"""
        logger.warning("Force closing all positions...")
        
        # Get current data for latest prices
        try:
            current_date = datetime.now()
            end_date = current_date.strftime('%Y-%m-%d')
            start_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
            
            data = self.data_manager.get_data(
                list(self.state.positions.keys()),
                start_date,
                end_date
            )
            
            positions_to_close = list(self.state.positions.items())
            
            for symbol, position in positions_to_close:
                if symbol in data and not data[symbol].empty:
                    current_price = data[symbol]['close'].iloc[-1]
                    self._close_position(symbol, current_price, current_date, 'force_close')
        
        except Exception as e:
            logger.error(f"Error force closing positions: {str(e)}")
    
    def save_state(self, filepath: str):
        """Save current state to file"""
        state_data = {
            'config': self.config.dict(),
            'state': self.state.__dict__,
            'performance_history': self.performance_history,
            'trade_history': self.trade_history,
            'strategy_performance': self.strategy_manager.get_strategy_performance()
        }
        
        # Convert datetime objects to strings
        state_data['state']['last_update'] = state_data['state']['last_update'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"State saved to {filepath}")
    
    def __str__(self):
        return f"TradingEngine(mode={self.state.mode}, portfolio_value=${self.state.portfolio_value:,.2f})"
    
    def __repr__(self):
        return self.__str__()