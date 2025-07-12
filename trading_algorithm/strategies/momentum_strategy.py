"""
Momentum Trading Strategy
Buys stocks with strong upward momentum and sells stocks with weak momentum
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy that identifies stocks with strong price momentum
    Buys stocks that are trending upward and sells those trending downward
    """
    
    def __init__(self, config: Dict):
        super().__init__("Momentum", config)
        
        # Strategy specific parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.threshold = config.get('threshold', 0.02)  # 2% momentum threshold
        self.min_volume = config.get('min_volume', 1000000)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.15)  # 15% take profit
        
        logger.info(f"Initialized Momentum Strategy with lookback: {self.lookback_period}, threshold: {self.threshold}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['returns', 'sma_20', 'sma_50', 'volume_sma', 'volatility', 'rsi']
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate momentum signals for all symbols"""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.lookback_period:
                continue
            
            # Calculate momentum signals
            symbol_signals = self._calculate_momentum_signals(symbol, df)
            signals.extend(symbol_signals)
        
        # Sort signals by strength (strongest first)
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Generated {len(signals)} momentum signals")
        return signals
    
    def _calculate_momentum_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """Calculate momentum signals for a single symbol"""
        signals = []
        
        # Get the latest data point
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        current_date = df.index[-1]
        
        # Calculate momentum metrics
        momentum_metrics = self._calculate_momentum_metrics(df)
        
        # Generate signals based on momentum
        signal_strength = self._calculate_signal_strength(momentum_metrics)
        
        if signal_strength > self.get_signal_strength_threshold():
            # Determine action based on momentum direction
            if momentum_metrics['momentum_score'] > self.threshold:
                action = 'buy'
            elif momentum_metrics['momentum_score'] < -self.threshold:
                action = 'sell'
            else:
                action = 'hold'
            
            if action != 'hold':
                # Calculate stop loss and take profit
                stop_loss = self._calculate_stop_loss(current_price, action)
                take_profit = self._calculate_take_profit(current_price, action)
                
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=current_date,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'momentum_score': momentum_metrics['momentum_score'],
                        'price_momentum': momentum_metrics['price_momentum'],
                        'volume_momentum': momentum_metrics['volume_momentum'],
                        'trend_strength': momentum_metrics['trend_strength'],
                        'volatility': momentum_metrics['volatility'],
                        'rsi': momentum_metrics['rsi']
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _calculate_momentum_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various momentum metrics"""
        
        # Price momentum - percentage change over lookback period
        price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-self.lookback_period]) / df['close'].iloc[-self.lookback_period]
        
        # Volume momentum - recent volume vs historical average
        recent_volume = df['volume'].tail(5).mean()
        historical_volume = df['volume_sma'].iloc[-1]
        volume_momentum = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
        
        # Trend strength - based on moving averages
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        # Trend strength: positive if price > SMA20 > SMA50
        trend_strength = 0
        if current_price > sma_20 > sma_50:
            trend_strength = 1
        elif current_price < sma_20 < sma_50:
            trend_strength = -1
        
        # Volatility factor - prefer moderate volatility
        volatility = df['volatility'].iloc[-1]
        volatility_factor = 1 / (1 + volatility) if volatility > 0 else 0
        
        # RSI factor - avoid overbought/oversold
        rsi = df['rsi'].iloc[-1]
        rsi_factor = 1 - abs(rsi - 50) / 50 if not pd.isna(rsi) else 0.5
        
        # Combined momentum score
        momentum_score = (
            price_momentum * 0.4 +
            volume_momentum * 0.2 +
            trend_strength * 0.2 +
            volatility_factor * 0.1 +
            rsi_factor * 0.1
        )
        
        return {
            'momentum_score': momentum_score,
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'rsi': rsi,
            'volatility_factor': volatility_factor,
            'rsi_factor': rsi_factor
        }
    
    def _calculate_signal_strength(self, momentum_metrics: Dict[str, float]) -> float:
        """Calculate signal strength based on momentum metrics"""
        
        # Base strength from momentum score
        base_strength = abs(momentum_metrics['momentum_score'])
        
        # Boost strength for strong trends
        trend_boost = abs(momentum_metrics['trend_strength']) * 0.2
        
        # Boost for volume confirmation
        volume_boost = min(abs(momentum_metrics['volume_momentum']), 0.5) * 0.3
        
        # Penalize extreme RSI values
        rsi_penalty = 0
        if momentum_metrics['rsi'] > 80 or momentum_metrics['rsi'] < 20:
            rsi_penalty = 0.2
        
        # Combine factors
        strength = base_strength + trend_boost + volume_boost - rsi_penalty
        
        # Normalize to 0-1 range
        strength = max(0, min(1, strength))
        
        return strength
    
    def _calculate_stop_loss(self, current_price: float, action: str) -> float:
        """Calculate stop loss price"""
        if action == 'buy':
            return current_price * (1 - self.stop_loss_pct)
        else:  # sell/short
            return current_price * (1 + self.stop_loss_pct)
    
    def _calculate_take_profit(self, current_price: float, action: str) -> float:
        """Calculate take profit price"""
        if action == 'buy':
            return current_price * (1 + self.take_profit_pct)
        else:  # sell/short
            return current_price * (1 - self.take_profit_pct)
    
    def should_exit_position(self, position, current_data: pd.DataFrame) -> bool:
        """Check if we should exit a position based on momentum reversal"""
        if len(current_data) < self.lookback_period:
            return False
        
        # Calculate current momentum
        current_momentum = self._calculate_momentum_metrics(current_data)
        
        # Exit if momentum has reversed significantly
        if position.side == 'long' and current_momentum['momentum_score'] < -self.threshold:
            return True
        elif position.side == 'short' and current_momentum['momentum_score'] > self.threshold:
            return True
        
        return False
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        Momentum Strategy:
        - Lookback Period: {self.lookback_period} days
        - Momentum Threshold: {self.threshold * 100:.1f}%
        - Stop Loss: {self.stop_loss_pct * 100:.1f}%
        - Take Profit: {self.take_profit_pct * 100:.1f}%
        - Minimum Volume: {self.min_volume:,}
        
        This strategy identifies stocks with strong price momentum and trend strength.
        It considers price movement, volume confirmation, trend direction, and RSI levels.
        """