"""
RSI (Relative Strength Index) Trading Strategy
Uses RSI to identify overbought and oversold conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    RSI strategy that uses the Relative Strength Index to identify overbought and oversold conditions
    Buys when RSI indicates oversold conditions and sells when overbought
    """
    
    def __init__(self, config: Dict):
        super().__init__("RSI", config)
        
        # Strategy specific parameters
        self.period = config.get('period', 14)
        self.overbought = config.get('overbought', 70)
        self.oversold = config.get('oversold', 30)
        self.min_volume = config.get('min_volume', 1000000)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.06)  # 6% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.08)  # 8% take profit
        
        # RSI specific parameters
        self.rsi_extreme_threshold = config.get('rsi_extreme_threshold', 80)  # Very overbought
        self.rsi_extreme_low = config.get('rsi_extreme_low', 20)  # Very oversold
        self.divergence_lookback = config.get('divergence_lookback', 20)
        
        logger.info(f"Initialized RSI Strategy with period: {self.period}, overbought: {self.overbought}, oversold: {self.oversold}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['returns', 'rsi', 'volume_sma', 'volatility', 'sma_20', 'sma_50']
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate RSI signals for all symbols"""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.period + 10:  # Need sufficient data for RSI calculation
                continue
            
            # Calculate RSI signals
            symbol_signals = self._calculate_rsi_signals(symbol, df)
            signals.extend(symbol_signals)
        
        # Sort signals by strength (strongest first)
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Generated {len(signals)} RSI signals")
        return signals
    
    def _calculate_rsi_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """Calculate RSI signals for a single symbol"""
        signals = []
        
        # Get the latest data point
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        current_date = df.index[-1]
        current_rsi = latest_data['rsi']
        
        # Skip if RSI is not available
        if pd.isna(current_rsi):
            return signals
        
        # Calculate RSI metrics
        rsi_metrics = self._calculate_rsi_metrics(df)
        
        # Generate signals based on RSI
        signal_strength = self._calculate_signal_strength(rsi_metrics)
        
        if signal_strength > self.get_signal_strength_threshold():
            # Determine action based on RSI levels
            action = 'hold'
            
            if current_rsi <= self.oversold:
                action = 'buy'  # RSI oversold, expect price to rise
            elif current_rsi >= self.overbought:
                action = 'sell'  # RSI overbought, expect price to fall
            
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
                        'rsi': current_rsi,
                        'rsi_trend': rsi_metrics['rsi_trend'],
                        'price_momentum': rsi_metrics['price_momentum'],
                        'volume_confirmation': rsi_metrics['volume_confirmation'],
                        'volatility': rsi_metrics['volatility'],
                        'trend_alignment': rsi_metrics['trend_alignment'],
                        'rsi_divergence': rsi_metrics['rsi_divergence'],
                        'extreme_rsi': rsi_metrics['extreme_rsi']
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _calculate_rsi_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various RSI-related metrics"""
        
        current_rsi = df['rsi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # RSI trend over last 5 periods
        rsi_trend = 0
        if len(df) >= 5:
            rsi_change = df['rsi'].iloc[-1] - df['rsi'].iloc[-5]
            rsi_trend = rsi_change / 5  # Average daily change
        
        # Price momentum
        price_momentum = df['returns'].tail(5).mean()
        
        # Volume confirmation
        recent_volume = df['volume'].tail(3).mean()
        avg_volume = df['volume_sma'].iloc[-1]
        volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Volatility
        volatility = df['volatility'].iloc[-1]
        
        # Trend alignment (RSI direction vs price trend)
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        price_trend = 1 if current_price > sma_20 > sma_50 else -1 if current_price < sma_20 < sma_50 else 0
        rsi_trend_direction = 1 if rsi_trend > 0 else -1 if rsi_trend < 0 else 0
        
        trend_alignment = 1 if price_trend == rsi_trend_direction else 0
        
        # RSI divergence detection
        rsi_divergence = self._detect_rsi_divergence(df)
        
        # Extreme RSI levels
        extreme_rsi = 0
        if current_rsi >= self.rsi_extreme_threshold:
            extreme_rsi = 1  # Extremely overbought
        elif current_rsi <= self.rsi_extreme_low:
            extreme_rsi = -1  # Extremely oversold
        
        return {
            'rsi_trend': rsi_trend,
            'price_momentum': price_momentum,
            'volume_confirmation': volume_confirmation,
            'volatility': volatility,
            'trend_alignment': trend_alignment,
            'rsi_divergence': rsi_divergence,
            'extreme_rsi': extreme_rsi
        }
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> float:
        """Detect RSI divergence with price"""
        if len(df) < self.divergence_lookback:
            return 0
        
        # Get recent data for divergence analysis
        recent_data = df.tail(self.divergence_lookback)
        
        # Find local highs and lows in price and RSI
        price_highs = recent_data['close'].rolling(window=3, center=True).max() == recent_data['close']
        price_lows = recent_data['close'].rolling(window=3, center=True).min() == recent_data['close']
        
        rsi_highs = recent_data['rsi'].rolling(window=3, center=True).max() == recent_data['rsi']
        rsi_lows = recent_data['rsi'].rolling(window=3, center=True).min() == recent_data['rsi']
        
        # Check for bearish divergence (price higher highs, RSI lower highs)
        price_high_values = recent_data.loc[price_highs, 'close']
        rsi_high_values = recent_data.loc[rsi_highs, 'rsi']
        
        bearish_divergence = 0
        if len(price_high_values) >= 2 and len(rsi_high_values) >= 2:
            if price_high_values.iloc[-1] > price_high_values.iloc[-2] and rsi_high_values.iloc[-1] < rsi_high_values.iloc[-2]:
                bearish_divergence = 1
        
        # Check for bullish divergence (price lower lows, RSI higher lows)
        price_low_values = recent_data.loc[price_lows, 'close']
        rsi_low_values = recent_data.loc[rsi_lows, 'rsi']
        
        bullish_divergence = 0
        if len(price_low_values) >= 2 and len(rsi_low_values) >= 2:
            if price_low_values.iloc[-1] < price_low_values.iloc[-2] and rsi_low_values.iloc[-1] > rsi_low_values.iloc[-2]:
                bullish_divergence = -1
        
        return bearish_divergence + bullish_divergence
    
    def _calculate_signal_strength(self, rsi_metrics: Dict[str, float]) -> float:
        """Calculate signal strength based on RSI metrics"""
        
        # Base strength from RSI extreme levels
        base_strength = 0
        current_rsi = rsi_metrics.get('rsi', 50)  # Default to neutral if not available
        
        if current_rsi <= self.oversold:
            base_strength = (self.oversold - current_rsi) / self.oversold
        elif current_rsi >= self.overbought:
            base_strength = (current_rsi - self.overbought) / (100 - self.overbought)
        
        # Boost for extreme RSI levels
        extreme_boost = abs(rsi_metrics['extreme_rsi']) * 0.3
        
        # Boost for RSI divergence
        divergence_boost = abs(rsi_metrics['rsi_divergence']) * 0.4
        
        # Boost for volume confirmation
        volume_boost = min(rsi_metrics['volume_confirmation'] - 1, 0.5) * 0.2
        
        # Boost for trend alignment
        trend_boost = rsi_metrics['trend_alignment'] * 0.2
        
        # Penalize high volatility (RSI works better in normal volatility)
        volatility_penalty = min(rsi_metrics['volatility'], 0.5) * 0.1
        
        # Combine factors
        strength = base_strength + extreme_boost + divergence_boost + volume_boost + trend_boost - volatility_penalty
        
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
        """Check if we should exit a position based on RSI conditions"""
        if len(current_data) < self.period:
            return False
        
        current_rsi = current_data['rsi'].iloc[-1]
        
        if pd.isna(current_rsi):
            return False
        
        # Exit long positions when RSI becomes overbought
        if position.side == 'long' and current_rsi >= self.overbought:
            return True
        
        # Exit short positions when RSI becomes oversold
        if position.side == 'short' and current_rsi <= self.oversold:
            return True
        
        # Exit when RSI returns to neutral zone
        if self.oversold < current_rsi < self.overbought:
            return True
        
        return False
    
    def get_rsi_analysis(self, symbol: str, df: pd.DataFrame) -> Dict[str, any]:
        """Get detailed RSI analysis for a symbol"""
        if len(df) < self.period:
            return {}
        
        current_rsi = df['rsi'].iloc[-1]
        rsi_metrics = self._calculate_rsi_metrics(df)
        
        # RSI classification
        if current_rsi >= self.rsi_extreme_threshold:
            rsi_condition = "Extremely Overbought"
        elif current_rsi >= self.overbought:
            rsi_condition = "Overbought"
        elif current_rsi <= self.rsi_extreme_low:
            rsi_condition = "Extremely Oversold"
        elif current_rsi <= self.oversold:
            rsi_condition = "Oversold"
        else:
            rsi_condition = "Neutral"
        
        return {
            'symbol': symbol,
            'current_rsi': current_rsi,
            'rsi_condition': rsi_condition,
            'rsi_trend': rsi_metrics['rsi_trend'],
            'divergence_detected': rsi_metrics['rsi_divergence'] != 0,
            'extreme_level': rsi_metrics['extreme_rsi'] != 0,
            'volume_confirmation': rsi_metrics['volume_confirmation'],
            'signal_strength': self._calculate_signal_strength(rsi_metrics)
        }
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        RSI Strategy:
        - RSI Period: {self.period} days
        - Overbought Level: {self.overbought}
        - Oversold Level: {self.oversold}
        - Extreme Overbought: {self.rsi_extreme_threshold}
        - Extreme Oversold: {self.rsi_extreme_low}
        - Stop Loss: {self.stop_loss_pct * 100:.1f}%
        - Take Profit: {self.take_profit_pct * 100:.1f}%
        - Minimum Volume: {self.min_volume:,}
        
        This strategy uses RSI to identify overbought and oversold conditions.
        It includes divergence detection, volume confirmation, and trend alignment.
        """