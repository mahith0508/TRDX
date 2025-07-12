"""
Moving Average Crossover Strategy
Generates signals when fast MA crosses above/below slow MA
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover strategy that generates buy/sell signals
    when fast moving average crosses above/below slow moving average
    """
    
    def __init__(self, config: Dict):
        super().__init__("Moving Average Crossover", config)
        
        # Strategy specific parameters
        self.fast_ma = config.get('fast_ma', 10)
        self.slow_ma = config.get('slow_ma', 30)
        self.min_volume = config.get('min_volume', 1000000)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.07)  # 7% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.12)  # 12% take profit
        
        # MA specific parameters
        self.trend_filter = config.get('trend_filter', True)  # Use longer MA as trend filter
        self.trend_ma = config.get('trend_ma', 200)  # Trend filter MA period
        self.crossover_confirmation = config.get('crossover_confirmation', 2)  # Days to confirm crossover
        
        logger.info(f"Initialized MA Crossover Strategy with fast: {self.fast_ma}, slow: {self.slow_ma}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['returns', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 'volume_sma', 'volatility', 'rsi']
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate moving average crossover signals for all symbols"""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < max(self.slow_ma, self.trend_ma) + 10:
                continue
            
            # Calculate moving averages
            df = self._calculate_moving_averages(df)
            
            # Calculate crossover signals
            symbol_signals = self._calculate_crossover_signals(symbol, df)
            signals.extend(symbol_signals)
        
        # Sort signals by strength (strongest first)
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Generated {len(signals)} MA crossover signals")
        return signals
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required moving averages"""
        df = df.copy()
        
        # Calculate fast and slow MAs if not already present
        if f'sma_{self.fast_ma}' not in df.columns:
            df[f'sma_{self.fast_ma}'] = df['close'].rolling(window=self.fast_ma).mean()
        
        if f'sma_{self.slow_ma}' not in df.columns:
            df[f'sma_{self.slow_ma}'] = df['close'].rolling(window=self.slow_ma).mean()
        
        if f'sma_{self.trend_ma}' not in df.columns:
            df[f'sma_{self.trend_ma}'] = df['close'].rolling(window=self.trend_ma).mean()
        
        return df
    
    def _calculate_crossover_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """Calculate crossover signals for a single symbol"""
        signals = []
        
        # Get the latest data point
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        current_date = df.index[-1]
        
        # Get moving averages
        fast_ma_current = latest_data[f'sma_{self.fast_ma}']
        slow_ma_current = latest_data[f'sma_{self.slow_ma}']
        
        # Skip if MAs are not available
        if pd.isna(fast_ma_current) or pd.isna(slow_ma_current):
            return signals
        
        # Detect crossover
        crossover_metrics = self._detect_crossover(df)
        
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(df, crossover_metrics)
        
        if signal_strength > self.get_signal_strength_threshold():
            # Determine action based on crossover
            action = 'hold'
            
            if crossover_metrics['bullish_crossover']:
                action = 'buy'
            elif crossover_metrics['bearish_crossover']:
                action = 'sell'
            
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
                        'fast_ma': fast_ma_current,
                        'slow_ma': slow_ma_current,
                        'ma_spread': crossover_metrics['ma_spread'],
                        'crossover_strength': crossover_metrics['crossover_strength'],
                        'trend_alignment': crossover_metrics['trend_alignment'],
                        'volume_confirmation': crossover_metrics['volume_confirmation'],
                        'volatility': crossover_metrics['volatility'],
                        'rsi': crossover_metrics['rsi']
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _detect_crossover(self, df: pd.DataFrame) -> Dict[str, any]:
        """Detect MA crossover and calculate related metrics"""
        
        # Get recent MA values
        fast_ma = df[f'sma_{self.fast_ma}'].tail(self.crossover_confirmation + 1)
        slow_ma = df[f'sma_{self.slow_ma}'].tail(self.crossover_confirmation + 1)
        
        # Check for crossover
        bullish_crossover = False
        bearish_crossover = False
        
        # Bullish crossover: fast MA crosses above slow MA
        if len(fast_ma) > self.crossover_confirmation and len(slow_ma) > self.crossover_confirmation:
            # Check if fast MA was below slow MA and is now above
            if (fast_ma.iloc[-self.crossover_confirmation-1] < slow_ma.iloc[-self.crossover_confirmation-1] and
                fast_ma.iloc[-1] > slow_ma.iloc[-1]):
                bullish_crossover = True
            
            # Bearish crossover: fast MA crosses below slow MA
            if (fast_ma.iloc[-self.crossover_confirmation-1] > slow_ma.iloc[-self.crossover_confirmation-1] and
                fast_ma.iloc[-1] < slow_ma.iloc[-1]):
                bearish_crossover = True
        
        # MA spread (distance between fast and slow MA)
        ma_spread = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1] if slow_ma.iloc[-1] > 0 else 0
        
        # Crossover strength (how decisively the crossover occurred)
        crossover_strength = abs(ma_spread) if bullish_crossover or bearish_crossover else 0
        
        # Trend alignment (check if price is aligned with longer trend)
        current_price = df['close'].iloc[-1]
        trend_ma = df[f'sma_{self.trend_ma}'].iloc[-1]
        
        trend_alignment = 0
        if not pd.isna(trend_ma):
            if bullish_crossover and current_price > trend_ma:
                trend_alignment = 1  # Bullish crossover with uptrend
            elif bearish_crossover and current_price < trend_ma:
                trend_alignment = 1  # Bearish crossover with downtrend
        
        # Volume confirmation
        recent_volume = df['volume'].tail(3).mean()
        avg_volume = df['volume_sma'].iloc[-1]
        volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Volatility
        volatility = df['volatility'].iloc[-1]
        
        # RSI
        rsi = df['rsi'].iloc[-1]
        
        return {
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover,
            'ma_spread': ma_spread,
            'crossover_strength': crossover_strength,
            'trend_alignment': trend_alignment,
            'volume_confirmation': volume_confirmation,
            'volatility': volatility,
            'rsi': rsi
        }
    
    def _calculate_signal_strength(self, df: pd.DataFrame, crossover_metrics: Dict[str, any]) -> float:
        """Calculate signal strength based on crossover metrics"""
        
        # Base strength from crossover occurrence
        base_strength = 0.6 if crossover_metrics['bullish_crossover'] or crossover_metrics['bearish_crossover'] else 0
        
        # Boost for crossover strength (decisive crossover)
        crossover_boost = min(crossover_metrics['crossover_strength'] * 2, 0.3)
        
        # Boost for trend alignment
        trend_boost = crossover_metrics['trend_alignment'] * 0.3
        
        # Boost for volume confirmation
        volume_boost = min(crossover_metrics['volume_confirmation'] - 1, 0.5) * 0.2
        
        # RSI confirmation
        rsi_boost = 0
        if not pd.isna(crossover_metrics['rsi']):
            if crossover_metrics['bullish_crossover'] and crossover_metrics['rsi'] < 70:
                rsi_boost = 0.1  # Bullish crossover not overbought
            elif crossover_metrics['bearish_crossover'] and crossover_metrics['rsi'] > 30:
                rsi_boost = 0.1  # Bearish crossover not oversold
        
        # MA slope confirmation
        fast_ma_slope = self._calculate_ma_slope(df, self.fast_ma)
        slow_ma_slope = self._calculate_ma_slope(df, self.slow_ma)
        
        slope_boost = 0
        if crossover_metrics['bullish_crossover'] and fast_ma_slope > 0 and slow_ma_slope > 0:
            slope_boost = 0.2  # Both MAs trending up
        elif crossover_metrics['bearish_crossover'] and fast_ma_slope < 0 and slow_ma_slope < 0:
            slope_boost = 0.2  # Both MAs trending down
        
        # Penalize high volatility
        volatility_penalty = min(crossover_metrics['volatility'], 0.5) * 0.1
        
        # Combine factors
        strength = base_strength + crossover_boost + trend_boost + volume_boost + rsi_boost + slope_boost - volatility_penalty
        
        # Normalize to 0-1 range
        strength = max(0, min(1, strength))
        
        return strength
    
    def _calculate_ma_slope(self, df: pd.DataFrame, period: int) -> float:
        """Calculate the slope of moving average"""
        ma_values = df[f'sma_{period}'].tail(5)
        if len(ma_values) < 2:
            return 0
        
        # Calculate slope as percentage change per day
        slope = (ma_values.iloc[-1] - ma_values.iloc[0]) / (len(ma_values) - 1)
        return slope / ma_values.iloc[0] if ma_values.iloc[0] > 0 else 0
    
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
        """Check if we should exit a position based on MA crossover reversal"""
        if len(current_data) < max(self.slow_ma, self.trend_ma):
            return False
        
        # Calculate current moving averages
        current_data = self._calculate_moving_averages(current_data)
        crossover_metrics = self._detect_crossover(current_data)
        
        # Exit long position on bearish crossover
        if position.side == 'long' and crossover_metrics['bearish_crossover']:
            return True
        
        # Exit short position on bullish crossover
        if position.side == 'short' and crossover_metrics['bullish_crossover']:
            return True
        
        return False
    
    def get_ma_analysis(self, symbol: str, df: pd.DataFrame) -> Dict[str, any]:
        """Get detailed MA analysis for a symbol"""
        if len(df) < max(self.slow_ma, self.trend_ma):
            return {}
        
        df = self._calculate_moving_averages(df)
        crossover_metrics = self._detect_crossover(df)
        
        current_price = df['close'].iloc[-1]
        fast_ma = df[f'sma_{self.fast_ma}'].iloc[-1]
        slow_ma = df[f'sma_{self.slow_ma}'].iloc[-1]
        trend_ma = df[f'sma_{self.trend_ma}'].iloc[-1]
        
        # Determine current trend
        if current_price > fast_ma > slow_ma:
            trend = "Strong Uptrend"
        elif current_price > fast_ma and fast_ma < slow_ma:
            trend = "Weak Uptrend"
        elif current_price < fast_ma < slow_ma:
            trend = "Strong Downtrend"
        elif current_price < fast_ma and fast_ma > slow_ma:
            trend = "Weak Downtrend"
        else:
            trend = "Sideways"
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'trend_ma': trend_ma,
            'trend': trend,
            'bullish_crossover': crossover_metrics['bullish_crossover'],
            'bearish_crossover': crossover_metrics['bearish_crossover'],
            'ma_spread': crossover_metrics['ma_spread'],
            'signal_strength': self._calculate_signal_strength(df, crossover_metrics)
        }
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        Moving Average Crossover Strategy:
        - Fast MA: {self.fast_ma} days
        - Slow MA: {self.slow_ma} days
        - Trend Filter MA: {self.trend_ma} days
        - Crossover Confirmation: {self.crossover_confirmation} days
        - Stop Loss: {self.stop_loss_pct * 100:.1f}%
        - Take Profit: {self.take_profit_pct * 100:.1f}%
        - Minimum Volume: {self.min_volume:,}
        
        This strategy generates buy signals when fast MA crosses above slow MA
        and sell signals when fast MA crosses below slow MA. It includes trend
        filtering, volume confirmation, and slope analysis.
        """