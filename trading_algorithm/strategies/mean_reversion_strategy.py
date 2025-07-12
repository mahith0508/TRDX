"""
Mean Reversion Trading Strategy
Identifies stocks that have deviated significantly from their mean and are likely to revert
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy that identifies stocks that have moved significantly away from their mean
    and are likely to revert back to their historical average
    """
    
    def __init__(self, config: Dict):
        super().__init__("Mean Reversion", config)
        
        # Strategy specific parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        self.min_volume = config.get('min_volume', 1000000)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.08)  # 8% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.05)  # 5% take profit (smaller for mean reversion)
        
        # Mean reversion specific parameters
        self.mean_window = config.get('mean_window', 50)  # Window for calculating mean
        self.std_window = config.get('std_window', 20)    # Window for calculating standard deviation
        
        logger.info(f"Initialized Mean Reversion Strategy with lookback: {self.lookback_period}, z-score threshold: {self.z_score_threshold}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['returns', 'sma_20', 'sma_50', 'volume_sma', 'volatility', 'rsi', 'bb_upper', 'bb_lower', 'bb_position']
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate mean reversion signals for all symbols"""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < max(self.lookback_period, self.mean_window):
                continue
            
            # Calculate mean reversion signals
            symbol_signals = self._calculate_mean_reversion_signals(symbol, df)
            signals.extend(symbol_signals)
        
        # Sort signals by strength (strongest first)
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Generated {len(signals)} mean reversion signals")
        return signals
    
    def _calculate_mean_reversion_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """Calculate mean reversion signals for a single symbol"""
        signals = []
        
        # Get the latest data point
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        current_date = df.index[-1]
        
        # Calculate mean reversion metrics
        reversion_metrics = self._calculate_reversion_metrics(df)
        
        # Generate signals based on mean reversion
        signal_strength = self._calculate_signal_strength(reversion_metrics)
        
        if signal_strength > self.get_signal_strength_threshold():
            # Determine action based on deviation from mean
            if reversion_metrics['z_score'] > self.z_score_threshold:
                action = 'sell'  # Price is too high, expect reversion down
            elif reversion_metrics['z_score'] < -self.z_score_threshold:
                action = 'buy'   # Price is too low, expect reversion up
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
                        'z_score': reversion_metrics['z_score'],
                        'price_deviation': reversion_metrics['price_deviation'],
                        'volatility_adjusted_deviation': reversion_metrics['volatility_adjusted_deviation'],
                        'bollinger_position': reversion_metrics['bollinger_position'],
                        'rsi': reversion_metrics['rsi'],
                        'volume_confirmation': reversion_metrics['volume_confirmation'],
                        'mean_price': reversion_metrics['mean_price'],
                        'volatility': reversion_metrics['volatility']
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _calculate_reversion_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various mean reversion metrics"""
        
        # Current price and mean
        current_price = df['close'].iloc[-1]
        mean_price = df['close'].rolling(window=self.mean_window).mean().iloc[-1]
        
        # Standard deviation for z-score calculation
        std_dev = df['close'].rolling(window=self.std_window).std().iloc[-1]
        
        # Z-score calculation
        z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
        
        # Price deviation from mean (percentage)
        price_deviation = (current_price - mean_price) / mean_price if mean_price > 0 else 0
        
        # Volatility-adjusted deviation
        volatility = df['volatility'].iloc[-1]
        volatility_adjusted_deviation = price_deviation / volatility if volatility > 0 else 0
        
        # Bollinger Band position (0 = at lower band, 1 = at upper band)
        bollinger_position = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
        
        # RSI for overbought/oversold confirmation
        rsi = df['rsi'].iloc[-1]
        
        # Volume confirmation - higher volume during extreme moves is good
        recent_volume = df['volume'].tail(3).mean()
        avg_volume = df['volume_sma'].iloc[-1]
        volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Trend strength - mean reversion works better in sideways markets
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        trend_strength = abs(sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
        
        return {
            'z_score': z_score,
            'price_deviation': price_deviation,
            'volatility_adjusted_deviation': volatility_adjusted_deviation,
            'bollinger_position': bollinger_position,
            'rsi': rsi,
            'volume_confirmation': volume_confirmation,
            'trend_strength': trend_strength,
            'mean_price': mean_price,
            'volatility': volatility
        }
    
    def _calculate_signal_strength(self, reversion_metrics: Dict[str, float]) -> float:
        """Calculate signal strength based on mean reversion metrics"""
        
        # Base strength from z-score magnitude
        base_strength = min(abs(reversion_metrics['z_score']) / 3.0, 1.0)  # Normalize to 0-1
        
        # Boost for extreme Bollinger Band positions
        bb_boost = 0
        if reversion_metrics['bollinger_position'] > 0.9 or reversion_metrics['bollinger_position'] < 0.1:
            bb_boost = 0.3
        
        # Boost for RSI confirmation
        rsi_boost = 0
        if reversion_metrics['rsi'] > 70 and reversion_metrics['z_score'] > 0:  # Overbought + high price
            rsi_boost = 0.2
        elif reversion_metrics['rsi'] < 30 and reversion_metrics['z_score'] < 0:  # Oversold + low price
            rsi_boost = 0.2
        
        # Boost for volume confirmation
        volume_boost = min(reversion_metrics['volume_confirmation'] - 1, 0.5) * 0.2
        
        # Penalize strong trends (mean reversion works better in sideways markets)
        trend_penalty = reversion_metrics['trend_strength'] * 0.3
        
        # Combine factors
        strength = base_strength + bb_boost + rsi_boost + volume_boost - trend_penalty
        
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
        """Check if we should exit a position based on mean reversion"""
        if len(current_data) < max(self.lookback_period, self.mean_window):
            return False
        
        # Calculate current reversion metrics
        current_reversion = self._calculate_reversion_metrics(current_data)
        
        # Exit if the price has reverted close to the mean
        if abs(current_reversion['z_score']) < 0.5:  # Close to mean
            return True
        
        # Exit if the reversion signal has weakened significantly
        if abs(current_reversion['z_score']) < self.z_score_threshold * 0.5:
            return True
        
        return False
    
    def get_optimal_hold_period(self, symbol: str, entry_metrics: Dict[str, float]) -> int:
        """Calculate optimal holding period based on mean reversion characteristics"""
        # More extreme deviations might take longer to revert
        base_period = 5  # days
        
        # Adjust based on z-score magnitude
        z_score_factor = min(abs(entry_metrics['z_score']) / 2.0, 2.0)
        
        # Adjust based on volatility (higher volatility = faster reversion)
        volatility_factor = 1 / (1 + entry_metrics['volatility']) if entry_metrics['volatility'] > 0 else 1
        
        optimal_period = int(base_period * z_score_factor * volatility_factor)
        
        return max(2, min(optimal_period, 20))  # Between 2 and 20 days
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        Mean Reversion Strategy:
        - Mean Window: {self.mean_window} days
        - Standard Deviation Window: {self.std_window} days
        - Z-Score Threshold: {self.z_score_threshold}
        - Stop Loss: {self.stop_loss_pct * 100:.1f}%
        - Take Profit: {self.take_profit_pct * 100:.1f}%
        - Minimum Volume: {self.min_volume:,}
        
        This strategy identifies stocks that have deviated significantly from their mean
        and are likely to revert back. It uses z-scores, Bollinger Bands, RSI, and volume
        confirmation to generate signals.
        """