"""
Bollinger Bands Trading Strategy
Trades based on price movements relative to Bollinger Bands
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy that trades based on price movements relative to the bands
    Buys when price touches lower band and sells when price touches upper band
    """
    
    def __init__(self, config: Dict):
        super().__init__("Bollinger Bands", config)
        
        # Strategy specific parameters
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2.0)
        self.min_volume = config.get('min_volume', 1000000)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.06)  # 6% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.08)  # 8% take profit
        
        # Bollinger Bands specific parameters
        self.band_penetration_threshold = config.get('band_penetration_threshold', 0.02)  # 2% penetration
        self.squeeze_threshold = config.get('squeeze_threshold', 0.05)  # 5% band width for squeeze
        self.expansion_threshold = config.get('expansion_threshold', 0.15)  # 15% band width for expansion
        
        logger.info(f"Initialized Bollinger Bands Strategy with period: {self.period}, std_dev: {self.std_dev}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['returns', 'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position', 
                'volume_sma', 'volatility', 'rsi', 'sma_20', 'sma_50']
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate Bollinger Bands signals for all symbols"""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.period + 10:
                continue
            
            # Calculate Bollinger Bands signals
            symbol_signals = self._calculate_bb_signals(symbol, df)
            signals.extend(symbol_signals)
        
        # Sort signals by strength (strongest first)
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Generated {len(signals)} Bollinger Bands signals")
        return signals
    
    def _calculate_bb_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """Calculate Bollinger Bands signals for a single symbol"""
        signals = []
        
        # Get the latest data point
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        current_date = df.index[-1]
        
        # Get Bollinger Bands values
        bb_upper = latest_data['bb_upper']
        bb_lower = latest_data['bb_lower']
        bb_middle = latest_data['bb_middle']
        bb_position = latest_data['bb_position']
        
        # Skip if BB values are not available
        if any(pd.isna(val) for val in [bb_upper, bb_lower, bb_middle, bb_position]):
            return signals
        
        # Calculate BB metrics
        bb_metrics = self._calculate_bb_metrics(df)
        
        # Generate signals based on BB
        signal_strength = self._calculate_signal_strength(bb_metrics)
        
        if signal_strength > self.get_signal_strength_threshold():
            # Determine action based on BB position
            action = 'hold'
            
            # Buy signal: price near lower band (oversold)
            if bb_position <= 0.2 and bb_metrics['band_bounce_potential'] > 0:
                action = 'buy'
            
            # Sell signal: price near upper band (overbought)
            elif bb_position >= 0.8 and bb_metrics['band_bounce_potential'] < 0:
                action = 'sell'
            
            # Squeeze breakout signals
            elif bb_metrics['squeeze_breakout']:
                if bb_metrics['breakout_direction'] > 0:
                    action = 'buy'
                else:
                    action = 'sell'
            
            if action != 'hold':
                # Calculate stop loss and take profit
                stop_loss = self._calculate_stop_loss(current_price, action, bb_metrics)
                take_profit = self._calculate_take_profit(current_price, action, bb_metrics)
                
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=current_date,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'bb_position': bb_position,
                        'bb_width': bb_metrics['bb_width'],
                        'band_bounce_potential': bb_metrics['band_bounce_potential'],
                        'squeeze_detected': bb_metrics['squeeze_detected'],
                        'squeeze_breakout': bb_metrics['squeeze_breakout'],
                        'breakout_direction': bb_metrics['breakout_direction'],
                        'volatility_regime': bb_metrics['volatility_regime'],
                        'volume_confirmation': bb_metrics['volume_confirmation'],
                        'rsi': bb_metrics['rsi'],
                        'trend_alignment': bb_metrics['trend_alignment']
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _calculate_bb_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various Bollinger Bands metrics"""
        
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]
        bb_position = df['bb_position'].iloc[-1]
        bb_width = df['bb_width'].iloc[-1]
        
        # Band width relative to price (volatility measure)
        bb_width_pct = bb_width / bb_middle if bb_middle > 0 else 0
        
        # Squeeze detection (narrow bands)
        squeeze_detected = bb_width_pct < self.squeeze_threshold
        
        # Expansion detection (wide bands)
        expansion_detected = bb_width_pct > self.expansion_threshold
        
        # Band bounce potential
        band_bounce_potential = 0
        if bb_position <= 0.2:  # Near lower band
            band_bounce_potential = 1  # Potential upward bounce
        elif bb_position >= 0.8:  # Near upper band
            band_bounce_potential = -1  # Potential downward bounce
        
        # Squeeze breakout detection
        squeeze_breakout = False
        breakout_direction = 0
        
        if len(df) >= 5:
            # Check if we're breaking out of a squeeze
            recent_width = df['bb_width'].tail(3).mean()
            prev_width = df['bb_width'].tail(8).iloc[:5].mean()
            
            if prev_width < self.squeeze_threshold and recent_width > prev_width * 1.2:
                squeeze_breakout = True
                # Determine breakout direction
                recent_returns = df['returns'].tail(3).mean()
                breakout_direction = 1 if recent_returns > 0 else -1
        
        # Volatility regime
        volatility_regime = 'normal'
        if bb_width_pct < self.squeeze_threshold:
            volatility_regime = 'low'
        elif bb_width_pct > self.expansion_threshold:
            volatility_regime = 'high'
        
        # Volume confirmation
        recent_volume = df['volume'].tail(3).mean()
        avg_volume = df['volume_sma'].iloc[-1]
        volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # RSI for confirmation
        rsi = df['rsi'].iloc[-1]
        
        # Trend alignment
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        trend_alignment = 0
        if current_price > sma_20 > sma_50:
            trend_alignment = 1  # Uptrend
        elif current_price < sma_20 < sma_50:
            trend_alignment = -1  # Downtrend
        
        # Price momentum within bands
        price_momentum = df['returns'].tail(5).mean()
        
        # Distance to bands
        dist_to_upper = (bb_upper - current_price) / current_price
        dist_to_lower = (current_price - bb_lower) / current_price
        
        return {
            'bb_width': bb_width_pct,
            'band_bounce_potential': band_bounce_potential,
            'squeeze_detected': squeeze_detected,
            'expansion_detected': expansion_detected,
            'squeeze_breakout': squeeze_breakout,
            'breakout_direction': breakout_direction,
            'volatility_regime': volatility_regime,
            'volume_confirmation': volume_confirmation,
            'rsi': rsi,
            'trend_alignment': trend_alignment,
            'price_momentum': price_momentum,
            'dist_to_upper': dist_to_upper,
            'dist_to_lower': dist_to_lower
        }
    
    def _calculate_signal_strength(self, bb_metrics: Dict[str, float]) -> float:
        """Calculate signal strength based on Bollinger Bands metrics"""
        
        # Base strength from band position
        base_strength = 0
        if abs(bb_metrics['band_bounce_potential']) > 0:
            base_strength = 0.6
        
        # Boost for squeeze breakout
        breakout_boost = 0
        if bb_metrics['squeeze_breakout']:
            breakout_boost = 0.4
        
        # Boost for extreme band positions
        extreme_boost = 0
        if bb_metrics['dist_to_upper'] < 0.01 or bb_metrics['dist_to_lower'] < 0.01:
            extreme_boost = 0.3
        
        # Boost for volume confirmation
        volume_boost = min(bb_metrics['volume_confirmation'] - 1, 0.5) * 0.2
        
        # RSI confirmation
        rsi_boost = 0
        if not pd.isna(bb_metrics['rsi']):
            if bb_metrics['band_bounce_potential'] > 0 and bb_metrics['rsi'] < 30:
                rsi_boost = 0.2  # Oversold bounce
            elif bb_metrics['band_bounce_potential'] < 0 and bb_metrics['rsi'] > 70:
                rsi_boost = 0.2  # Overbought reversal
        
        # Trend alignment boost
        trend_boost = 0
        if bb_metrics['band_bounce_potential'] > 0 and bb_metrics['trend_alignment'] > 0:
            trend_boost = 0.15  # Bullish bounce in uptrend
        elif bb_metrics['band_bounce_potential'] < 0 and bb_metrics['trend_alignment'] < 0:
            trend_boost = 0.15  # Bearish bounce in downtrend
        
        # Volatility regime adjustment
        volatility_adjustment = 0
        if bb_metrics['volatility_regime'] == 'low':
            volatility_adjustment = 0.1  # Favor squeeze breakouts
        elif bb_metrics['volatility_regime'] == 'high':
            volatility_adjustment = -0.1  # Penalize high volatility
        
        # Combine factors
        strength = (base_strength + breakout_boost + extreme_boost + volume_boost + 
                   rsi_boost + trend_boost + volatility_adjustment)
        
        # Normalize to 0-1 range
        strength = max(0, min(1, strength))
        
        return strength
    
    def _calculate_stop_loss(self, current_price: float, action: str, bb_metrics: Dict[str, float]) -> float:
        """Calculate stop loss price based on BB levels"""
        if action == 'buy':
            # For buy signals, use lower band as stop loss reference
            return current_price * (1 - self.stop_loss_pct)
        else:  # sell/short
            # For sell signals, use upper band as stop loss reference
            return current_price * (1 + self.stop_loss_pct)
    
    def _calculate_take_profit(self, current_price: float, action: str, bb_metrics: Dict[str, float]) -> float:
        """Calculate take profit price based on BB levels"""
        if action == 'buy':
            # For buy signals, target middle band or upper band
            return current_price * (1 + self.take_profit_pct)
        else:  # sell/short
            # For sell signals, target middle band or lower band
            return current_price * (1 - self.take_profit_pct)
    
    def should_exit_position(self, position, current_data: pd.DataFrame) -> bool:
        """Check if we should exit a position based on BB conditions"""
        if len(current_data) < self.period:
            return False
        
        latest_data = current_data.iloc[-1]
        bb_position = latest_data['bb_position']
        
        # Exit long positions when price reaches upper band
        if position.side == 'long' and bb_position >= 0.8:
            return True
        
        # Exit short positions when price reaches lower band
        if position.side == 'short' and bb_position <= 0.2:
            return True
        
        # Exit when price returns to middle band
        if 0.4 <= bb_position <= 0.6:
            return True
        
        return False
    
    def get_bb_analysis(self, symbol: str, df: pd.DataFrame) -> Dict[str, any]:
        """Get detailed Bollinger Bands analysis for a symbol"""
        if len(df) < self.period:
            return {}
        
        latest_data = df.iloc[-1]
        bb_metrics = self._calculate_bb_metrics(df)
        
        current_price = latest_data['close']
        bb_upper = latest_data['bb_upper']
        bb_lower = latest_data['bb_lower']
        bb_middle = latest_data['bb_middle']
        bb_position = latest_data['bb_position']
        
        # Determine position relative to bands
        if bb_position >= 0.8:
            band_position = "Near Upper Band (Overbought)"
        elif bb_position <= 0.2:
            band_position = "Near Lower Band (Oversold)"
        elif 0.4 <= bb_position <= 0.6:
            band_position = "Near Middle Band (Neutral)"
        else:
            band_position = "Between Bands"
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'bb_position': bb_position,
            'band_position': band_position,
            'bb_width': bb_metrics['bb_width'],
            'volatility_regime': bb_metrics['volatility_regime'],
            'squeeze_detected': bb_metrics['squeeze_detected'],
            'squeeze_breakout': bb_metrics['squeeze_breakout'],
            'signal_strength': self._calculate_signal_strength(bb_metrics)
        }
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        Bollinger Bands Strategy:
        - Period: {self.period} days
        - Standard Deviation: {self.std_dev}
        - Squeeze Threshold: {self.squeeze_threshold * 100:.1f}%
        - Expansion Threshold: {self.expansion_threshold * 100:.1f}%
        - Stop Loss: {self.stop_loss_pct * 100:.1f}%
        - Take Profit: {self.take_profit_pct * 100:.1f}%
        - Minimum Volume: {self.min_volume:,}
        
        This strategy trades based on price movements relative to Bollinger Bands.
        It identifies squeeze conditions, breakouts, and mean reversion opportunities.
        """