"""
Pairs Trading Strategy
Identifies correlated pairs of stocks and trades the spread between them
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from itertools import combinations

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy that identifies correlated pairs of stocks and trades
    the spread between them when it deviates from the mean
    """
    
    def __init__(self, config: Dict):
        super().__init__("Pairs Trading", config)
        
        # Strategy specific parameters
        self.lookback_period = config.get('lookback_period', 60)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.min_volume = config.get('min_volume', 1000000)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.10)  # 10% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.05)  # 5% take profit
        
        # Pairs specific parameters
        self.spread_window = config.get('spread_window', 20)  # Window for spread calculation
        self.cointegration_test = config.get('cointegration_test', True)
        self.max_pairs = config.get('max_pairs', 10)  # Maximum number of pairs to track
        self.rebalance_frequency = config.get('rebalance_frequency', 5)  # Days between rebalancing
        
        # Storage for pairs and their metrics
        self.pairs_data = {}
        self.active_pairs = []
        self.last_rebalance_date = None
        
        logger.info(f"Initialized Pairs Trading Strategy with correlation threshold: {self.correlation_threshold}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['returns', 'volume_sma', 'volatility']
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate pairs trading signals for all symbols"""
        signals = []
        
        try:
            # Update pairs if needed
            if self._should_update_pairs(data):
                self._update_pairs(data)
            
            # Generate signals for active pairs
            for pair_name, pair_data in self.pairs_data.items():
                if pair_name in self.active_pairs:
                    pair_signals = self._calculate_pair_signals(pair_name, pair_data, data)
                    signals.extend(pair_signals)
            
            # Sort signals by strength (strongest first)
            signals.sort(key=lambda x: x.strength, reverse=True)
            
            logger.info(f"Generated {len(signals)} pairs trading signals from {len(self.active_pairs)} active pairs")
        except Exception as e:
            logger.error(f"Error in calculate_signals: {str(e)}", exc_info=True)
        
        return signals
    
    def _should_update_pairs(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Check if we should update pairs"""
        if not self.pairs_data or self.last_rebalance_date is None:
            return True
        
        # Get the latest date from data
        latest_date = max(df.index[-1] for df in data.values())
        
        # Check if enough time has passed since last rebalance
        days_since_rebalance = (latest_date - self.last_rebalance_date).days
        
        return days_since_rebalance >= self.rebalance_frequency
    
    def _update_pairs(self, data: Dict[str, pd.DataFrame]):
        """Update pairs data and selection"""
        logger.info("Updating pairs data...")
        
        # Find eligible pairs
        eligible_pairs = self._find_eligible_pairs(data)
        
        # Calculate metrics for each pair
        pairs_metrics = {}
        for symbol1, symbol2 in eligible_pairs:
            if symbol1 in data and symbol2 in data:
                pair_name = f"{symbol1}_{symbol2}"
                metrics = self._calculate_pair_metrics(symbol1, symbol2, data[symbol1], data[symbol2])
                
                if metrics['correlation'] >= self.correlation_threshold:
                    pairs_metrics[pair_name] = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'metrics': metrics
                    }
        
        # Select best pairs
        self.pairs_data = self._select_best_pairs(pairs_metrics)
        self.active_pairs = list(self.pairs_data.keys())
        self.last_rebalance_date = max(df.index[-1] for df in data.values())
        
        logger.info(f"Updated to {len(self.active_pairs)} active pairs")
    
    def _find_eligible_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Find eligible pairs based on data availability and volume"""
        eligible_symbols = []
        
        for symbol, df in data.items():
            if (len(df) >= self.lookback_period and 
                df['volume'].median() >= self.min_volume):
                eligible_symbols.append(symbol)
        
        # Generate all possible pairs
        pairs = list(combinations(eligible_symbols, 2))
        
        return pairs
    
    def _calculate_pair_metrics(self, symbol1: str, symbol2: str, 
                               data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for a pair of stocks"""
        
        # Align data by date
        aligned_data = pd.merge(data1['close'], data2['close'], 
                              left_index=True, right_index=True, 
                              suffixes=('_1', '_2'))
        
        if len(aligned_data) < self.lookback_period:
            return {'correlation': 0, 'cointegration': 0, 'spread_std': 0}
        
        # Calculate correlation
        correlation = aligned_data['close_1'].corr(aligned_data['close_2'])
        
        # Calculate spread
        spread = aligned_data['close_1'] - aligned_data['close_2']
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Calculate cointegration score (simplified)
        cointegration_score = self._calculate_cointegration_score(aligned_data)
        
        # Calculate spread stationarity
        spread_stationarity = self._calculate_spread_stationarity(spread)
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        return {
            'correlation': correlation,
            'cointegration': cointegration_score,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_stationarity': spread_stationarity,
            'half_life': half_life,
            'data_points': len(aligned_data)
        }
    
    def _calculate_cointegration_score(self, aligned_data: pd.DataFrame) -> float:
        """Calculate simplified cointegration score"""
        # Simplified cointegration test using spread stationarity
        # Use close_1 and close_2 columns
        col1 = aligned_data.columns[0]
        col2 = aligned_data.columns[1]
        
        # Calculate spread
        spread = aligned_data[col1] - aligned_data[col2]
        
        # Test for mean reversion using autocorrelation
        if len(spread) > 1:
            autocorr = spread.autocorr(lag=1)
            return abs(autocorr)  # Higher autocorrelation suggests non-stationarity
        
        return 0
    
    def _calculate_spread_stationarity(self, spread: pd.Series) -> float:
        """Calculate spread stationarity score"""
        # Simple stationarity test using variance ratio
        if len(spread) < 20:
            return 0
        
        # Calculate variance ratios
        var_1 = spread.rolling(window=5).var().mean()
        var_2 = spread.rolling(window=10).var().mean()
        
        if var_2 > 0:
            variance_ratio = var_1 / var_2
            return 1 / (1 + abs(variance_ratio - 1))  # Closer to 1 is better
        
        return 0
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        if len(spread) < 10:
            return np.inf
        
        # Simple half-life calculation
        spread_lag = spread.shift(1)
        delta_spread = spread.diff()
        
        # Remove NaN values
        valid_idx = ~(pd.isna(spread_lag) | pd.isna(delta_spread))
        
        if valid_idx.sum() < 5:
            return np.inf
        
        # Regression: delta_spread = alpha + beta * spread_lag
        X = spread_lag[valid_idx]
        y = delta_spread[valid_idx]
        
        if len(X) > 0 and X.var() > 0:
            beta = np.cov(X, y)[0, 1] / X.var()
            if beta < 0:
                half_life = -np.log(2) / beta
                return min(half_life, 252)  # Cap at 1 year
        
        return np.inf
    
    def _select_best_pairs(self, pairs_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """Select best pairs based on multiple criteria"""
        if not pairs_metrics:
            return {}
        
        # Score each pair
        scored_pairs = []
        for pair_name, pair_data in pairs_metrics.items():
            metrics = pair_data['metrics']
            
            # Calculate composite score
            score = self._calculate_pair_score(metrics)
            
            scored_pairs.append((pair_name, pair_data, score))
        
        # Sort by score (highest first)
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Select top pairs
        selected_pairs = {}
        for pair_name, pair_data, score in scored_pairs[:self.max_pairs]:
            selected_pairs[pair_name] = pair_data
            logger.info(f"Selected pair {pair_name} with score {score:.3f}")
        
        return selected_pairs
    
    def _calculate_pair_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for a pair"""
        # Base score from correlation
        correlation_score = metrics['correlation']
        
        # Cointegration score (lower is better for cointegration)
        cointegration_score = 1 - min(metrics['cointegration'], 1)
        
        # Stationarity score
        stationarity_score = metrics['spread_stationarity']
        
        # Half-life score (shorter is better)
        half_life = metrics['half_life']
        half_life_score = 1 / (1 + half_life / 20) if half_life < np.inf else 0
        
        # Data availability score
        data_score = min(metrics['data_points'] / self.lookback_period, 1)
        
        # Composite score
        score = (correlation_score * 0.3 + 
                cointegration_score * 0.25 + 
                stationarity_score * 0.25 + 
                half_life_score * 0.15 + 
                data_score * 0.05)
        
        return score
    
    def _calculate_pair_signals(self, pair_name: str, pair_data: Dict, 
                               market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Calculate signals for a specific pair"""
        signals = []
        
        symbol1 = pair_data['symbol1']
        symbol2 = pair_data['symbol2']
        
        if symbol1 not in market_data or symbol2 not in market_data:
            return signals
        
        data1 = market_data[symbol1]
        data2 = market_data[symbol2]
        
        # Calculate current spread and z-score
        spread_metrics = self._calculate_current_spread_metrics(data1, data2)
        
        if spread_metrics['z_score'] is None:
            return signals
        
        # Generate signals based on z-score
        z_score = spread_metrics['z_score']
        current_date = max(data1.index[-1], data2.index[-1])
        
        signal_strength = self._calculate_pairs_signal_strength(spread_metrics, pair_data['metrics'])
        
        if signal_strength > self.get_signal_strength_threshold():
            # Generate pair signals
            if z_score > self.z_score_threshold:
                # Spread is too high: sell symbol1, buy symbol2
                sell_signal = Signal(
                    symbol=symbol1,
                    action='sell',
                    strength=signal_strength,
                    price=data1['close'].iloc[-1],
                    timestamp=current_date,
                    stop_loss=data1['close'].iloc[-1] * (1 + self.stop_loss_pct),
                    take_profit=data1['close'].iloc[-1] * (1 - self.take_profit_pct),
                    metadata={
                        'pair_name': pair_name,
                        'pair_symbol': symbol2,
                        'z_score': z_score,
                        'spread': spread_metrics['current_spread'],
                        'signal_type': 'pairs_trade',
                        'position_type': 'short_leg'
                    }
                )
                
                buy_signal = Signal(
                    symbol=symbol2,
                    action='buy',
                    strength=signal_strength,
                    price=data2['close'].iloc[-1],
                    timestamp=current_date,
                    stop_loss=data2['close'].iloc[-1] * (1 - self.stop_loss_pct),
                    take_profit=data2['close'].iloc[-1] * (1 + self.take_profit_pct),
                    metadata={
                        'pair_name': pair_name,
                        'pair_symbol': symbol1,
                        'z_score': z_score,
                        'spread': spread_metrics['current_spread'],
                        'signal_type': 'pairs_trade',
                        'position_type': 'long_leg'
                    }
                )
                
                signals.extend([sell_signal, buy_signal])
            
            elif z_score < -self.z_score_threshold:
                # Spread is too low: buy symbol1, sell symbol2
                buy_signal = Signal(
                    symbol=symbol1,
                    action='buy',
                    strength=signal_strength,
                    price=data1['close'].iloc[-1],
                    timestamp=current_date,
                    stop_loss=data1['close'].iloc[-1] * (1 - self.stop_loss_pct),
                    take_profit=data1['close'].iloc[-1] * (1 + self.take_profit_pct),
                    metadata={
                        'pair_name': pair_name,
                        'pair_symbol': symbol2,
                        'z_score': z_score,
                        'spread': spread_metrics['current_spread'],
                        'signal_type': 'pairs_trade',
                        'position_type': 'long_leg'
                    }
                )
                
                sell_signal = Signal(
                    symbol=symbol2,
                    action='sell',
                    strength=signal_strength,
                    price=data2['close'].iloc[-1],
                    timestamp=current_date,
                    stop_loss=data2['close'].iloc[-1] * (1 + self.stop_loss_pct),
                    take_profit=data2['close'].iloc[-1] * (1 - self.take_profit_pct),
                    metadata={
                        'pair_name': pair_name,
                        'pair_symbol': symbol1,
                        'z_score': z_score,
                        'spread': spread_metrics['current_spread'],
                        'signal_type': 'pairs_trade',
                        'position_type': 'short_leg'
                    }
                )
                
                signals.extend([buy_signal, sell_signal])
        
        return signals
    
    def _calculate_current_spread_metrics(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, float]:
        """Calculate current spread metrics"""
        # Align data
        aligned_data = pd.merge(data1['close'], data2['close'], 
                              left_index=True, right_index=True, 
                              suffixes=('_1', '_2'))
        
        if len(aligned_data) < self.spread_window:
            return {'current_spread': 0, 'z_score': None}
        
        # Calculate spread using the close_1 and close_2 columns
        spread = aligned_data['close_1'] - aligned_data['close_2']
        
        # Calculate rolling statistics
        spread_mean = spread.rolling(window=self.spread_window).mean()
        spread_std = spread.rolling(window=self.spread_window).std()
        
        # Current values
        current_spread = spread.iloc[-1]
        current_mean = spread_mean.iloc[-1]
        current_std = spread_std.iloc[-1]
        
        # Z-score
        z_score = None
        if current_std > 0:
            z_score = (current_spread - current_mean) / current_std
        
        return {
            'current_spread': current_spread,
            'spread_mean': current_mean,
            'spread_std': current_std,
            'z_score': z_score
        }
    
    def _calculate_pairs_signal_strength(self, spread_metrics: Dict[str, float], 
                                        pair_metrics: Dict[str, float]) -> float:
        """Calculate signal strength for pairs trade"""
        if spread_metrics['z_score'] is None:
            return 0
        
        # Base strength from z-score magnitude
        base_strength = min(abs(spread_metrics['z_score']) / 3.0, 1.0)
        
        # Boost for high correlation
        correlation_boost = (pair_metrics['correlation'] - self.correlation_threshold) * 2
        
        # Boost for good cointegration
        cointegration_boost = (1 - pair_metrics['cointegration']) * 0.3
        
        # Boost for short half-life
        half_life = pair_metrics['half_life']
        half_life_boost = 0.2 / (1 + half_life / 10) if half_life < np.inf else 0
        
        # Combine factors
        strength = base_strength + correlation_boost + cointegration_boost + half_life_boost
        
        # Normalize to 0-1 range
        strength = max(0, min(1, strength))
        
        return strength
    
    def should_exit_position(self, position, current_data: pd.DataFrame) -> bool:
        """Check if we should exit a pairs position"""
        # Check if this is a pairs trade
        if 'pair_name' not in position.metadata:
            return False
        
        # For pairs trades, we need to check the spread z-score
        pair_name = position.metadata['pair_name']
        
        if pair_name not in self.pairs_data:
            return True  # Exit if pair is no longer active
        
        # Check if spread has reverted to mean
        # This would require access to both symbols' data
        # For now, use a simple time-based exit
        days_held = (datetime.now() - position.entry_date).days
        
        return days_held > 30  # Exit after 30 days
    
    def get_pairs_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Get detailed pairs analysis"""
        analysis = {
            'active_pairs': len(self.active_pairs),
            'pairs_details': {}
        }
        
        for pair_name, pair_data in self.pairs_data.items():
            symbol1 = pair_data['symbol1']
            symbol2 = pair_data['symbol2']
            
            if symbol1 in data and symbol2 in data:
                spread_metrics = self._calculate_current_spread_metrics(data[symbol1], data[symbol2])
                
                analysis['pairs_details'][pair_name] = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation': pair_data['metrics']['correlation'],
                    'current_z_score': spread_metrics['z_score'],
                    'half_life': pair_data['metrics']['half_life'],
                    'signal_strength': self._calculate_pairs_signal_strength(spread_metrics, pair_data['metrics'])
                }
        
        return analysis
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        Pairs Trading Strategy:
        - Lookback Period: {self.lookback_period} days
        - Correlation Threshold: {self.correlation_threshold}
        - Z-Score Threshold: {self.z_score_threshold}
        - Spread Window: {self.spread_window} days
        - Max Pairs: {self.max_pairs}
        - Stop Loss: {self.stop_loss_pct * 100:.1f}%
        - Take Profit: {self.take_profit_pct * 100:.1f}%
        - Minimum Volume: {self.min_volume:,}
        
        This strategy identifies correlated pairs of stocks and trades the spread
        between them when it deviates from the mean. It includes cointegration
        testing and spread stationarity analysis.
        """