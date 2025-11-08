"""
Strategy Manager
Coordinates all trading strategies and manages their signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import importlib

from .base_strategy import BaseStrategy, Signal
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .rsi_strategy import RSIStrategy
from .moving_average_strategy import MovingAverageCrossoverStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .pairs_trading_strategy import PairsTradingStrategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manages multiple trading strategies and combines their signals
    """
    
    def __init__(self, config):
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights = {}
        self.performance_tracker = {}
        self.signal_history = []
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Signal combination parameters
        self.signal_combination_method = config.strategy.signal_combination_method
        self.max_signals_per_strategy = config.strategy.max_signals_per_strategy
        self.signal_decay_factor = config.strategy.signal_decay_factor
        
        logger.info(f"Strategy Manager initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self):
        """Initialize all enabled strategies"""
        strategy_classes = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'rsi_strategy': RSIStrategy,
            'moving_average_crossover': MovingAverageCrossoverStrategy,
            'bollinger_bands': BollingerBandsStrategy,
            'pairs_trading': PairsTradingStrategy
        }
        
        for strategy_name in self.config.strategy.enabled_strategies:
            if strategy_name in strategy_classes:
                try:
                    strategy_config = getattr(self.config.strategy, strategy_name, {})
                    strategy = strategy_classes[strategy_name](strategy_config)
                    self.strategies[strategy_name] = strategy
                    self.strategy_weights[strategy_name] = 1.0 / len(self.config.strategy.enabled_strategies)
                    self.performance_tracker[strategy_name] = {
                        'total_signals': 0,
                        'successful_signals': 0,
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'last_updated': datetime.now()
                    }
                    logger.info(f"Initialized strategy: {strategy_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {strategy_name}: {str(e)}")
            else:
                logger.warning(f"Unknown strategy: {strategy_name}")
    
    def initialize_strategies(self, data: Dict[str, pd.DataFrame]):
        """Initialize all strategies with historical data"""
        for strategy_name, strategy in self.strategies.items():
            try:
                # Filter data for strategy requirements
                filtered_data = strategy.filter_universe(data)
                
                if filtered_data:
                    strategy.initialize(filtered_data)
                    logger.info(f"Initialized strategy {strategy_name} with {len(filtered_data)} symbols")
                else:
                    logger.warning(f"No suitable data for strategy {strategy_name}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy_name}: {str(e)}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate signals from all strategies"""
        all_signals = []
        strategy_signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Filter data for strategy
                filtered_data = strategy.filter_universe(data)
                
                if filtered_data:
                    # Generate signals
                    signals = strategy.calculate_signals(filtered_data)
                    
                    # Limit number of signals per strategy
                    signals = signals[:self.max_signals_per_strategy]
                    
                    # Add strategy name to signals
                    for signal in signals:
                        signal.metadata['strategy'] = strategy_name
                    
                    strategy_signals[strategy_name] = signals
                    all_signals.extend(signals)
                    
                    logger.info(f"Strategy {strategy_name} generated {len(signals)} signals")
                
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_name}: {str(e)}")
        
        # Combine and filter signals
        combined_signals = self._combine_signals(strategy_signals)
        
        # Store signal history
        self.signal_history.append({
            'timestamp': datetime.now(),
            'strategy_signals': strategy_signals,
            'combined_signals': combined_signals
        })
        
        logger.info(f"Generated {len(combined_signals)} combined signals from {len(all_signals)} total signals")
        
        return combined_signals
    
    def _combine_signals(self, strategy_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """Combine signals from different strategies"""
        if self.signal_combination_method == 'weighted_average':
            return self._combine_signals_weighted_average(strategy_signals)
        elif self.signal_combination_method == 'consensus':
            return self._combine_signals_consensus(strategy_signals)
        elif self.signal_combination_method == 'best_strategy':
            return self._combine_signals_best_strategy(strategy_signals)
        else:
            # Default: simple concatenation with deduplication
            return self._combine_signals_simple(strategy_signals)
    
    def _combine_signals_weighted_average(self, strategy_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """Combine signals using weighted average based on strategy performance"""
        combined_signals = {}
        
        for strategy_name, signals in strategy_signals.items():
            weight = self.strategy_weights.get(strategy_name, 1.0)
            
            for signal in signals:
                symbol = signal.symbol
                
                if symbol not in combined_signals:
                    combined_signals[symbol] = {
                        'buy_strength': 0.0,
                        'sell_strength': 0.0,
                        'signals': []
                    }
                
                # Weight the signal strength
                weighted_strength = signal.strength * weight
                
                if signal.action == 'buy':
                    combined_signals[symbol]['buy_strength'] += weighted_strength
                elif signal.action == 'sell':
                    combined_signals[symbol]['sell_strength'] += weighted_strength
                
                combined_signals[symbol]['signals'].append(signal)
        
        # Create final signals
        final_signals = []
        for symbol, data in combined_signals.items():
            buy_strength = data['buy_strength']
            sell_strength = data['sell_strength']
            signals = data['signals']
            
            # Determine final action
            if buy_strength > sell_strength and buy_strength > 0.5:
                action = 'buy'
                final_strength = buy_strength
            elif sell_strength > buy_strength and sell_strength > 0.5:
                action = 'sell'
                final_strength = sell_strength
            else:
                continue  # No strong signal
            
            # Create combined signal using the strongest individual signal as base
            base_signal = max(signals, key=lambda s: s.strength)
            
            combined_signal = Signal(
                symbol=symbol,
                action=action,
                strength=final_strength,
                price=base_signal.price,
                timestamp=base_signal.timestamp,
                stop_loss=base_signal.stop_loss,
                take_profit=base_signal.take_profit,
                metadata={
                    'combination_method': 'weighted_average',
                    'contributing_strategies': [s.metadata.get('strategy') for s in signals],
                    'signal_count': len(signals),
                    'buy_strength': buy_strength,
                    'sell_strength': sell_strength
                }
            )
            
            final_signals.append(combined_signal)
        
        return final_signals
    
    def _combine_signals_consensus(self, strategy_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """Combine signals requiring consensus from multiple strategies"""
        symbol_signals = {}
        
        # Group signals by symbol
        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                symbol = signal.symbol
                if symbol not in symbol_signals:
                    symbol_signals[symbol] = []
                symbol_signals[symbol].append(signal)
        
        # Require consensus
        consensus_signals = []
        min_consensus = max(2, len(self.strategies) // 2)  # At least 2 strategies or half
        
        for symbol, signals in symbol_signals.items():
            if len(signals) >= min_consensus:
                # Check if signals agree on direction
                buy_count = sum(1 for s in signals if s.action == 'buy')
                sell_count = sum(1 for s in signals if s.action == 'sell')
                
                if buy_count >= min_consensus:
                    action = 'buy'
                    relevant_signals = [s for s in signals if s.action == 'buy']
                elif sell_count >= min_consensus:
                    action = 'sell'
                    relevant_signals = [s for s in signals if s.action == 'sell']
                else:
                    continue  # No consensus
                
                # Create consensus signal
                avg_strength = np.mean([s.strength for s in relevant_signals])
                base_signal = max(relevant_signals, key=lambda s: s.strength)
                
                consensus_signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=avg_strength,
                    price=base_signal.price,
                    timestamp=base_signal.timestamp,
                    stop_loss=base_signal.stop_loss,
                    take_profit=base_signal.take_profit,
                    metadata={
                        'combination_method': 'consensus',
                        'contributing_strategies': [s.metadata.get('strategy') for s in relevant_signals],
                        'consensus_count': len(relevant_signals),
                        'total_signals': len(signals)
                    }
                )
                
                consensus_signals.append(consensus_signal)
        
        return consensus_signals
    
    def _combine_signals_best_strategy(self, strategy_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """Use signals from the best performing strategy"""
        # Find best performing strategy
        best_strategy = self._get_best_strategy()
        
        if best_strategy and best_strategy in strategy_signals:
            signals = strategy_signals[best_strategy]
            
            # Add metadata
            for signal in signals:
                signal.metadata['combination_method'] = 'best_strategy'
                signal.metadata['selected_strategy'] = best_strategy
            
            return signals
        
        # Fallback to simple combination
        return self._combine_signals_simple(strategy_signals)
    
    def _combine_signals_simple(self, strategy_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """Simple combination: concatenate and deduplicate"""
        all_signals = []
        
        for signals in strategy_signals.values():
            all_signals.extend(signals)
        
        # Deduplicate by symbol, keeping the strongest signal
        symbol_signals = {}
        for signal in all_signals:
            symbol = signal.symbol
            if symbol not in symbol_signals or signal.strength > symbol_signals[symbol].strength:
                symbol_signals[symbol] = signal
        
        # Add metadata
        for signal in symbol_signals.values():
            signal.metadata['combination_method'] = 'simple'
        
        return list(symbol_signals.values())
    
    def _get_best_strategy(self) -> Optional[str]:
        """Get the best performing strategy"""
        best_strategy = None
        best_score = -np.inf
        
        for strategy_name, metrics in self.performance_tracker.items():
            # Calculate composite score
            success_rate = metrics['successful_signals'] / max(metrics['total_signals'], 1)
            score = success_rate * 0.5 + metrics['sharpe_ratio'] * 0.3 + metrics['total_return'] * 0.2
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        return best_strategy
    
    def update_strategy_performance(self, strategy_name: str, signal: Signal, outcome: Dict[str, Any]):
        """Update strategy performance metrics"""
        if strategy_name in self.performance_tracker:
            tracker = self.performance_tracker[strategy_name]
            
            tracker['total_signals'] += 1
            
            if outcome.get('success', False):
                tracker['successful_signals'] += 1
            
            if 'return' in outcome:
                tracker['total_return'] += outcome['return']
            
            if 'sharpe_ratio' in outcome:
                tracker['sharpe_ratio'] = outcome['sharpe_ratio']
            
            if 'max_drawdown' in outcome:
                tracker['max_drawdown'] = outcome['max_drawdown']
            
            tracker['last_updated'] = datetime.now()
            
            # Update strategy weights based on performance
            self._update_strategy_weights()
    
    def _update_strategy_weights(self):
        """Update strategy weights based on performance"""
        total_score = 0
        strategy_scores = {}
        
        for strategy_name, metrics in self.performance_tracker.items():
            # Calculate performance score
            success_rate = metrics['successful_signals'] / max(metrics['total_signals'], 1)
            score = success_rate * 0.5 + metrics['sharpe_ratio'] * 0.3 + metrics['total_return'] * 0.2
            
            # Apply decay to older performance
            days_since_update = (datetime.now() - metrics['last_updated']).days
            decay_factor = np.exp(-days_since_update * self.signal_decay_factor)
            
            strategy_scores[strategy_name] = max(score * decay_factor, 0.1)  # Minimum weight
            total_score += strategy_scores[strategy_name]
        
        # Normalize weights
        if total_score > 0:
            for strategy_name in strategy_scores:
                self.strategy_weights[strategy_name] = strategy_scores[strategy_name] / total_score
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies"""
        performance = {}
        
        for strategy_name, metrics in self.performance_tracker.items():
            success_rate = metrics['successful_signals'] / max(metrics['total_signals'], 1)
            
            performance[strategy_name] = {
                'total_signals': metrics['total_signals'],
                'success_rate': success_rate,
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'weight': self.strategy_weights.get(strategy_name, 0),
                'last_updated': metrics['last_updated']
            }
        
        return performance
    
    def get_strategy_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get detailed analysis from all strategies"""
        analysis = {
            'strategy_performance': self.get_strategy_performance(),
            'strategy_analysis': {}
        }
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Get strategy-specific analysis
                if hasattr(strategy, 'get_rsi_analysis'):
                    analysis['strategy_analysis'][strategy_name] = strategy.get_rsi_analysis(data)
                elif hasattr(strategy, 'get_ma_analysis'):
                    analysis['strategy_analysis'][strategy_name] = strategy.get_ma_analysis(data)
                elif hasattr(strategy, 'get_bb_analysis'):
                    analysis['strategy_analysis'][strategy_name] = strategy.get_bb_analysis(data)
                elif hasattr(strategy, 'get_pairs_analysis'):
                    analysis['strategy_analysis'][strategy_name] = strategy.get_pairs_analysis(data)
                
            except Exception as e:
                logger.error(f"Error getting analysis for strategy {strategy_name}: {str(e)}")
        
        return analysis
    
    def get_strategy_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all strategies"""
        descriptions = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                descriptions[strategy_name] = strategy.get_strategy_description()
            except Exception as e:
                descriptions[strategy_name] = f"Error getting description: {str(e)}"
        
        return descriptions
    
    def reset_all_strategies(self):
        """Reset all strategies"""
        for strategy in self.strategies.values():
            strategy.reset()
        
        # Reset performance tracking
        for strategy_name in self.performance_tracker:
            self.performance_tracker[strategy_name] = {
                'total_signals': 0,
                'successful_signals': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'last_updated': datetime.now()
            }
            self.strategy_weights[strategy_name] = 1.0 / len(self.strategies)
        
        self.signal_history.clear()
        logger.info("All strategies reset")
    
    def get_signal_history(self, lookback_days: int = 30) -> List[Dict]:
        """Get signal history for the last N days"""
        cutoff_date = datetime.now() - pd.Timedelta(days=lookback_days)
        
        recent_history = [
            entry for entry in self.signal_history
            if entry['timestamp'] >= cutoff_date
        ]
        
        return recent_history
    
    def __str__(self):
        return f"StrategyManager with {len(self.strategies)} strategies: {list(self.strategies.keys())}"
    
    def __repr__(self):
        return self.__str__()