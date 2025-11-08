"""
Data Management Module
Handles fetching, processing, and caching of market data from various sources
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def get_data(self, symbols: List[str], start_date: str, end_date: str, 
                 interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for given symbols and date range"""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time data for given symbols"""
        pass

class YahooDataProvider(DataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self, rate_limit: int = 2000):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
    
    def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter if hour has passed
        if current_time - self.request_window_start > 3600:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we've exceeded rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 3600 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()
    
    def get_data(self, symbols: List[str], start_date: str, end_date: str, 
                 interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch historical data from Yahoo Finance"""
        self._rate_limit_check()
        
        data = {}
        for symbol in symbols:
            try:
                self.request_count += 1
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not df.empty:
                    # Clean and standardize data - keep only expected columns
                    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[expected_columns]
                    df.columns = [col.lower() for col in df.columns]
                    df.index.name = 'date'
                    df = df.dropna()
                    
                    # Add additional columns
                    df['symbol'] = symbol
                    df['returns'] = df['close'].pct_change()
                    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                    
                    data[symbol] = df
                    logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data found for symbol: {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return data
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time data from Yahoo Finance"""
        self._rate_limit_check()
        
        data = {}
        for symbol in symbols:
            try:
                self.request_count += 1
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price data
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    
                    data[symbol] = {
                        'price': current_price,
                        'volume': volume,
                        'timestamp': datetime.now(),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'beta': info.get('beta', 0),
                        '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                        '52_week_low': info.get('fiftyTwoWeekLow', 0)
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
                continue
        
        return data

class DataCache:
    """Database-backed cache for market data"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                returns REAL,
                log_returns REAL,
                created_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_data (
                symbol TEXT PRIMARY KEY,
                price REAL,
                volume INTEGER,
                timestamp TEXT,
                market_cap INTEGER,
                pe_ratio REAL,
                beta REAL,
                fifty_two_week_high REAL,
                fifty_two_week_low REAL,
                updated_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cached_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data for a symbol"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM market_data 
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        
        return None
    
    def cache_data(self, symbol: str, data: pd.DataFrame):
        """Cache data for a symbol"""
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data for caching
        df = data.copy()
        df.reset_index(inplace=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df['created_at'] = datetime.now().isoformat()
        
        # Keep only columns that exist in the database schema
        db_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns', 'created_at']
        df = df[db_columns]
        
        # Insert data (replace if exists)
        df.to_sql('market_data', conn, if_exists='append', index=False)
        
        conn.close()
    
    def cache_realtime_data(self, data: Dict[str, Dict]):
        """Cache real-time data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol, info in data.items():
            cursor.execute('''
                INSERT OR REPLACE INTO realtime_data 
                (symbol, price, volume, timestamp, market_cap, pe_ratio, beta, 
                 fifty_two_week_high, fifty_two_week_low, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, info['price'], info['volume'], info['timestamp'].isoformat(),
                info['market_cap'], info['pe_ratio'], info['beta'],
                info['52_week_high'], info['52_week_low'], datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()

class DataManager:
    """Main data manager class"""
    
    def __init__(self, config):
        self.config = config
        self.providers = self._init_providers()
        self.cache = DataCache()
        self.data_cache = {}
        
    def _init_providers(self) -> Dict[str, DataProvider]:
        """Initialize data providers"""
        providers = {}
        
        # Yahoo Finance
        if self.config.data.sources["yahoo"]["enabled"]:
            providers["yahoo"] = YahooDataProvider(
                rate_limit=self.config.data.sources["yahoo"]["rate_limit"]
            )
        
        # Add CSV provider as fallback
        try:
            from .csv_provider import CSVDataProvider
            providers["csv"] = CSVDataProvider()
            logger.info("CSV data provider initialized as fallback")
        except ImportError as e:
            logger.warning(f"Could not initialize CSV provider: {e}")
        
        # Add other providers as needed
        # providers["alpaca"] = AlpacaDataProvider()
        # providers["ib"] = InteractiveBrokersDataProvider()
        
        return providers
    
    def get_data(self, symbols: List[str], start_date: str, end_date: str, 
                 interval: str = "1d", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Get historical data for symbols with CSV fallback"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        data = {}
        symbols_to_fetch = []
        
        # Check cache first
        if use_cache:
            for symbol in symbols:
                cached_data = self.cache.get_cached_data(symbol, start_date, end_date)
                if cached_data is not None and not cached_data.empty:
                    data[symbol] = cached_data
                    logger.info(f"Retrieved cached data for {symbol}")
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = symbols
        
        # Fetch missing data with fallback
        if symbols_to_fetch:
            # Try primary provider first
            try:
                provider = self.providers[self.config.data.default_source]
                logger.info(f"Fetching data for {symbols_to_fetch} from {self.config.data.default_source}")
                new_data = provider.get_data(symbols_to_fetch, start_date, end_date, interval)
                
                # Cache new data
                for symbol, df in new_data.items():
                    self.cache.cache_data(symbol, df)
                    data[symbol] = df
                
                # Check if any symbols failed
                successful_symbols = set(new_data.keys())
                failed_symbols = [s for s in symbols_to_fetch if s not in successful_symbols]
                
                # Try CSV fallback for failed symbols
                if failed_symbols and "csv" in self.providers:
                    logger.info(f"Trying CSV fallback for failed symbols: {failed_symbols}")
                    csv_provider = self.providers["csv"]
                    csv_data = csv_provider.get_data(failed_symbols, start_date, end_date, interval)
                    
                    for symbol, df in csv_data.items():
                        self.cache.cache_data(symbol, df)
                        data[symbol] = df
                        logger.info(f"Successfully loaded {symbol} from CSV fallback")
                        
            except Exception as e:
                logger.error(f"Primary data provider failed: {e}")
                
                # Try CSV provider as complete fallback
                if "csv" in self.providers:
                    logger.info(f"Using CSV provider as complete fallback for {symbols_to_fetch}")
                    csv_provider = self.providers["csv"]
                    csv_data = csv_provider.get_data(symbols_to_fetch, start_date, end_date, interval)
                    
                    for symbol, df in csv_data.items():
                        self.cache.cache_data(symbol, df)
                        data[symbol] = df
                        logger.info(f"Successfully loaded {symbol} from CSV fallback")
                else:
                    logger.error("No CSV fallback available")
        
        return data
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time data for symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        provider = self.providers[self.config.data.default_source]
        data = provider.get_realtime_data(symbols)
        
        # Cache real-time data
        if data:
            self.cache.cache_realtime_data(data)
        
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a dataset"""
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Drop temporary columns
        df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
        
        return df
    
    def get_data_with_indicators(self, symbols: List[str], start_date: str, 
                                end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get data with technical indicators calculated"""
        data = self.get_data(symbols, start_date, end_date, interval)
        
        # Calculate indicators for each symbol
        for symbol in data:
            data[symbol] = self.calculate_technical_indicators(data[symbol])
        
        return data
    
    def get_correlation_matrix(self, symbols: List[str], start_date: str, 
                              end_date: str, method: str = "pearson") -> pd.DataFrame:
        """Calculate correlation matrix for given symbols"""
        data = self.get_data(symbols, start_date, end_date)
        
        # Create returns dataframe
        returns_df = pd.DataFrame()
        for symbol, df in data.items():
            returns_df[symbol] = df['returns']
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr(method=method)
        
        return correlation_matrix
    
    def save_data_to_csv(self, symbols: List[str], start_date: str, end_date: str, 
                         directory: str = "csv_data"):
        """
        Save historical data to CSV files for offline use
        
        Args:
            symbols: List of symbols to save
            start_date: Start date for data
            end_date: End date for data
            directory: Directory to save CSV files
        """
        logger.info(f"Saving data for {len(symbols)} symbols to CSV files...")
        
        # Get data without indicators (raw price data)
        data = self.get_data(symbols, start_date, end_date, use_cache=True)
        
        if "csv" in self.providers:
            csv_provider = self.providers["csv"]
            csv_provider.save_data_to_csv(data, directory)
            logger.info(f"Successfully saved {len(data)} symbols to {directory}")
        else:
            logger.error("CSV provider not available")
    
    def list_csv_symbols(self) -> List[str]:
        """List all symbols available in CSV files"""
        if "csv" in self.providers:
            csv_provider = self.providers["csv"]
            return csv_provider.list_available_symbols()
        return []
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean data"""
        cleaned_data = {}
        
        for symbol, df in data.items():
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol}")
                continue
            
            # Remove rows with missing values
            df_clean = df.dropna(subset=required_columns)
            
            # Check for data quality issues
            if len(df_clean) < len(df) * 0.8:  # More than 20% missing data
                logger.warning(f"High percentage of missing data for {symbol}")
            
            # Check for unrealistic values
            if (df_clean['high'] < df_clean['low']).any():
                logger.warning(f"Data quality issue detected for {symbol}: high < low")
            
            if (df_clean['close'] <= 0).any():
                logger.warning(f"Data quality issue detected for {symbol}: negative or zero prices")
                df_clean = df_clean[df_clean['close'] > 0]
            
            cleaned_data[symbol] = df_clean
        
        return cleaned_data