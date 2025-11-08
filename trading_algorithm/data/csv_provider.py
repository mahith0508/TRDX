"""
CSV Data Provider for importing data from CSV files
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os
from .data_manager import DataProvider

logger = logging.getLogger(__name__)

class CSVDataProvider(DataProvider):
    """CSV file data provider for importing historical data"""
    
    def __init__(self, csv_directory: str = "csv_data"):
        """
        Initialize CSV data provider
        
        Args:
            csv_directory: Directory containing CSV files
        """
        self.csv_directory = csv_directory
        os.makedirs(csv_directory, exist_ok=True)
        
    def get_data(self, symbols: List[str], start_date: str, end_date: str, 
                 interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Load data from CSV files
        
        Args:
            symbols: List of symbol names
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (currently only '1d' supported)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Try to load CSV file
                csv_file = os.path.join(self.csv_directory, f"{symbol}.csv")
                
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    
                    # Standardize column names
                    df.columns = df.columns.str.lower().str.replace(' ', '_')
                    
                    # Ensure required columns exist
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        logger.warning(f"CSV file {csv_file} missing columns: {missing_columns}")
                        continue
                    
                    # Convert date column to datetime
                    date_col = 'date' if 'date' in df.columns else df.index.name if df.index.name else 'index'
                    if date_col != 'date' and 'date' not in df.columns:
                        if df.index.name and 'date' in df.index.name.lower():
                            df.reset_index(inplace=True)
                            date_col = df.columns[0]
                        else:
                            # Assume first column is date
                            date_col = df.columns[0]
                    
                    df['date'] = pd.to_datetime(df[date_col])
                    df.set_index('date', inplace=True)
                    
                    # Filter by date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    # Sort by date
                    df.sort_index(inplace=True)
                    
                    # Calculate returns
                    df['returns'] = df['close'].pct_change()
                    
                    logger.info(f"Loaded {len(df)} rows of data for {symbol} from CSV")
                    data[symbol] = df
                    
                else:
                    logger.warning(f"CSV file not found for {symbol}: {csv_file}")
                    
            except Exception as e:
                logger.error(f"Error loading CSV data for {symbol}: {e}")
                continue
        
        return data
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        CSV provider doesn't support real-time data
        
        Returns:
            Empty dict
        """
        logger.warning("CSV data provider does not support real-time data")
        return {}
    
    def save_data_to_csv(self, data: Dict[str, pd.DataFrame], directory: str = None):
        """
        Save data to CSV files
        
        Args:
            data: Dictionary of symbol -> DataFrame
            directory: Directory to save files (defaults to self.csv_directory)
        """
        if directory is None:
            directory = self.csv_directory
            
        os.makedirs(directory, exist_ok=True)
        
        for symbol, df in data.items():
            try:
                csv_file = os.path.join(directory, f"{symbol}.csv")
                df.to_csv(csv_file)
                logger.info(f"Saved data for {symbol} to {csv_file}")
            except Exception as e:
                logger.error(f"Error saving CSV for {symbol}: {e}")
    
    def list_available_symbols(self) -> List[str]:
        """
        List all available CSV files
        
        Returns:
            List of symbol names
        """
        symbols = []
        
        try:
            for file in os.listdir(self.csv_directory):
                if file.endswith('.csv'):
                    symbol = file[:-4]  # Remove .csv extension
                    symbols.append(symbol)
        except Exception as e:
            logger.error(f"Error listing CSV files: {e}")
        
        return symbols