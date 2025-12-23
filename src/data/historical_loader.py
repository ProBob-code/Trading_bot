"""
Historical Data Loader
======================

Handles loading historical data from various file formats (Excel, CSV)
and caching for faster subsequent loads.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from loguru import logger


class HistoricalLoader:
    """
    Loads and manages historical market data from files.
    
    Supports:
    - Excel files (.xlsx, .xls)
    - CSV files
    - Parquet files (for efficient storage)
    - Caching for faster reloads
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the historical data loader.
        
        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
    def load_excel(
        self,
        file_path: str,
        sheet_name: Union[str, int] = 0,
        header_row: Optional[int] = None,
        date_column: str = "Local Date",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load historical data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            header_row: Row number containing headers (auto-detected if None)
            date_column: Name of the datetime column
            use_cache: Whether to use cached version
            
        Returns:
            Normalized DataFrame with OHLCV data
        """
        cache_key = f"{file_path}_{sheet_name}"
        
        # Check memory cache
        if use_cache and cache_key in self._data_cache:
            logger.debug(f"Using memory cache for {file_path}")
            return self._data_cache[cache_key].copy()
        
        # Check disk cache
        cache_file = self._get_cache_path(file_path, sheet_name)
        if use_cache and cache_file.exists():
            logger.info(f"Loading from disk cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            self._data_cache[cache_key] = df
            return df.copy()
        
        # Load from Excel
        logger.info(f"Loading Excel file: {file_path}")
        
        try:
            # Auto-detect header row if needed
            if header_row is None:
                raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                header_row = self._find_header_row(raw_df, date_column)
            
            # Load with proper header
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
            
            # Normalize the data
            df = self._normalize_ohlcv(df, date_column)
            
            # Cache the result
            if use_cache:
                df.to_parquet(cache_file)
                self._data_cache[cache_key] = df
                logger.info(f"Cached to {cache_file}")
                
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
            
    def load_csv(
        self,
        file_path: str,
        date_column: str = "datetime",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of the datetime column
            use_cache: Whether to use cached version
            
        Returns:
            Normalized DataFrame with OHLCV data
        """
        cache_key = file_path
        
        if use_cache and cache_key in self._data_cache:
            return self._data_cache[cache_key].copy()
            
        cache_file = self._get_cache_path(file_path)
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            self._data_cache[cache_key] = df
            return df.copy()
            
        logger.info(f"Loading CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df = self._normalize_ohlcv(df, date_column)
            
            if use_cache:
                df.to_parquet(cache_file)
                self._data_cache[cache_key] = df
                
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
            
    def load_directory(
        self,
        directory: str,
        pattern: str = "*.xlsx",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all matching files from a directory.
        
        Args:
            directory: Directory path
            pattern: Glob pattern (e.g., "*.xlsx", "*.csv")
            **kwargs: Additional arguments for load_excel/load_csv
            
        Returns:
            Dictionary mapping filename to DataFrame
        """
        dir_path = Path(directory)
        result = {}
        
        for file_path in dir_path.glob(pattern):
            symbol = file_path.stem  # Use filename as symbol
            
            try:
                if file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = self.load_excel(str(file_path), **kwargs)
                elif file_path.suffix.lower() == '.csv':
                    df = self.load_csv(str(file_path), **kwargs)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                    
                if not df.empty:
                    result[symbol] = df
                    logger.info(f"Loaded {symbol}: {len(df)} rows")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return result
        
    def _find_header_row(self, df: pd.DataFrame, date_column: str) -> int:
        """Auto-detect the header row by finding the date column."""
        for idx, row in df.iterrows():
            if date_column in row.values:
                return idx
        return 0  # Default to first row
        
    def _normalize_ohlcv(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        Normalize DataFrame to standard OHLCV format.
        
        Standard format:
        - Index: datetime
        - Columns: open, high, low, close, volume
        """
        # Find date column (case-insensitive)
        date_col = None
        for col in df.columns:
            if col.lower() in [date_column.lower(), 'datetime', 'date', 'time', 'timestamp']:
                date_col = col
                break
                
        if date_col is None:
            # Try to use first column as datetime
            date_col = df.columns[0]
            logger.warning(f"Date column not found, using first column: {date_col}")
        
        # Set datetime index
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Normalize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['open', 'o']:
                column_mapping[col] = 'open'
            elif col_lower in ['high', 'h']:
                column_mapping[col] = 'high'
            elif col_lower in ['low', 'l']:
                column_mapping[col] = 'low'
            elif col_lower in ['close', 'c', 'last']:
                column_mapping[col] = 'close'
            elif col_lower in ['volume', 'vol', 'v']:
                column_mapping[col] = 'volume'
                
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df['volume'] = 0  # Default volume to 0 if missing
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Convert to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df.sort_index(inplace=True)
        
        return df[required]
        
    def _get_cache_path(self, file_path: str, sheet_name: Union[str, int] = 0) -> Path:
        """Generate cache file path."""
        file_name = Path(file_path).stem
        cache_name = f"{file_name}_{sheet_name}.parquet"
        return self.cache_dir / cache_name
        
    def clear_cache(self, file_path: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            file_path: Specific file to clear, or None to clear all
        """
        if file_path:
            cache_file = self._get_cache_path(file_path)
            if cache_file.exists():
                cache_file.unlink()
                
            # Clear memory cache
            keys_to_remove = [k for k in self._data_cache if file_path in k]
            for key in keys_to_remove:
                del self._data_cache[key]
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            self._data_cache.clear()
            
        logger.info("Cache cleared")
        
    def get_date_range(self, df: pd.DataFrame) -> tuple:
        """Get the date range of a DataFrame."""
        return df.index.min(), df.index.max()


# Convenience function
def load_stock_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Quick function to load stock data from a file.
    
    Args:
        file_path: Path to Excel or CSV file
        **kwargs: Additional arguments
        
    Returns:
        Normalized OHLCV DataFrame
    """
    loader = HistoricalLoader()
    
    path = Path(file_path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        return loader.load_excel(file_path, **kwargs)
    elif path.suffix.lower() == '.csv':
        return loader.load_csv(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
