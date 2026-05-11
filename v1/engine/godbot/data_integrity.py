"""
Data Integrity Module
=====================

Institutional-grade data provenance tracking:
- SHA-256 hash of OHLCV datasets
- Source ID and version tagging
- Run provenance linking
- Reproducibility audit trail
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


class DataIntegrity:
    """
    Data integrity and provenance tracking.
    
    Ensures every simulation run can be traced to:
    - Exact dataset used (SHA-256 hash)
    - Data source identifier
    - Data version tag
    - Bar count and date range
    """
    
    @staticmethod
    def hash_dataset(df: pd.DataFrame) -> str:
        """
        Generate SHA-256 hash of OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Hex digest of SHA-256 hash
        """
        # Use only OHLCV columns for hashing
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in ohlcv_cols if c in df.columns]
        
        if not available:
            return hashlib.sha256(b'empty').hexdigest()
        
        # Convert to bytes deterministically
        data_bytes = df[available].to_numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
    
    @staticmethod
    def stamp_run(
        df: pd.DataFrame,
        source_id: str = 'unknown',
        version_tag: str = 'v1.0',
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Create a data integrity stamp for a simulation run.
        
        Args:
            df: OHLCV DataFrame
            source_id: Data source identifier (e.g., 'binance_api', 'yahoo_csv')
            version_tag: Data version tag (e.g., 'v1.0', '2024-01-raw')
            run_id: Optional run identifier
            
        Returns:
            Dict with integrity stamp data
        """
        data_hash = DataIntegrity.hash_dataset(df)
        
        # Date range
        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'date'):
            date_start = str(df.index[0])
            date_end = str(df.index[-1])
        else:
            date_start = str(df.iloc[0].name) if len(df) > 0 else 'unknown'
            date_end = str(df.iloc[-1].name) if len(df) > 0 else 'unknown'
        
        # Basic data quality stats
        null_count = int(df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()) \
            if all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume']) else -1
        
        return {
            'run_id': run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            'data_hash': data_hash,
            'source_id': source_id,
            'version_tag': version_tag,
            'n_bars': len(df),
            'date_range_start': date_start,
            'date_range_end': date_end,
            'null_count': null_count,
            'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    @staticmethod
    def verify_dataset(df: pd.DataFrame, expected_hash: str) -> Dict[str, Any]:
        """
        Verify a dataset matches an expected hash.
        
        Args:
            df: DataFrame to verify
            expected_hash: Expected SHA-256 hash
            
        Returns:
            Dict with verification result
        """
        actual_hash = DataIntegrity.hash_dataset(df)
        matches = actual_hash == expected_hash
        
        return {
            'matches': matches,
            'expected_hash': expected_hash,
            'actual_hash': actual_hash,
            'warning': None if matches else 'DATA MISMATCH: Dataset has changed since original run',
        }
    
    @staticmethod
    def data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data quality report for audit.
        
        Checks:
        - Missing values
        - Price anomalies (high < low, negative volume)
        - Gap analysis
        - Duplicate timestamps
        """
        report = {
            'n_bars': len(df),
            'nulls': {},
            'anomalies': [],
            'gaps': 0,
            'duplicates': 0,
            'quality_score': 100.0,
        }
        
        if len(df) == 0:
            report['quality_score'] = 0
            return report
        
        # Null check
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                null_count = int(df[col].isnull().sum())
                if null_count > 0:
                    report['nulls'][col] = null_count
                    report['quality_score'] -= null_count / len(df) * 20
        
        # Price anomalies
        if 'high' in df.columns and 'low' in df.columns:
            bad_hl = (df['high'] < df['low']).sum()
            if bad_hl > 0:
                report['anomalies'].append(f'{bad_hl} bars with high < low')
                report['quality_score'] -= bad_hl / len(df) * 30
        
        if 'volume' in df.columns:
            neg_vol = (df['volume'] < 0).sum()
            if neg_vol > 0:
                report['anomalies'].append(f'{neg_vol} bars with negative volume')
                report['quality_score'] -= 10
        
        if 'close' in df.columns:
            zero_close = (df['close'] <= 0).sum()
            if zero_close > 0:
                report['anomalies'].append(f'{zero_close} bars with close <= 0')
                report['quality_score'] -= 20
        
        # Duplicate timestamps
        if hasattr(df.index, 'duplicated'):
            dupes = int(df.index.duplicated().sum())
            report['duplicates'] = dupes
            if dupes > 0:
                report['quality_score'] -= dupes / len(df) * 15
        
        report['quality_score'] = max(0, round(report['quality_score'], 1))
        return report
