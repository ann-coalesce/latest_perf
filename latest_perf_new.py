import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import credentials
import db_utils
import export_dashboard
import get_balance_nav
import get_exposure
import get_transfers
import metrics_utils
import qtd_perf_metrics
import sheet_utils


class PerformanceMetricsProcessor:
    """Handles the processing and updating of performance metrics data."""
    
    # Configuration constants
    EXPOSURE_ADJUSTMENTS = {
        'BRAVOWHISKEY_ACCOUNT': 100,
        'BRAVOWHISKEY_PM': 100,
        'BTC_FUND_LEVEL': 100
    }
    
    SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1wMaan-MEcLdGKGIIYm0r3iaw8p_bkuzQ-Nj55rhKGuI/edit?gid=0#gid=0'
    
    COLUMN_MAPPING = {
        "pm": "pm",
        "balance": "balance",
        "ITD_PnL": "itd_pnl",
        "ITD_PnL%": "itd_pnl_percentage",
        "MTD_PnL": "mtd_pnl",
        "MTD_PnL%": "mtd_pnl_percentage",
        "QTD_PnL": "qtd_pnl",
        "QTD_PnL%": "qtd_pnl_percentage",
        "YTD_PnL": "ytd_pnl",
        "YTD_PnL%": "ytd_pnl_percentage"
    }
    
    def __init__(self):
        self.current_time = self._get_current_time()
    
    def _get_current_time(self) -> datetime:
        """Get current UTC time with seconds and microseconds set to 0."""
        return datetime.now(timezone.utc).replace(second=0, microsecond=0)
    
    def _get_base_data(self, end_time: Optional[datetime] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch and prepare base dataframes for processing."""
        print(f"Processing data for: {self.current_time}")
        
        # Get processed dataframes
        bal_merged_df = get_balance_nav.get_bal_from_nav(end_time=end_time)
        nav_merged_df = get_balance_nav.get_nav(end_time=end_time)
        transfer_df = get_transfers.get_qtd_mtd_transfers(target_date=end_time)
        
        return bal_merged_df, nav_merged_df, transfer_df
    
    def _merge_grouping_data(self, bal_merged_df: pd.DataFrame) -> pd.DataFrame:
        """Merge balance data with PM grouping information."""
        grouping_df = pd.DataFrame(credentials.PM_DATA)
        return bal_merged_df.merge(grouping_df, on='pm', how='left')
    
    def _merge_transfer_data(self, bal_merged_df: pd.DataFrame, transfer_df: pd.DataFrame) -> pd.DataFrame:
        """Merge balance data with transfer information and fill missing values."""
        merged_df = pd.merge(bal_merged_df, transfer_df, on='pm', how='left')
        
        # Fill missing transfer values with 0
        transfer_columns = ['mtd_transfer', 'qtd_transfer', 'ytd_transfer', 'itd_transfer']
        for col in transfer_columns:
            merged_df[col] = merged_df[col].fillna(0)
        
        return merged_df
    
    def _calculate_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate P&L for different time periods."""
        # Calculate PnL (balance_inception is actually the sum of principal invested into a pm)
        df['ITD_PnL'] = df['balance'] - df['itd_transfer']
        df['QTD_PnL'] = df['balance'] - df['balance_quarter_start'] - df['qtd_transfer']
        df['MTD_PnL'] = df['balance'] - df['balance_month_start'] - df['mtd_transfer']
        df['YTD_PnL'] = df['balance'] - df['balance_year_start'] - df['ytd_transfer']
        
        # Fill missing values with ITD_PnL
        pnl_columns = ['MTD_PnL', 'QTD_PnL', 'YTD_PnL']
        for col in pnl_columns:
            df[col] = df[col].fillna(df['ITD_PnL'])
        
        # Force cash PnL to 0 (not including cash PnL to upper level groups)
        cash_mask = df['pm'] == 'cash'
        df.loc[cash_mask, ['ITD_PnL', 'MTD_PnL', 'QTD_PnL', 'YTD_PnL']] = 0
        
        return df
    
    def _calculate_returns(self, nav_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns using NAV data."""
        nav_df['ITD_PnL%'] = nav_df['nav'] / 1 - 1
        nav_df['QTD_PnL%'] = nav_df['nav'] / nav_df['nav_quarter_start'] - 1
        nav_df['MTD_PnL%'] = nav_df['nav'] / nav_df['nav_month_start'] - 1
        nav_df['YTD_PnL%'] = nav_df['nav'] / nav_df['nav_year_start'] - 1
        
        # Fill missing values with ITD_PnL%
        return_columns = ['QTD_PnL%', 'MTD_PnL%', 'YTD_PnL%']
        for col in return_columns:
            nav_df[col] = nav_df[col].fillna(nav_df['ITD_PnL%'])
        
        return nav_df
    
    def _apply_exposure_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply exposure adjustments for specific PM groups."""
        for pm_group, adjustment in self.EXPOSURE_ADJUSTMENTS.items():
            if hasattr(credentials, pm_group):
                mask = df['pm'].isin(getattr(credentials, pm_group))
                df.loc[mask, ['net_exposure', 'gross_exposure']] -= adjustment
        
        return df
    
    def _calculate_exposure_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate exposure ratios."""
        df['net_exposure_ratio'] = df['net_exposure'] / df['balance']
        df['gross_exposure_ratio'] = df['gross_exposure'] / df['balance']
        return df
    
    def _prepare_final_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the final dataframe with desired columns and clean data."""
        df['timestamp'] = self.current_time
        
        # Select and rename columns
        selected_columns = [
            'timestamp', 'pm', 'balance', 'ITD_PnL', 'ITD_PnL%', 
            'MTD_PnL', 'MTD_PnL%', 'QTD_PnL', 'QTD_PnL%', 
            'YTD_PnL', 'YTD_PnL%', 'net_exposure_ratio', 'gross_exposure_ratio'
        ]
        df = df[selected_columns]
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Clean data
        df[['net_exposure_ratio', 'gross_exposure_ratio']] = df[['net_exposure_ratio', 'gross_exposure_ratio']].fillna(0)
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _process_qtd_metrics(self, merged_df: pd.DataFrame, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Process QTD metrics and prepare for spreadsheet update."""
        qtd_metrics = qtd_perf_metrics.get_qtd_sharpe_df(ending_time=end_time)
        
        # Merge with performance data
        perf_subset = merged_df[['pm', 'qtd_pnl', 'qtd_pnl_percentage', 'timestamp']]
        qtd_metrics = pd.merge(qtd_metrics, perf_subset, on='pm', how='left')
        
        # Clean and prepare data
        qtd_metrics = qtd_metrics[['pm', 'sharpe', 'qtd_pnl', 'qtd_pnl_percentage', 'timestamp']]
        qtd_metrics.dropna(inplace=True)
        qtd_metrics = qtd_metrics.replace([np.inf, -np.inf], 0)
        
        return qtd_metrics
    
    def _update_spreadsheet(self, qtd_metrics: pd.DataFrame) -> None:
        """Update the Google Spreadsheet with QTD metrics."""
        quarter = (self.current_time.month - 1) // 3 + 1
        sheet_name = f'{self.current_time.year}Q{quarter}'
        month_in_quarter = (self.current_time.month - 1) % 3 + 1
        column = (month_in_quarter - 1) * 4 + month_in_quarter
        
        print(f"Updating spreadsheet: {sheet_name}, month: {month_in_quarter}, column: {column}")
        
        sheet_utils.set_dataframe(
            df=qtd_metrics, 
            url=self.SPREADSHEET_URL, 
            sheet_name=sheet_name, 
            col=column
        )
    
    def _get_usdtusd_price(self) -> float:
        """Get USDTUSD price from database."""
        price_sql_query = "SELECT price FROM prices WHERE symbol = 'USDTUSD';"
        return db_utils.get_db_table(query=price_sql_query).iloc[0, 0]
    
    def process_performance_metrics(self, end_time: Optional[datetime] = None) -> None:
        """Main method to process and update performance metrics."""
        # Get base data
        bal_merged_df, nav_merged_df, transfer_df = self._get_base_data(end_time)
        
        # Process balance data
        bal_merged_df = self._merge_grouping_data(bal_merged_df)
        bal_merged_df = self._merge_transfer_data(bal_merged_df, transfer_df)
        bal_merged_df = self._calculate_pnl(bal_merged_df)
        
        # Process NAV data
        nav_merged_df = self._calculate_returns(nav_merged_df)
        
        # Get exposure data
        exposure_df = get_exposure.get_exposure_df()
        
        # Merge all data
        merged_df = pd.merge(bal_merged_df, nav_merged_df, on='pm', how='inner')
        merged_df = pd.merge(merged_df, exposure_df, on='pm', how='left')
        
        # Apply adjustments and calculations
        merged_df = self._apply_exposure_adjustments(merged_df)
        merged_df = self._calculate_exposure_ratios(merged_df)
        merged_df = self._prepare_final_dataframe(merged_df)
        
        # Add metrics
        metrics_df = metrics_utils.get_metrics()
        merged_df = pd.merge(merged_df, metrics_df, on='pm', how='left')
        
        # print('Performance metrics table:')
        # print(merged_df)
        
        # Export dashboard and update database
        usdtusd_price = self._get_usdtusd_price()
        export_dashboard.export_dashboard(merged_df.copy(), usdtusd_price)
        db_utils.df_replace_table(table_name='performance_metrics', df=merged_df)
        print('Updated performance_metrics table')
        
        # Process and update QTD metrics
        qtd_metrics = self._process_qtd_metrics(merged_df, end_time)
        self._update_spreadsheet(qtd_metrics)
        
        print('QTD metrics:')
        print(qtd_metrics)
        print('Processing complete')


def job():
    """Main job function to process performance metrics."""
    processor = PerformanceMetricsProcessor()
    processor.process_performance_metrics()


def main():
    """Main entry point with timing."""
    start_time = time.time()
    job()
    end_time = time.time()
    print(f'Time taken: {end_time - start_time:.2f} seconds')


if __name__ == "__main__":
    main()
