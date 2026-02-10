import time
from datetime import datetime, timezone
from typing import Optional, List, Optional
import logging

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
import alert

class PerformanceMetricsProcessor:
    """Handles the processing and updating of performance metrics data."""
    
    # Configuration constants
    EXPOSURE_ADJUSTMENTS = {
        # 'VICTORCHARLIE_ACCOUNT': 6.25,
        # 'VICTORCHARLIE_PM': 50,
        # 'BTC_FUND_LEVEL': 50
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
        self._pm_mapping_cache = None  # Cache for pm_mapping data
    
    def _get_current_time(self) -> datetime:
        """Get current UTC time with seconds and microseconds set to 0."""
        return datetime.now(timezone.utc).replace(second=0, microsecond=0)
        
    def _get_pm_mapping(self) -> pd.DataFrame:
        """
        Get PM mapping data from database with caching.
        
        Returns:
            DataFrame with pm_mapping data
        """
        if self._pm_mapping_cache is None:
            query = "SELECT * FROM pm_mapping;"
            self._pm_mapping_cache = db_utils.get_db_table(query=query)
            logging.info("Loaded pm_mapping from database")
        return self._pm_mapping_cache
    
    def _get_base_data(self, end_time: Optional[datetime] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch and prepare base dataframes for processing."""
        print(f"Processing data for: {self.current_time}")
        
        # Get processed dataframes
        bal_merged_df = get_balance_nav.get_bal_from_nav_test(end_time=end_time)
        nav_merged_df = get_balance_nav.get_nav(end_time=end_time)
        transfer_df = get_transfers.get_qtd_mtd_transfers(target_date=end_time)
        
        return bal_merged_df, nav_merged_df, transfer_df
    
    def _merge_grouping_data(self, bal_merged_df: pd.DataFrame) -> pd.DataFrame:
        """Merge balance data with PM grouping information."""
        grouping_df = self._get_pm_mapping()
        return bal_merged_df.merge(grouping_df, on='pm', how='left')
    
    def _merge_transfer_data(self, bal_merged_df: pd.DataFrame, transfer_df: pd.DataFrame) -> pd.DataFrame:
        """Merge balance data with transfer information and fill missing values."""
        merged_df = pd.merge(bal_merged_df, transfer_df, on='pm', how='right')
        
        # Fill missing transfer values with 0
        transfer_columns = ['mtd_transfer', 'qtd_transfer', 'ytd_transfer', 'itd_transfer', 'balance', 'balance_month_start', 'balance_quarter_start', 'balance_year_start']
        for col in transfer_columns:
            merged_df[col] = merged_df[col].fillna(0)
        
        return merged_df

    def _validate_pnl_aggregation(self, df_before: pd.DataFrame, df_after: pd.DataFrame, 
                             pnl_columns: List[str], 
                             tolerance_pct: float = 0.02, send_alert: bool = True,
                             custom_thresholds: dict = None, exclude_entities: list = None,
                             notional_thresholds: dict = None) -> bool:
        """
        Validate that new aggregated PnL values are reasonable compared to original values
        
        Args:
            df_before: DataFrame before aggregation replacement
            df_after: DataFrame after aggregation replacement  
            pnl_columns: List of PnL columns to validate
            tolerance_pct: Maximum allowed percentage difference (default 2%)
            send_alert: Whether to send alerts for validation failures
            custom_thresholds: Dict mapping entity names to custom tolerance percentages
            exclude_entities: List of entity names to exclude from validation
            notional_thresholds: Dict mapping entity names to absolute dollar thresholds
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        validation_passed = True
        pm_df = self._get_pm_mapping()
        
        # Set default exclusions, custom thresholds, and notional thresholds
        if exclude_entities is None:
            exclude_entities = ['sp2-gross']  # Default exclusion
        
        if custom_thresholds is None:
            custom_thresholds = {'sp2-classb-gross': 0.10}  # Default 10% for sp2-classb-gross
            
        if notional_thresholds is None:
            notional_thresholds = {}  # Default empty - no notional thresholds
        
        # Get all aggregated entities (groups and funds)
        groups = pm_df['group'].unique().tolist()
        base_funds = pm_df['fund'].unique().tolist()
        fund_variants = base_funds + [f"{fund}-gross" for fund in base_funds]
        
        aggregated_entities = groups + fund_variants
        
        # Remove excluded entities
        aggregated_entities = [entity for entity in aggregated_entities if entity not in exclude_entities]
        
        # Check each aggregated entity
        for entity in aggregated_entities:
            # Skip if entity doesn't exist in either dataframe
            if entity not in df_before['pm'].values or entity not in df_after['pm'].values:
                continue
                
            # Get the threshold for this entity
            current_threshold = custom_thresholds.get(entity, tolerance_pct)
            notional_threshold = notional_thresholds.get(entity)
                
            for pnl_col in pnl_columns:
                try:
                    original_value = df_before.loc[df_before['pm'] == entity, pnl_col].iloc[0]
                    new_value = df_after.loc[df_after['pm'] == entity, pnl_col].iloc[0]
                    
                    # Determine which validation method to use
                    use_notional = notional_threshold is not None
                    
                    if use_notional:
                        # Use absolute difference validation
                        abs_diff = abs(new_value - original_value)
                        validation_failed = abs_diff > notional_threshold
                        
                        # Always log the difference for transparency
                        log_msg = (f"{entity} {pnl_col}: Original={original_value:.2f}, "
                                  f"New={new_value:.2f}, AbsDiff=${abs_diff:.2f} "
                                  f"({'✓' if not validation_failed else '✗'})")
                        
                        if validation_failed:
                            logging.error(log_msg)
                            
                            # Format for alert notification
                            if send_alert:
                                alert_msg = (
                                    f"⚠️ PnL Validation Failure ⚠️\n\n"
                                    f"Fund/Group: {entity}\n"
                                    f"Period: {pnl_col.replace('_', ' ')}\n"
                                    f"Original Value: ${original_value:.2f}\n"
                                    f"Calculated Value: ${new_value:.2f}\n"
                                    f"Absolute Difference: ${abs_diff:.2f}\n"
                                    f"Notional Threshold: ${notional_threshold:.2f}\n\n"
                                )
                                alert.send_notif(alert_msg)
                            validation_passed = False
                        else:
                            # Use warning level so it shows up in logs
                            logging.warning(log_msg)
                    
                    else:
                        # Use percentage difference validation (original logic)
                        if abs(original_value) > 1e-6:  # Avoid division by zero
                            pct_diff = abs(new_value - original_value) / abs(original_value)
                            
                            # Always log the difference for transparency
                            log_msg = (f"{entity} {pnl_col}: Original={original_value:.2f}, "
                                      f"New={new_value:.2f}, Diff={pct_diff:.2%} "
                                      f"({'✓' if pct_diff <= current_threshold else '✗'})")
                            
                            if pct_diff > current_threshold:
                                logging.error(log_msg)
                                
                                # Format for alert notification
                                if send_alert:
                                    alert_msg = (
                                        f"⚠️ PnL Aggregation Alert ⚠️\n\n"
                                        f"Fund/Group: {entity}\n"
                                        f"Period: {pnl_col.replace('_', ' ')}\n"
                                        f"Original Value: ${original_value:.2f}\n"
                                        f"Calculated Value: ${new_value:.2f}\n"
                                        f"Difference: {pct_diff:.2%}\n"
                                        f"Threshold: {current_threshold:.2%}\n\n"
                                    )
                                    alert.send_notif(alert_msg)
                                validation_passed = False
                            else:
                                # Use warning level so it shows up in logs
                                logging.warning(log_msg)
                                
                        elif abs(new_value) > 1e-6:  # Original was ~0 but new value is significant
                            # For notional thresholds, check if new value exceeds threshold
                            if use_notional:
                                validation_failed = abs(new_value) > notional_threshold
                                log_msg = (f"{entity} {pnl_col}: Original≈0, New={new_value:.2f} "
                                          f"({'✓' if not validation_failed else '✗'})")
                            else:
                                validation_failed = True
                                log_msg = (f"{entity} {pnl_col}: Original≈0, New={new_value:.2f} ✗")
                            
                            if validation_failed:
                                logging.error(log_msg)
                                
                                # Format for alert notification
                                if send_alert:
                                    if use_notional:
                                        alert_msg = (
                                            f"⚠️ PnL Aggregation Alert ⚠️\n\n"
                                            f"Fund/Group: {entity}\n"
                                            f"Period: {pnl_col.replace('_', ' ')}\n"
                                            f"Original Value: ~$0.00\n"
                                            f"Calculated Value: ${new_value:.2f}\n"
                                            f"Absolute Difference: ${abs(new_value):.2f}\n"
                                            f"Notional Threshold: ${notional_threshold:.2f}\n\n"
                                        )
                                    else:
                                        alert_msg = (
                                            f"⚠️ PnL Aggregation Alert ⚠️\n\n"
                                            f"Fund/Group: {entity}\n"
                                            f"Period: {pnl_col.replace('_', ' ')}\n"
                                            f"Original Value: ~$0.00\n"
                                            f"Calculated Value: ${new_value:.2f}\n"
                                            f"Difference: Significant change from zero\n"
                                            f"Threshold: {current_threshold:.2%}\n\n"
                                        )
                                    alert.send_notif(alert_msg)
                                validation_passed = False
                            else:
                                # Use warning level so it shows up in logs  
                                logging.warning(log_msg)
                        
                except (IndexError, KeyError) as e:
                    logging.warning(f"Could not validate {entity} {pnl_col}: {e}")
        
        return validation_passed


    


    def _replace_aggregated_pnl(self, df, pnl_columns='ITD_PnL', 
                            validate: bool = True, tolerance_pct: float = 0.02):
        """
        Replace aggregated PnL values with sum of their components
        
        Args:
            df: DataFrame with columns ['pm', 'ITD_PnL'] or ['pm', pnl_columns]
            pnl_columns: Name of PnL column (str) or list of PnL columns (list)
            validate: Whether to run validation checks
            tolerance_pct: Maximum allowed percentage difference for validation
        
        Returns:
            DataFrame with updated aggregated values
        """
        
        # Convert pnl_columns to list if it's a single string
        if isinstance(pnl_columns, str):
            pnl_columns = [pnl_columns]
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Store original for validation
        if validate:
            original_df = df.copy()
        
        pm_df = self._get_pm_mapping()
        # Create mapping from pm to each hierarchy level
        pm_to_hierarchy = pm_df.set_index('pm').to_dict('index')
        
        # Step 2: Replace group aggregated values with sum of pm_groups
        groups_in_df = result_df[result_df['pm'].isin(pm_df['group'].unique())]['pm'].tolist()
        
        for group in groups_in_df:
            # Find all pm_groups that belong to this group
            pm_groups_in_group = pm_df[pm_df['group'] == group]['pm_group'].unique().tolist()
            # Filter to only pm_groups that exist in our dataframe
            existing_pm_groups = result_df[result_df['pm'].isin(pm_groups_in_group)]['pm'].tolist()
            
            if existing_pm_groups:
                # Sum PnL of all pm_groups in this group for each column
                for pnl_col in pnl_columns:
                    group_pnl_sum = result_df[result_df['pm'].isin(existing_pm_groups)][pnl_col].sum()
                    # Replace the group row's PnL with the sum
                    result_df.loc[result_df['pm'] == group, pnl_col] = group_pnl_sum
        
        # Step 3: Replace fund aggregated values with sum of groups
        # Create list of both fund names and fund-gross names
        base_funds = pm_df['fund'].unique().tolist()
        fund_gross_variants = [f"{fund}-gross" for fund in base_funds]
        funds_in_df = result_df[result_df['pm'].isin(fund_gross_variants)]['pm'].tolist()

        for fund_name in funds_in_df:
            # Remove -gross suffix to get base fund name for hierarchy lookup
            base_fund = fund_name.replace('-gross', '')
            
            # Find all groups that belong to this base fund
            groups_in_fund = pm_df[pm_df['fund'] == base_fund]['group'].unique().tolist()
            # Filter to only groups that exist in our dataframe
            existing_groups = result_df[result_df['pm'].isin(groups_in_fund)]['pm'].tolist()
            
            if existing_groups:
                # Sum PnL of all groups in this fund for each column
                for pnl_col in pnl_columns:
                    fund_pnl_sum = result_df[result_df['pm'].isin(existing_groups)][pnl_col].sum()
                    # Replace the fund row's PnL with the sum
                    result_df.loc[result_df['pm'] == fund_name, pnl_col] = fund_pnl_sum
        
        # Run validation if requested
        if validate:
            validation_passed = self._validate_pnl_aggregation(
                original_df, result_df, pnl_columns, tolerance_pct,
                custom_thresholds={'sp1-gross': 0.60, 'sp3-gross':0.2},  # Can be overridden
                notional_thresholds={'sp2-classb-gross': 2, 'sp2-smaclassb': 0.02},  # Can be overridden
                exclude_entities=['sp2-gross', 'sp2-classa-gross', 'sp2-classb-manual']  # Can be overridden
            )
            if not validation_passed:
                logging.warning("PnL aggregation validation failed - check logs for details")
        
        return result_df
        
    def _calculate_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        print('calculate pnl')
        print(df)
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
        
        df = self._replace_aggregated_pnl(df, ['ITD_PnL', 'MTD_PnL', 'QTD_PnL', 'YTD_PnL'])
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
        # sheet_utils.set_dataframe(df=bal_merged_df, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='transfer_merged')


        bal_merged_df = self._calculate_pnl(bal_merged_df)
        
        # print('caculate pnl \n')
        # print(bal_merged_df)
        # sheet_utils.set_dataframe(df=bal_merged_df, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='pnl')
        # Process NAV data
        nav_merged_df = self._calculate_returns(nav_merged_df)
        
        # Get exposure data
        exposure_df = get_exposure.get_exposure_df()
        
        # Merge all data
        merged_df = pd.merge(bal_merged_df, nav_merged_df, on='pm', how='left')
        merged_df = pd.merge(merged_df, exposure_df, on='pm', how='left')
        
        # Apply adjustments and calculations
        merged_df = self._apply_exposure_adjustments(merged_df)
        merged_df = self._calculate_exposure_ratios(merged_df)
        merged_df = self._prepare_final_dataframe(merged_df)
        
        # Add metrics
        metrics_df = metrics_utils.get_metrics()
        merged_df = pd.merge(merged_df, metrics_df, on='pm', how='left')
        
        print('Performance metrics table:')
        print(merged_df)
        # sheet_utils.set_dataframe(df=merged_df, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='perf_metrics')
        
        # Export dashboard and update database
        usdtusd_price = self._get_usdtusd_price()
        export_dashboard.export_dashboard(merged_df.copy(), usdtusd_price)
        db_utils.df_replace_table(table_name='performance_metrics', df=merged_df)
        print('Updated performance_metrics table')
        
        # Process and update QTD metrics
        qtd_metrics = self._process_qtd_metrics(merged_df, end_time)
        # sheet_utils.set_dataframe(df=qtd_metrics, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='qtd_test')
        
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