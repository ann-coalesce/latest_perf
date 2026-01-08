# import db_utils
# import sheet_utils
# import pandas as pd
# from datetime import datetime, timezone, timedelta
# import numpy as np


# def get_nav(end_time=None):
#   if end_time == None:
#     curr = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
#   else:
#     curr = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
  
#   curr_hour = curr.replace(minute=0,second=0, microsecond=0)
#   query = f'''WITH latest_nav AS (
#         SELECT 
#             pm, 
#             timestamp, 
#             nav AS nav_value, 
#             'nav' AS nav_type,
#             ROW_NUMBER() OVER (PARTITION BY pm ORDER BY timestamp DESC) AS rn
#         FROM nav_table
#         WHERE timestamp = '{curr}' OR timestamp = '{curr_hour}'
#     ),
#     month_start_nav AS (
#         SELECT 
#             pm, 
#             timestamp, 
#             nav AS nav_value, 
#             'nav_month_start' AS nav_type
#         FROM nav_table
#         WHERE pm NOT IN ('deribit_master') 
#           AND timestamp = DATE_TRUNC('month', NOW())
#     ),
#     quarter_start_nav AS (
#         SELECT 
#             pm, 
#             timestamp, 
#             nav AS nav_value, 
#             'nav_quarter_start' AS nav_type
#         FROM nav_table
#         WHERE pm NOT IN ('deribit_master') 
#           AND timestamp = DATE_TRUNC('quarter', NOW())
#     ),
#     year_start_nav AS (
#         SELECT 
#             pm, 
#             timestamp, 
#             nav AS nav_value, 
#             'nav_year_start' AS nav_type
#         FROM nav_table
#         WHERE pm NOT IN ('deribit_master') 
#           AND timestamp = DATE_TRUNC('year', NOW())
#     )
#     SELECT pm, timestamp, nav_value AS nav, nav_type
#     FROM (
#         SELECT pm, timestamp, nav_value, nav_type FROM latest_nav WHERE rn = 1
#         UNION ALL
#         SELECT pm, timestamp, nav_value, nav_type FROM month_start_nav
#         UNION ALL
#         SELECT pm, timestamp, nav_value, nav_type FROM quarter_start_nav
#         UNION ALL
#         SELECT pm, timestamp, nav_value, nav_type FROM year_start_nav
#     ) AS combined_data;'''
  
#   nav = db_utils.get_db_table(query=query)
#   # print(nav)
#   # sheet_utils.set_dataframe

#   # Pivot the data to bring all balance types into separate columns
#   try:
#     if not nav.empty:
#       nav_pivot = nav.pivot(index='pm', columns='nav_type', values='nav').reset_index()
      
#       nav_pivot = nav_pivot.dropna(subset=['nav'])

#       # Fill NA values in last 3 columns using 'nav' column
#       nav_pivot[['nav_month_start', 'nav_quarter_start', 'nav_year_start']] = \
#       nav_pivot[['nav_month_start', 'nav_quarter_start', 'nav_year_start']].apply(lambda x: x.fillna(1))
#       nav_pivot.replace([np.inf, -np.inf], 0, inplace=True)

#       nav_pivot.dropna(inplace=True)
#       print(nav_pivot)
#   except Exception as e:
#     nav_pivot = pd.DataFrame()
#     print(e)

#   return nav_pivot



# # end_time like '2025-01-07 03:30:00'
# def get_bal_from_nav_old(end_time=None):
#   if end_time == None:
#     curr = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
#   else:
#     curr = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
  
#   curr_hour = curr.replace(minute=0,second=0, microsecond=0)

#   query = f'''WITH latest_balance AS (
#       SELECT 
#           pm, 
#       timestamp,
#           balance, 
#           'balance' AS balance_type,
#           ROW_NUMBER() OVER (PARTITION BY pm ORDER BY timestamp DESC) AS rn
#       FROM nav_table
#       WHERE timestamp = '{curr}' OR timestamp = '{curr_hour}'
#   ),
#   month_start_balance AS (
#       SELECT 
#           pm, 
#       timestamp,
#           balance, 
#           'balance_month_start' AS balance_type
#       FROM nav_table
#       WHERE pm NOT IN ('deribit_master') 
#         AND timestamp = DATE_TRUNC('month', NOW())
#   ),
#   quarter_start_balance AS (
#       SELECT 
#           pm, 
#       timestamp,
#           balance, 
#           'balance_quarter_start' AS balance_type
#       FROM nav_table
#       WHERE pm NOT IN ('deribit_master') 
#         AND timestamp = DATE_TRUNC('quarter', NOW())
#   ),
#   year_start_balance AS (
#       SELECT 
#           pm, 
#       timestamp,
#           balance, 
#           'balance_year_start' AS balance_type
#       FROM nav_table
#       WHERE pm NOT IN ('deribit_master') 
#         AND timestamp = DATE_TRUNC('year', NOW())
#   )
#   SELECT pm, timestamp, balance, balance_type
#   FROM (
#       SELECT pm, timestamp, balance, balance_type FROM latest_balance WHERE rn = 1
#       UNION ALL
#       SELECT pm, timestamp, balance, balance_type FROM month_start_balance
#       UNION ALL
#       SELECT pm, timestamp, balance, balance_type FROM quarter_start_balance
#       UNION ALL
#       SELECT pm, timestamp, balance, balance_type FROM year_start_balance
#   ) AS combined_data;'''

#   balances = db_utils.get_db_table(query=query)
#   # print(balances)

#   # Pivot the data to bring all balance types into separate columns
#   balance_pivot = pd.DataFrame()
  
#   try:
#     if not balances.empty:
#       balance_pivot = balances.pivot(index='pm', columns='balance_type', values='balance').reset_index()
#       balance_pivot = balance_pivot.dropna(subset=['balance'])

#       # Fill NA values in last 3 columns using 'nav' column
#       balance_pivot[['balance_month_start', 'balance_quarter_start', 'balance_year_start']] = \
#       balance_pivot[['balance_month_start', 'balance_quarter_start', 'balance_year_start']].apply(lambda x: x.fillna(0))

#       # balance_pivot.dropna(inplace=True)
#       # print(balance_pivot)
#   except Exception as e:
#     print(e)
#   return balance_pivot


# nav = get_nav()
# sheet_utils.set_dataframe(df=nav, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='old_nav')
# # get_bal_from_nav()



import db_utils
import sheet_utils
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
from typing import Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_NAV = 1.0
DEFAULT_BALANCE = 0.0
EXCLUDED_PMS = []


def _calculate_period_starts(curr: datetime) -> tuple[datetime, datetime, datetime]:
    """Calculate month, quarter, and year start timestamps."""
    month_start = curr.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate quarter start
    quarter_month = ((curr.month - 1) // 3) * 3 + 1
    quarter_start = curr.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    year_start = curr.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    return month_start, quarter_start, year_start


def _get_time_series_data(
        column_name: str, 
        curr: datetime, 
        curr_hour: datetime, 
        excluded_pms: Optional[list] = None
    ) -> pd.DataFrame:
    """
    Generic function to get time series data (nav or balance) for current time and period starts.
    
    Args:
        column_name: Name of the column to retrieve ('nav' or 'balance')
        curr: Current timestamp
        curr_hour: Current hour timestamp
        excluded_pms: List of portfolio managers to exclude from period calculations
        
    Returns:
        DataFrame with time series data
    """
    if excluded_pms is None:
        excluded_pms = EXCLUDED_PMS.copy()
    
    month_start, quarter_start, year_start = _calculate_period_starts(curr)
    
    # Build exclusion clause
    exclusion_clause = ""
    if excluded_pms:
        excluded_list = "', '".join(excluded_pms)
        exclusion_clause = f"AND pm NOT IN ('{excluded_list}')"
    
    query = f'''WITH latest_data AS (
        SELECT 
            pm, 
            timestamp, 
            {column_name} AS value, 
            '{column_name}' AS data_type,
            ROW_NUMBER() OVER (PARTITION BY pm ORDER BY timestamp DESC) AS rn
        FROM nav_table
        WHERE timestamp = '{curr}' OR timestamp = '{curr_hour}'
    ),
    month_start_data AS (
        SELECT 
            pm, 
            timestamp, 
            {column_name} AS value, 
            '{column_name}_month_start' AS data_type
        FROM nav_table
        WHERE timestamp = '{month_start}' {exclusion_clause}
    ),
    quarter_start_data AS (
        SELECT 
            pm, 
            timestamp, 
            {column_name} AS value, 
            '{column_name}_quarter_start' AS data_type
        FROM nav_table
        WHERE timestamp = '{quarter_start}' {exclusion_clause}
    ),
    year_start_data AS (
        SELECT 
            pm, 
            timestamp, 
            {column_name} AS value, 
            '{column_name}_year_start' AS data_type
        FROM nav_table
        WHERE timestamp = '{year_start}' {exclusion_clause}
    )
    SELECT pm, timestamp, value, data_type
    FROM (
        SELECT pm, timestamp, value, data_type FROM latest_data WHERE rn = 1
        UNION ALL
        SELECT pm, timestamp, value, data_type FROM month_start_data
        UNION ALL
        SELECT pm, timestamp, value, data_type FROM quarter_start_data
        UNION ALL
        SELECT pm, timestamp, value, data_type FROM year_start_data
    ) AS combined_data;'''
    print('time series query')
    print(query)
    try:
        return db_utils.get_db_table(query=query)
    except Exception as e:
        logger.error(f"Failed to execute query for {column_name}: {e}")
        return pd.DataFrame()


def _pivot_and_clean_data_test(
    data: pd.DataFrame, 
    value_column: str, 
    type_column: str, 
    main_column: str,
    default_fill_value: float
) -> pd.DataFrame:
    """
    Pivot and clean the time series data.
    
    Args:
        data: Raw data from database
        value_column: Name of the value column
        type_column: Name of the type column for pivoting
        main_column: Name of the main column (e.g., 'nav', 'balance')
        default_fill_value: Default value to fill NAs in period columns
        
    Returns:
        Cleaned and pivoted DataFrame
    """
    if data.empty:
        logger.warning("No data to pivot")
        return pd.DataFrame()
    
    try:
        # Pivot the data
        pivot_data = data.pivot(index='pm', columns=type_column, values=value_column).reset_index()
        
        # Remove rows where main column is null
        # pivot_data = pivot_data.dropna(subset=[main_column])
        pivot_data.fillna(0, inplace=True)
        
        # Get period columns (all columns except 'pm' and main column)
        period_columns = [col for col in pivot_data.columns 
                         if col not in ['pm', main_column]]
        
        # Fill NA values in period columns
        if period_columns:
            pivot_data[period_columns] = pivot_data[period_columns].fillna(default_fill_value)
        
        # Replace infinite values with 0
        pivot_data.replace([np.inf, -np.inf], 0, inplace=True)
        
        return pivot_data
        
    except Exception as e:
        logger.error(f"Failed to pivot and clean data: {e}")
        return pd.DataFrame()


def _parse_end_time(end_time: Optional[str]) -> datetime:
    """
    Parse end_time string or return current time.
    
    Args:
        end_time: Optional timestamp in 'YYYY-MM-DD HH:MM:SS' format
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If end_time format is invalid
    """
    if end_time is None:
        return datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
    
    try:
        return datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid end_time format. Expected 'YYYY-MM-DD HH:MM:SS': {e}")


def get_nav(end_time: Optional[str] = None) -> pd.DataFrame:
    """
    Get NAV data for portfolio managers at current/specified time and period starts.
    
    Args:
        end_time: Optional timestamp in 'YYYY-MM-DD HH:MM:SS' format.
                 If None, uses current time minus 1 minute.
        
    Returns:
        DataFrame with columns: pm, nav, nav_month_start, nav_quarter_start, nav_year_start
        Returns empty DataFrame if no data found or on error.
        
    Raises:
        ValueError: If end_time format is invalid
    """
    try:
        curr = _parse_end_time(end_time)
        curr_hour = curr.replace(minute=0, second=0, microsecond=0)
        
        # Get raw nav data
        nav_data = _get_time_series_data('nav', curr, curr_hour)
        
        # Pivot and clean the data
        nav_pivot = _pivot_and_clean_data(
            data=nav_data,
            value_column='value',
            type_column='data_type',
            main_column='nav',
            default_fill_value=DEFAULT_NAV
        )
        
        return nav_pivot
        
    except Exception as e:
        logger.error(f"Failed to get NAV data: {e}")
        return pd.DataFrame()


def get_bal_from_nav(end_time: Optional[str] = None) -> pd.DataFrame:
    """
    Get balance data for portfolio managers at current/specified time and period starts.
    
    Args:
        end_time: Optional timestamp in 'YYYY-MM-DD HH:MM:SS' format.
                 If None, uses current time minus 1 minute.
        
    Returns:
        DataFrame with columns: pm, balance, balance_month_start, balance_quarter_start, balance_year_start
        Returns empty DataFrame if no data found or on error.
        
    Raises:
        ValueError: If end_time format is invalid
    """
    try:
        curr = _parse_end_time(end_time) + timedelta(hours=1)
        curr_hour = curr.replace(minute=0, second=0, microsecond=0)
        
        # Get raw balance data
        balance_data = _get_time_series_data('balance', curr, curr_hour)
        sheet_utils.set_dataframe(df=balance_data, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='bal_time_series')
        
        # Pivot and clean the data
        balance_pivot = _pivot_and_clean_data_test(
            data=balance_data,
            value_column='value',
            type_column='data_type',
            main_column='balance',
            default_fill_value=DEFAULT_BALANCE
        )
        
        return balance_pivot
        
    except Exception as e:
        logger.error(f"Failed to get balance data: {e}")
        return pd.DataFrame()


# Example usage:
if __name__ == "__main__":
    # Get current NAV data
    df = get_bal_from_nav()
    # df = get_bal_from_nav_old()
    print("Bal Data:")
    print(df)
    sheet_utils.set_dataframe(df=df, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=1948587986#gid=1948587986', sheet_name='bal_pivot')
    
    # # Get NAV data for specific time
    # nav_df_specific = get_nav('2025-01-07 03:30:00')
    # print("\nNAV Data for specific time:")
    # print(nav_df_specific)
    
    # # Get current balance data
    # balance_df = get_bal_from_nav()
    # print("\nBalance Data:")
    # print(balance_df)