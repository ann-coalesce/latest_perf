import db_utils
import credentials
import pandas as pd
import math
from datetime import datetime, timezone
import numpy as np
from pandas.tseries.offsets import QuarterBegin

def get_pm_grouped_bal(ending_time=None):
    """Fetches hourly-aligned NAV and balance data for each PM, optionally up to a specific ending_time."""
    curr_time = ending_time or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    
    query = f'''
        SELECT 
            timestamp,
            pm,
            nav, 
            balance
        FROM 
            nav_table 
        WHERE 
            (
                EXTRACT(HOUR FROM timestamp) = 0
                AND EXTRACT(MINUTE FROM timestamp) = 0
                AND EXTRACT(SECOND FROM timestamp) = 0
                AND timestamp BETWEEN '2024-01-01' AND '{curr_time}'
            )
            OR timestamp IN (
                SELECT timestamp - interval '1 minute' FROM max_timestamp
            )
    ''' if ending_time is None else f'''
        SELECT 
            timestamp,
            pm,
            nav, 
            balance
        FROM 
            nav_table 
        WHERE 
            EXTRACT(HOUR FROM timestamp) = 0
            AND EXTRACT(MINUTE FROM timestamp) = 0
            AND EXTRACT(SECOND FROM timestamp) = 0
            AND timestamp BETWEEN '2024-01-01' AND '{ending_time}'
    '''
    
    balance = db_utils.get_db_table(query=query)
    grouping_df = pd.DataFrame(credentials.PM_DATA)

    # Join to enrich with PM group info if needed
    if 'pm_group' in grouping_df.columns:
        balance = balance[balance['pm'].isin(grouping_df['pm_group'])]

    return balance

def calculate_metric(df_pm):
    """Calculates the Sharpe ratio for a given PM's NAV return series."""
    avg_return = df_pm['daily_return'].mean()
    std_return = df_pm['daily_return'].std()

    if std_return == 0 or pd.isna(std_return):
        sharpe = float('nan')
    else:
        sharpe = (avg_return * 365 - credentials.RISK_FREE_RATE) / (std_return * math.sqrt(365))

    return pd.Series({'sharpe': sharpe})

def get_qtd_sharpe(df):
    """Computes QTD Sharpe ratio per PM."""
    df = df.sort_values(by='timestamp')
    df['daily_return'] = df.groupby('pm')['nav'].pct_change(fill_method='ffill')

    # Filter for QTD only
    current_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    quarter_start = current_time - QuarterBegin(startingMonth=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    filtered_df = df[df['timestamp'] >= quarter_start]

    # Calculate sharpe ratio by PM
    result_df = filtered_df.groupby('pm').apply(calculate_metric, include_groups=False).reset_index()

    return result_df

def get_qtd_sharpe_df(ending_time=None):
    """Wrapper to fetch QTD Sharpe ratios for all PMs."""
    ending_time = ending_time or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    balance = get_pm_grouped_bal(ending_time=ending_time)
    result_df = get_qtd_sharpe(df=balance)
    result_df.fillna(0, inplace=True)
    # print(result_df)
    return result_df

# if __name__ == '__main__':
#     get_qtd_sharpe_df()