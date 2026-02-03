import db_utils
import sheet_utils
import pandas as pd
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import credentials

def calculate_max_drawdown(nav_series):
    rolling_max = nav_series.cummax()
    drawdowns = nav_series / rolling_max - 1
    return drawdowns.min()


def calculate_metric_for_pm(df_pm, risk_free_rate=0.0):
    df_pm = df_pm.copy()
    
    # when victor charlie starts using matrixport loan (2x leverage), then the below adjustment is needed
    # df_pm.loc[df_pm['pm'] == 'sp2-sma-victorcharliebtc', 'daily_return'] *= 2

    pm_list = ['sp2-sma-romeocharliebtc', 'farbromeocharliebybit_btc', 'farbromeocharliegate_btc']
    if df_pm['pm'].iloc[0] in pm_list:
        df_pm = df_pm[df_pm['timestamp'] > pd.to_datetime('2025-10-05 00:00:00').tz_localize('UTC')]
        print(df_pm.head(5))

    # Compute basic statistics
    avg_daily_return = df_pm['daily_return'].mean()
    std_daily_return = df_pm['daily_return'].std()
    std_negative_return = df_pm['negative_return'].std()

    annualized_vol = std_daily_return * math.sqrt(365)
    annualized_downside_vol = std_negative_return * math.sqrt(365)
    
    # Add safety checks for division by zero
    if std_daily_return == 0 or np.isnan(std_daily_return):
        sharpe = np.nan
    else:
        sharpe = (avg_daily_return * 365 - risk_free_rate) / (std_daily_return * math.sqrt(365))
    
    if std_negative_return == 0 or np.isnan(std_negative_return):
        sortino = np.nan
    else:
        sortino = (avg_daily_return * 365 - risk_free_rate) / (std_negative_return * math.sqrt(365))

    # Get start and end NAV values
    start_nav = df_pm['nav'].iloc[0]
    end_nav = df_pm['nav'].iloc[-1]

    # Calculate the period return
    period_return = (end_nav / start_nav) - 1

    # Calculate accumulated returns
    df_pm['accumulated_return'] = df_pm['nav'] - 1

    # Calculate rolling max of accumulated returns (365 periods)
    df_pm['rolling_max'] = df_pm['accumulated_return'].rolling(window=365, min_periods=1).max()

    # Calculate drawdowns
    df_pm['drawdown'] = (1 + df_pm['accumulated_return']) / (1 + df_pm['rolling_max']) - 1

    # Calculate max drawdown
    max_drawdown = df_pm['drawdown'].min()

    # Current drawdown
    curr_drawdown = df_pm['drawdown'].iloc[-1]

    # Number of days in the period
    num_days = df_pm.shape[0]

    # Calculate Calmar ratio
    calmar_ratio = ((1 + period_return) ** (365.0 / num_days) - 1) / -max_drawdown

    # Annualized return
    annualized_return_all = end_nav ** (365 / num_days) - 1

    # ====== VaR and CVaR Metrics ====== #
    var_1d_99 = np.percentile(df_pm['daily_return'][1:], 1)
    var_10d_99 = var_1d_99 * np.sqrt(10)
    cvar_1d_99 = df_pm['daily_return'][1:][df_pm['daily_return'] <= var_1d_99].mean()

    # ====== Rolling Metrics (30-day, 60-day, 90-day, YTD) ====== #
    # Step 1: 30, 60, 90 day rolling metrics
    for window in [30, 60, 90, 365]:
        window_label = f'{window}d'
        if len(df_pm) >= window:
            sub_df = df_pm.iloc[-window:]
            avg_return = sub_df['daily_return'].mean()
            std_return = sub_df['daily_return'].std()
            std_neg_return = sub_df['negative_return'].std()

            period_return = (sub_df['nav'].iloc[-1] / sub_df['nav'].iloc[0]) - 1
            annualized_return = (1 + period_return) ** (365.0 / window) - 1
            max_dd = calculate_max_drawdown(sub_df['nav'])

            df_pm[f'rolling_sharpe_{window_label}'] = (avg_return * 365 - risk_free_rate) / (std_return * math.sqrt(365)) if std_return else np.nan
            df_pm[f'rolling_sortino_{window_label}'] = (avg_return * 365 - risk_free_rate) / (std_neg_return * math.sqrt(365)) if std_neg_return else np.nan
            df_pm[f'rolling_annualized_vol_{window_label}'] = std_return * math.sqrt(365)
            df_pm[f'rolling_annualized_downside_vol_{window_label}'] = std_neg_return * math.sqrt(365)
            df_pm[f'rolling_period_return_{window_label}'] = period_return
            df_pm[f'rolling_annualized_return_{window_label}'] = annualized_return
            df_pm[f'rolling_max_dd_{window_label}'] = [None] * (len(df_pm) - 1) + [max_dd]
            df_pm[f'rolling_calmar_{window_label}'] = annualized_return / -max_dd if max_dd != 0 else np.nan
            df_pm[f'rolling_avg_return_{window_label}'] = avg_return
            df_pm[f'rolling_std_return_{window_label}'] = std_return
            df_pm[f'rolling_std_neg_return_{window_label}'] = std_neg_return
        else:
            # Use period values as fallback
            df_pm[f'rolling_sharpe_{window_label}'] = sharpe
            df_pm[f'rolling_sortino_{window_label}'] = sortino
            df_pm[f'rolling_annualized_vol_{window_label}'] = annualized_vol
            df_pm[f'rolling_annualized_downside_vol_{window_label}'] = annualized_downside_vol
            df_pm[f'rolling_max_dd_{window_label}'] = max_drawdown
            df_pm[f'rolling_calmar_{window_label}'] = calmar_ratio
            df_pm[f'rolling_period_return_{window_label}'] = period_return
            df_pm[f'rolling_annualized_return_{window_label}'] = annualized_return_all
            df_pm[f'rolling_avg_return_{window_label}'] = avg_daily_return
            df_pm[f'rolling_std_return_{window_label}'] = std_daily_return
            df_pm[f'rolling_std_neg_return_{window_label}'] = std_negative_return
        

    # Step 2: Process YTD separately
    if 'timestamp' in df_pm.columns:
        current_year = pd.Timestamp.now().year
        current_year_mask = pd.to_datetime(df_pm['timestamp']).dt.year == current_year
        ytd_data = df_pm[current_year_mask].copy()

        if not ytd_data.empty:
            # 移除 1/1 資料
            jan_1_mask = pd.to_datetime(ytd_data['timestamp']).dt.strftime('%m-%d') == '01-01'
            ytd_data = ytd_data[~jan_1_mask]

            if not ytd_data.empty:
                ytd_label = 'ytd'
                avg_return = ytd_data['daily_return'].mean()
                std_return = ytd_data['daily_return'].std()
                std_neg_return = ytd_data['negative_return'].std()
                period_return = (ytd_data['nav'].iloc[-1] / ytd_data['nav'].iloc[0]) - 1
                num_days = len(ytd_data)
                annualized_return = (1 + period_return) ** (365.0 / num_days) - 1
                max_dd = calculate_max_drawdown(ytd_data['nav'])

                df_pm[f'rolling_sharpe_ytd'] = (avg_return * 365 - risk_free_rate) / (std_return * math.sqrt(365)) if std_return else np.nan
                df_pm[f'rolling_sortino_ytd'] = (avg_return * 365 - risk_free_rate) / (std_neg_return * math.sqrt(365)) if std_neg_return else np.nan
                df_pm[f'rolling_annualized_vol_ytd'] = std_return * math.sqrt(365)
                df_pm[f'rolling_annualized_downside_vol_ytd'] = std_neg_return * math.sqrt(365)
                df_pm[f'rolling_period_return_ytd'] = period_return
                df_pm[f'rolling_annualized_return_ytd'] = annualized_return
                df_pm[f'rolling_max_dd_ytd'] = [None] * (len(df_pm) - 1) + [max_dd]
                df_pm[f'rolling_calmar_ytd'] = annualized_return / -max_dd if max_dd != 0 else np.nan
                df_pm['rolling_avg_return_ytd'] = avg_return
                df_pm['rolling_std_return_ytd'] = std_return
                df_pm['rolling_std_neg_return_ytd'] = std_neg_return
            else:
                df_pm['rolling_sharpe_ytd'] = np.nan
                df_pm['rolling_sortino_ytd'] = np.nan
                df_pm['rolling_annualized_vol_ytd'] = np.nan
                df_pm['rolling_annualized_downside_vol_ytd'] = np.nan
                df_pm['rolling_max_dd_ytd'] = np.nan
                df_pm['rolling_calmar_ytd'] = np.nan
                df_pm['rolling_period_return_ytd'] = np.nan
                df_pm['rolling_annualized_return_ytd'] = np.nan
                df_pm['rolling_avg_return_ytd'] = np.nan
                df_pm['rolling_std_return_ytd'] = np.nan
                df_pm['rolling_std_neg_return_ytd'] = np.nan

    # Helper function to safely get the last value from a column
    def safe_get_last_value(column_name):
        if column_name in df_pm.columns:
            col_data = df_pm[column_name]
            if isinstance(col_data, pd.Series):
                return col_data.iloc[-1]
            else:
                return col_data
        return np.nan

    # Get latest available values for rolling metrics
    latest_rolling_metrics = {
        f'sharpe_rolling_30d': safe_get_last_value('rolling_sharpe_30d'),
        f'sharpe_rolling_60d': safe_get_last_value('rolling_sharpe_60d'),
        f'sharpe_rolling_90d': safe_get_last_value('rolling_sharpe_90d'),
        f'sharpe_rolling_365d': safe_get_last_value('rolling_sharpe_365d'),
        f'sharpe_rolling_ytd': safe_get_last_value('rolling_sharpe_ytd'),
        f'sortino_rolling_30d': safe_get_last_value('rolling_sortino_30d'),
        f'sortino_rolling_60d': safe_get_last_value('rolling_sortino_60d'),
        f'sortino_rolling_90d': safe_get_last_value('rolling_sortino_90d'),
        f'sortino_rolling_365d': safe_get_last_value('rolling_sortino_365d'),
        f'sortino_rolling_ytd': safe_get_last_value('rolling_sortino_ytd'),
        f'annualized_vol_rolling_30d': safe_get_last_value('rolling_annualized_vol_30d'),
        f'annualized_vol_rolling_60d': safe_get_last_value('rolling_annualized_vol_60d'),
        f'annualized_vol_rolling_90d': safe_get_last_value('rolling_annualized_vol_90d'),
        f'annualized_vol_rolling_365d': safe_get_last_value('rolling_annualized_vol_365d'),
        f'annualized_vol_rolling_ytd': safe_get_last_value('rolling_annualized_vol_ytd'),
        f'annualized_downside_vol_rolling_30d': safe_get_last_value('rolling_annualized_downside_vol_30d'),
        f'annualized_downside_vol_rolling_60d': safe_get_last_value('rolling_annualized_downside_vol_60d'),
        f'annualized_downside_vol_rolling_90d': safe_get_last_value('rolling_annualized_downside_vol_90d'),
        f'annualized_downside_vol_rolling_365d': safe_get_last_value('rolling_annualized_downside_vol_365d'),
        f'annualized_downside_vol_rolling_ytd': safe_get_last_value('rolling_annualized_downside_vol_ytd'),
        f'max_dd_rolling_30d': safe_get_last_value('rolling_max_dd_30d'),
        f'max_dd_rolling_60d': safe_get_last_value('rolling_max_dd_60d'),
        f'max_dd_rolling_90d': safe_get_last_value('rolling_max_dd_90d'),
        f'max_dd_rolling_365d': safe_get_last_value('rolling_max_dd_365d'),
        f'max_dd_rolling_ytd': safe_get_last_value('rolling_max_dd_ytd'),
        f'calmar_rolling_30d': safe_get_last_value('rolling_calmar_30d'),
        f'calmar_rolling_60d': safe_get_last_value('rolling_calmar_60d'),
        f'calmar_rolling_90d': safe_get_last_value('rolling_calmar_90d'),
        f'calmar_rolling_365d': safe_get_last_value('rolling_calmar_365d'),
        f'calmar_rolling_ytd': safe_get_last_value('rolling_calmar_ytd'),
        f'rolling_period_return_30d': safe_get_last_value('rolling_period_return_30d'),
        f'rolling_period_return_60d': safe_get_last_value('rolling_period_return_60d'),
        f'rolling_period_return_90d': safe_get_last_value('rolling_period_return_90d'),
        f'rolling_period_return_365d': safe_get_last_value('rolling_period_return_365d'),
        f'rolling_period_return_ytd': safe_get_last_value('rolling_period_return_ytd'),
        f'annualized_return_rolling_30d': safe_get_last_value('rolling_annualized_return_30d'),
        f'annualized_return_rolling_60d': safe_get_last_value('rolling_annualized_return_60d'),
        f'annualized_return_rolling_90d': safe_get_last_value('rolling_annualized_return_90d'),
        f'annualized_return_rolling_365d': safe_get_last_value('rolling_annualized_return_365d'),
        f'annualized_return_rolling_ytd': safe_get_last_value('rolling_annualized_return_ytd'),
    }

    # Return all metrics
    return pd.Series({
        'annualized_return': annualized_return_all,
        'sharpe': sharpe,
        'sortino': sortino,
        'annualized_vol': annualized_vol,
        'annualized_downside_vol': annualized_downside_vol,
        'std_daily_return': std_daily_return,
        'std_negative_return': std_negative_return,
        'curr_dd': curr_drawdown,
        'max_dd': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'number_of_days': num_days,
        'var_1d_99': var_1d_99,
        'var_10d_99': var_10d_99,
        'cvar_1d_99': cvar_1d_99,
        **latest_rolling_metrics
    })


def get_metrics():
    curr = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
    query = f'''
        SELECT 
            timestamp,
            pm,
            nav
        FROM 
            nav_table 
        WHERE 
            (
                EXTRACT(HOUR FROM timestamp) = 0
                AND EXTRACT(MINUTE FROM timestamp) = 0
                AND EXTRACT(SECOND FROM timestamp) = 0
                AND timestamp BETWEEN '2024-01-01' AND '{curr}'
            )
            OR timestamp in (select timestamp - interval '2 minute' from max_timestamp)
    '''

    df = db_utils.get_db_table(query=query)
    unique_pms = df['pm'].unique()
    new_rows = pd.DataFrame({
        'timestamp': ['2000-01-01 00:00:00+00:00'] * len(unique_pms),
        'pm': unique_pms,
        'nav': [1] * len(unique_pms)
    })

    # Append the new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    df['daily_return'] = df.groupby('pm')['nav'].pct_change(fill_method='ffill')
    df['negative_return'] = np.where(df['daily_return'] < 0, df['daily_return'], 0)
 
    # Apply the calculation for each PM and collect the results in a new DataFrame
    result_df = df.groupby('pm').apply(calculate_metric_for_pm)

    # Replace infinities with strings
    result_df.replace({np.inf: 'inf', -np.inf: '-inf'}, inplace=True)
    result_df.fillna(0, inplace=True) 
    
    return result_df

if __name__ == "__main__":
    result_df = get_metrics()
    result_df.reset_index(inplace=True)
    sheet_url = 'https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=0#gid=0'
    sheet_utils.set_dataframe(result_df, url=sheet_url, sheet_name='Sheet1', row=1, include_header=True) 