import numpy as np
import pandas as pd
import sheet_utils

def export_dashboard(df, usdtusd_price):
    df['vol_aum'] = df['annualized_vol'] * df['balance']
    df['vol_aum_rolling_30d'] = df['annualized_vol_rolling_30d'] * df['balance']
    df['vol_aum_rolling_60d'] = df['annualized_vol_rolling_60d'] * df['balance']
    df['vol_aum_rolling_90d'] = df['annualized_vol_rolling_90d'] * df['balance']
    df['vol_aum_rolling_ytd'] = df['annualized_vol_rolling_ytd'] * df['balance']
    # df = df[['pm', 'timestamp','balance', 'itd_pnl', 'itd_pnl_percentage', 'mtd_pnl', 'mtd_pnl_percentage', 'qtd_pnl', 'qtd_pnl_percentage', 'ytd_pnl', 'ytd_pnl_percentage', 'annualized_return', 'sharpe', 'sortino', 'calmar_ratio', 'max_dd', 'curr_dd', 'annualized_vol', 'vol_aum', 'net_exposure_ratio', 'gross_exposure_ratio']]

    df = df[['pm', 'timestamp', 'balance', 'itd_pnl', 'itd_pnl_percentage', 'mtd_pnl', 'mtd_pnl_percentage', 'qtd_pnl', 'qtd_pnl_percentage', 'ytd_pnl', 'ytd_pnl_percentage', 'annualized_return', 'sharpe', 'sortino', 'calmar_ratio', 'annualized_vol', 'annualized_downside_vol', 'vol_aum', 'max_dd', 'curr_dd', 'net_exposure_ratio', 'gross_exposure_ratio', 

        # 30-day metrics
        'annualized_return_rolling_30d', 'sharpe_rolling_30d', 'sortino_rolling_30d', 'calmar_rolling_30d', 'annualized_vol_rolling_30d', 'annualized_downside_vol_rolling_30d',  'vol_aum_rolling_30d', 'max_dd_rolling_30d',

        # 60-day metrics
        'annualized_return_rolling_60d', 'sharpe_rolling_60d', 'sortino_rolling_60d', 'calmar_rolling_60d', 'annualized_vol_rolling_60d', 'annualized_downside_vol_rolling_60d', 'vol_aum_rolling_60d', 'max_dd_rolling_60d',

        # 90-day metrics
        'annualized_return_rolling_90d', 'sharpe_rolling_90d', 'sortino_rolling_90d', 'calmar_rolling_90d', 'annualized_vol_rolling_90d', 'annualized_downside_vol_rolling_90d', 'vol_aum_rolling_90d', 'max_dd_rolling_90d',

        # ytd metrics
        'annualized_return_rolling_ytd', 'sharpe_rolling_ytd', 'sortino_rolling_ytd', 'calmar_rolling_ytd', 'annualized_vol_rolling_ytd', 'annualized_downside_vol_rolling_ytd', 'vol_aum_rolling_ytd', 'max_dd_rolling_ytd',

        'var_1d_99', 'var_10d_99', 'cvar_1d_99'
    ]]
    # df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.replace([np.inf, -np.inf], 0)
    df_usd = df.copy()
    df_usd[['balance', 'itd_pnl', 'mtd_pnl', 'qtd_pnl', 'ytd_pnl']] *= usdtusd_price

    # Modify the 'pm' column to add '_usd'
    df_usd['pm'] = df_usd['pm'] + '_usd'
    df_combined = pd.concat([df, df_usd], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset='pm', keep='first')

    if 'timestamp' in df_combined.columns:
        data_to_write = df_combined.copy()  # Avoid modifying original dataframe
        data_to_write['timestamp'] = data_to_write['timestamp'].dt.tz_localize(None)

    sheet_url = 'https://docs.google.com/spreadsheets/d/1-D4GjmjNfmnsgm8RIX4ge4kldWKWNyQSgnwXn82X5HM/edit?gid=0#gid=0'
    sheet_utils.set_dataframe(data_to_write, url=sheet_url, sheet_name='Sheet1', row=1, include_header=True)  
    # sheet_url = 'https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=0#gid=0'
    # sheet_utils.set_dataframe(df_combined, url=sheet_url, sheet_name='Sheet1', row=1, include_header=True) 
    print("Performance Dashboard Done")