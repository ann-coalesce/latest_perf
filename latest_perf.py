import db_utils
import sheet_utils
import pandas as pd
from datetime import datetime, timezone
import get_balance_nav
import get_transfers
import qtd_perf_metrics
import credentials
import time
import metrics_utils
import numpy as np
import get_exposure
import export_dashboard

# ----- updates 1. performance_metrics table 2. qtd metrics (in 2024Q4 PM Incentive Fee - Ver.2)
def job():
    end_time = None

    curr = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    print(curr)

    # ----- get processed dataframes -----
    bal_merged_df = get_balance_nav.get_bal_from_nav(end_time=end_time)
    nav_merged_df = get_balance_nav.get_nav(end_time=end_time)
    transfer_df = get_transfers.get_qtd_mtd_transfers(target_date=end_time)
    # grouping_df = pd.DataFrame(credentials.PM_DATA)
    query = "select * from pm_mapping;"
    grouping_df = db_utils.get_db_table(query=query)
    bal_merged_df = bal_merged_df.merge(grouping_df, on='pm', how='left')

    bal_merged_df = pd.merge(bal_merged_df, transfer_df, on='pm', how='left')

    bal_merged_df['mtd_transfer'] = bal_merged_df['mtd_transfer'].fillna(0)
    bal_merged_df['qtd_transfer'] = bal_merged_df['qtd_transfer'].fillna(0)
    bal_merged_df['ytd_transfer'] = bal_merged_df['ytd_transfer'].fillna(0)
    bal_merged_df['itd_transfer'] = bal_merged_df['itd_transfer'].fillna(0)

    # Calculate PnL
    # balance_inception is actually the sum of principal invested into a pm
    bal_merged_df['ITD_PnL'] = bal_merged_df['balance'] - bal_merged_df['itd_transfer']
    bal_merged_df['QTD_PnL'] = bal_merged_df['balance'] - bal_merged_df['balance_quarter_start'] - bal_merged_df['qtd_transfer']
    bal_merged_df['MTD_PnL'] = bal_merged_df['balance'] - bal_merged_df['balance_month_start'] - bal_merged_df['mtd_transfer']
    bal_merged_df['YTD_PnL'] = bal_merged_df['balance'] - bal_merged_df['balance_year_start'] - bal_merged_df['ytd_transfer']

    bal_merged_df['MTD_PnL'] = bal_merged_df['MTD_PnL'].fillna(bal_merged_df['ITD_PnL'])
    bal_merged_df['QTD_PnL'] = bal_merged_df['QTD_PnL'].fillna(bal_merged_df['ITD_PnL'])
    bal_merged_df['YTD_PnL'] = bal_merged_df['YTD_PnL'].fillna(bal_merged_df['ITD_PnL'])

    # forcing cash PnL to 0 (not including cash PnL to upper level groups)
    bal_merged_df.loc[bal_merged_df['pm'] == 'cash', ['ITD_PnL', 'MTD_PnL', 'QTD_PnL', 'YTD_PnL']] = 0

    # ----- calculate returns using NAV -----
    nav_merged_df['ITD_PnL%'] = nav_merged_df['nav'] / 1 - 1
    nav_merged_df['QTD_PnL%'] = nav_merged_df['nav'] / nav_merged_df['nav_quarter_start'] - 1
    nav_merged_df['MTD_PnL%'] = nav_merged_df['nav'] / nav_merged_df['nav_month_start'] - 1
    nav_merged_df['YTD_PnL%'] = nav_merged_df['nav'] / nav_merged_df['nav_year_start'] - 1

    nav_merged_df['QTD_PnL%'] = nav_merged_df['QTD_PnL%'].fillna(nav_merged_df['ITD_PnL%'])
    nav_merged_df['MTD_PnL%'] = nav_merged_df['MTD_PnL%'].fillna(nav_merged_df['ITD_PnL%'])
    nav_merged_df['YTD_PnL%'] = nav_merged_df['YTD_PnL%'].fillna(nav_merged_df['ITD_PnL%'])

    # ----- get exposure info -----
    exposure_df = get_exposure.get_exposure_df()

    # ----- merge balance (with pnl) and nav (with returns) -----
    merged_df = pd.merge(bal_merged_df, nav_merged_df, on='pm', how='inner')


    merged_df = pd.merge(merged_df, exposure_df, on='pm', how='left')

    exposure_adjustments = {
        'BRAVOWHISKEY_ACCOUNT': 100,
        'BRAVOWHISKEY_PM': 100,
        'BTC_FUND_LEVEL': 100
    }

    for pm_group, adjustment in exposure_adjustments.items():
        mask = merged_df['pm'].isin(credentials.__dict__[pm_group])
        merged_df.loc[mask, ['net_exposure', 'gross_exposure']] -= adjustment

    merged_df['net_exposure_ratio'] = merged_df['net_exposure'] / merged_df['balance']
    merged_df['gross_exposure_ratio'] = merged_df['gross_exposure'] / merged_df['balance']

    # ----- keep desired columns and rename them -----
    merged_df['timestamp'] = curr
    merged_df = merged_df[['timestamp','pm','balance','ITD_PnL','ITD_PnL%','MTD_PnL','MTD_PnL%','QTD_PnL','QTD_PnL%', 'YTD_PnL','YTD_PnL%', 'net_exposure_ratio', 'gross_exposure_ratio']]
    merged_df.rename(columns={
            "pm": "pm",
            "balance": "balance",
            "ITD_PnL": "itd_pnl",
            "ITD_PnL%": "itd_pnl_percentage",
            "MTD_PnL": "mtd_pnl",
            "MTD_PnL%": "mtd_pnl_percentage",
            "QTD_PnL": "qtd_pnl",
            "QTD_PnL%": "qtd_pnl_percentage",
            "YTD_PnL": "ytd_pnl",
            "YTD_PnL%": "ytd_pnl_percentage"}, inplace=True)

    merged_df[['net_exposure_ratio', 'gross_exposure_ratio']] = merged_df[['net_exposure_ratio', 'gross_exposure_ratio']].fillna(0)
    merged_df = merged_df.fillna(0)
    # merged_df.replace([np.inf, -np.inf], 0, inplace=True)
    merged_df = merged_df.replace([np.inf, -np.inf], 0)
    # print('merged_df')
    # print(merged_df)

    metrics_df = metrics_utils.get_metrics()
    merged_df = pd.merge(merged_df, metrics_df, on='pm', how='left')

    print('perf metrics table')
    print(merged_df)
    
    price_sql_query = "select price from prices where symbol = 'USDTUSD';"
    usdtusd_price = db_utils.get_db_table(query=price_sql_query).iloc[0,0]

    export_dashboard.export_dashboard(merged_df.copy(), usdtusd_price)


    db_utils.df_replace_table(table_name='performance_metrics', df=merged_df)
    print('updated performance_metrics')

    qtd_metrics = qtd_perf_metrics.get_qtd_sharpe_df(ending_time=end_time)
    merged_df = merged_df[['pm','qtd_pnl','qtd_pnl_percentage', 'timestamp']]
    qtd_metrics = pd.merge(qtd_metrics, merged_df, on='pm', how='left')
    qtd_metrics = qtd_metrics[['pm', 'sharpe','qtd_pnl','qtd_pnl_percentage', 'timestamp']]
    qtd_metrics.dropna(inplace=True)
    qtd_metrics.replace([np.inf, -np.inf], 0, inplace=True)

    quarter = (curr.month - 1) // 3 + 1
    sheet_name = f'{curr.year}Q{quarter}'
    month_in_quarter = (curr.month - 1) % 3 + 1
    column = (month_in_quarter - 1) * 4 + month_in_quarter
    print(sheet_name, month_in_quarter, column)

    sheet_utils.set_dataframe(df=qtd_metrics, url='https://docs.google.com/spreadsheets/d/1wMaan-MEcLdGKGIIYm0r3iaw8p_bkuzQ-Nj55rhKGuI/edit?gid=0#gid=0', sheet_name=sheet_name, col=column)
    print('qtd_metrics')
    print(qtd_metrics)
    print('done')

if __name__ == "__main__":
    start = time.time()
    job()
    end = time.time()
    print('time taken', end-start)