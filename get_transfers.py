import db_utils
import sheet_utils
import pandas as pd
from datetime import datetime, timezone

def get_qtd_mtd_transfers(target_date=None):
    # ---- get target date ----
    if target_date is None:
        target_date = datetime.now(timezone.utc)
    else:
        target_date = pd.Timestamp(target_date).tz_localize(timezone.utc)

    # ---- get specified quarter and month ----
    target_quarter = pd.Period(target_date, freq='Q')
    target_month = pd.Period(target_date, freq='M')
    target_year = pd.Period(target_date, freq='Y')

    # ---- get transfer history ----
    query = 'select timestamp, pm, transfer_amount from shares_table;'
    transfer_df = db_utils.get_db_table(query=query)

    transfer_df['quarter'] = transfer_df['timestamp'].dt.to_period('Q')
    transfer_df['month'] = transfer_df['timestamp'].dt.to_period('M')
    transfer_df['year'] = transfer_df['timestamp'].dt.to_period('Y')

    # ---- filter out quarter-to-date and month-to-date ----
    qtd_transfer = transfer_df[
        (transfer_df['quarter'] == target_quarter) & (transfer_df['timestamp'] <= target_date)
    ].groupby(['quarter', 'pm'], as_index=False).aggregate({'transfer_amount': 'sum'})[['quarter', 'pm', 'transfer_amount']].reset_index(drop=True)

    mtd_transfer = transfer_df[
        (transfer_df['month'] == target_month) & (transfer_df['timestamp'] <= target_date)
    ].groupby(['month', 'pm'], as_index=False).aggregate({'transfer_amount': 'sum'})[['month', 'pm', 'transfer_amount']].reset_index(drop=True)

    ytd_transfer = transfer_df[
        (transfer_df['year'] == target_year) & (transfer_df['timestamp'] <= target_date)
    ].groupby(['year', 'pm'], as_index=False).aggregate({'transfer_amount': 'sum'})[['year', 'pm', 'transfer_amount']].reset_index(drop=True)


    itd_transfer = transfer_df.groupby(['pm'], as_index=False).aggregate({'transfer_amount': 'sum'})[['pm', 'transfer_amount']].reset_index(drop=True)

    # ---- rename columns ----
    qtd_transfer.rename(columns={'transfer_amount': 'qtd_transfer'}, inplace=True)
    mtd_transfer.rename(columns={'transfer_amount': 'mtd_transfer'}, inplace=True)
    ytd_transfer.rename(columns={'transfer_amount': 'ytd_transfer'}, inplace=True)
    itd_transfer.rename(columns={'transfer_amount': 'itd_transfer'}, inplace=True)

    # ---- merge qtd transfers and mtd transfers ----
    result_df = pd.merge(qtd_transfer, mtd_transfer, on='pm', how='outer')
    result_df = pd.merge(result_df, itd_transfer, on='pm', how='outer')
    result_df = pd.merge(result_df, ytd_transfer, on='pm', how='outer')
    result_df = result_df[['pm', 'qtd_transfer', 'mtd_transfer', 'ytd_transfer','itd_transfer']]

    return result_df


# result_df = get_qtd_mtd_transfers(target_date='2025-01-07')
# sheet_utils.set_dataframe(df=result_df, url='https://docs.google.com/spreadsheets/d/1vGzpotR0-VecbQvLqWKNAYZhTTmPkOylSymLzfQYrp4/edit?gid=990963253#gid=990963253', sheet_name='transfer_test', row=1)
