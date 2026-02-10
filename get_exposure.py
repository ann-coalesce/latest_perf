import db_utils
from datetime import datetime, timedelta, timezone
import pandas as pd
import credentials
import sheet_utils

def get_exposure_df():
    curr = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
    query = f'''SELECT 
            timestamp, 
            pm,
            symbol,
            exposure
        FROM 
            position_all_consolidated 
        WHERE 
            timestamp = '{curr}' AND symbol NOT IN ('USDT', 'FDUSD', 'USDC')
    '''

    exposure_df = db_utils.get_db_table(query=query)
    # grouping_df = pd.DataFrame(credentials.PM_DATA)
    query = "select * from pm_mapping;"
    grouping_df = db_utils.get_db_table(query=query)
    grouping_df['if_btc'] = grouping_df['if_btc'].fillna(False).astype(bool)
    exposure = pd.merge(exposure_df, grouping_df, on='pm', how='left')
    exposure = exposure.drop(exposure[(exposure["symbol"] == "BTC") & (exposure["if_btc"] == True)].index)

    exposure_agg = exposure.groupby(by=['timestamp', 'pm'], as_index=False).agg(
        net_exposure=('exposure', 'sum'),
        gross_exposure=('exposure', lambda x: x.abs().sum())
    )

    exposure_agg = pd.merge(exposure_agg, grouping_df, on='pm', how='left')

    pm_grouped = exposure_agg.groupby(by='pm_group').aggregate({'net_exposure':'sum', 'gross_exposure':'sum'}).reset_index()
    pm_grouped.rename(columns={'pm_group':'pm'}, inplace=True)
    
    group_grouped = exposure_agg.groupby(by='group').aggregate({'net_exposure':'sum', 'gross_exposure':'sum'}).reset_index()
    group_grouped.rename(columns={'group':'pm'}, inplace=True)

    fund_grouped = exposure_agg.groupby(by='fund').aggregate({'net_exposure':'sum', 'gross_exposure':'sum'}).reset_index()
    fund_grouped.rename(columns={'fund':'pm'}, inplace=True)
    duplicated_rows = fund_grouped[fund_grouped['pm'] == 'sp1'].copy()
    duplicated_rows['pm'] = 'sp1-gross'

    duplicated_rows_2 = fund_grouped[fund_grouped['pm'] == 'sp2'].copy()
    duplicated_rows_2['pm'] = 'sp2-gross'

    duplicated_rows_3 = fund_grouped[fund_grouped['pm'] == 'sp2-classb'].copy()
    duplicated_rows_3['pm'] = 'sp2-classb-gross'

    duplicated_rows_4 = fund_grouped[fund_grouped['pm'] == 'sp3'].copy()
    duplicated_rows_4['pm'] = 'sp3-gross'

    duplicated_rows_5 = fund_grouped[fund_grouped['pm'] == 'sp2-classa'].copy()
    duplicated_rows_5['pm'] = 'sp2-classa-gross'

    fund_grouped = pd.concat([fund_grouped, duplicated_rows, duplicated_rows_2, duplicated_rows_3, duplicated_rows_4, duplicated_rows_5])


    exposure_merged = pd.concat([exposure_agg, pm_grouped, group_grouped, fund_grouped], ignore_index=True)

    return exposure_merged[['pm', 'net_exposure', 'gross_exposure']]

if __name__ == "__main__":
    df = get_exposure_df()
    print(df)