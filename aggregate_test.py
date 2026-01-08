import pandas as pd
import credentials
import sheet_utils

def replace_aggregated_pnl(df, pm_data, pnl_columns='ITD_PnL'):
    """
    Replace aggregated PnL values with sum of their components
    
    Args:
        df: DataFrame with columns ['pm', 'ITD_PnL'] or ['pm', pnl_columns]
        pm_data: List of dictionaries defining the hierarchy
        pnl_columns: Name of PnL column (str) or list of PnL columns (list)
    
    Returns:
        DataFrame with updated aggregated values
    """
    
    # Convert pnl_columns to list if it's a single string
    if isinstance(pnl_columns, str):
        pnl_columns = [pnl_columns]
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Convert pm_data to DataFrame for easier manipulation
    pm_df = pd.DataFrame(pm_data)
    
    # Create mapping from pm to each hierarchy level
    pm_to_hierarchy = pm_df.set_index('pm').to_dict('index')
    
    # Step 1: Replace pm_group aggregated values with sum of individual PMs
    # Get all pm_groups that exist in the dataframe
    # pm_groups_in_df = result_df[result_df['pm'].isin(pm_df['pm_group'].unique())]['pm'].tolist()
    
    # for pm_group in pm_groups_in_df:
    #     # Find all PMs that belong to this pm_group
    #     pms_in_group = pm_df[pm_df['pm_group'] == pm_group]['pm'].tolist()
    #     # Filter to only PMs that exist in our dataframe
    #     existing_pms = result_df[result_df['pm'].isin(pms_in_group)]['pm'].tolist()
        
    #     if existing_pms:
    #         # Sum PnL of all PMs in this group for each column
    #         for pnl_col in pnl_columns:
    #             group_pnl_sum = result_df[result_df['pm'].isin(existing_pms)][pnl_col].sum()
    #             # Replace the pm_group row's PnL with the sum
    #             result_df.loc[result_df['pm'] == pm_group, pnl_col] = group_pnl_sum
    
    # Step 2: Replace group aggregated values with sum of pm_groups
    groups_in_df = result_df[result_df['pm'].isin(pm_df['group'].unique())]['pm'].tolist()
    print('groups_in_df')
    print(groups_in_df)
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
    fund_variants = base_funds + [f"{fund}-gross" for fund in base_funds]
    funds_in_df = result_df[result_df['pm'].isin(fund_variants)]['pm'].tolist()
    print('funds_in_df')
    print(funds_in_df)
    for fund_name in funds_in_df:
        # Determine the base fund name (remove -gross suffix if present)
        base_fund = fund_name.replace('-gross', '') if fund_name.endswith('-gross') else fund_name
        
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
    
    return result_df



sample_df = sheet_utils.get_dataframe(url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=0#gid=0', sheet_name='Sheet1')
sample_df = sample_df[['pm', 'ITD_PnL', 'MTD_PnL']]
print("Before aggregation:")
print(sample_df)
print("\nAfter aggregation:")
result = replace_aggregated_pnl(sample_df, credentials.PM_DATA, ['ITD_PnL', 'MTD_PnL'])
print(result)
sheet_utils.set_dataframe(df=result, url='https://docs.google.com/spreadsheets/d/1CYfRlJ--aQknxzSjdMft4puFg-vVb8VPo-3jlFVlqBU/edit?gid=0#gid=0', sheet_name='Sheet2')