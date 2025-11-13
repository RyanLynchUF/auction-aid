from typing import List
import pandas as pd
from functools import reduce
import re


def denormalize(df, column, values, columns_to_keep:List[str]=None, index:List[str]=['player_name'], groupby:List[str] = ['player_name', 'year']):
    dfs = []
    for value in values:
        df_agg = df.groupby(groupby).max().reset_index()
        df_pivot = df_agg.pivot(columns=column, values=value, index=index)
        df_pivot.columns = [f'{value}_{year}' for year in df_pivot.columns]
        dfs.append(df_pivot)
    final_df = reduce(lambda left, right: pd.concat([left, right], axis=1), dfs)

    """# Merge with non-metric columns, repeating the values for each row
    if columns_to_keep:
        final_df = final_df.reset_index(names='player_name')
        final_df = pd.merge(final_df, df[['player_name'] + columns_to_keep], on='player_name', how='inner')
        final_df = final_df.set_index(index)
"""
    return final_df

def normalize(df:pd.DataFrame, columns_to_normalize:List[str], columns_to_keep:List[str], on:List[str]=['player_name', 'year']):
    normalized_df = pd.DataFrame()

    for column in columns_to_normalize:
        # Select the columns for the current metric
        metric_columns = [col for col in df.columns if col.startswith(column)]
        
        # Melt the DataFrame for the current metric
        melted = df[metric_columns].reset_index().melt(id_vars='player_name', 
                                                    var_name='year', 
                                                    value_name=column)
        
        # Extract the year from the column names
        melted['year'] = melted['year'].str.extract(r'(\d{4})')
        
        # Merge the melted data into the final DataFrame
        if normalized_df.empty:
            normalized_df = melted
        else:
            normalized_df = pd.merge(normalized_df, melted, on=on, how='outer')

    # Merge with non-metric columns, repeating the values for each row
    final_df = pd.merge(normalized_df, df[columns_to_keep].reset_index(), on='player_name', how='left')

    # Sort by player_name and year for better readability
    final_df = final_df.sort_values(by=['player_name', 'year']).reset_index(drop=True)

    return final_df

def min_max_scale(series:pd.Series, invert_min_and_max:bool=False):
    min_val = series.min()
    max_val = series.max()

    if invert_min_and_max:
        return 1 - (series - min_val) / (max_val - min_val)
    else:
        return (series - min_val) / (max_val - min_val)

def get_available_years_for_metric(df_columns, metric_prefix: str, exclude_year: int = None) -> List[int]:
    """
    Extract sorted list of years that exist for a given metric prefix (e.g., 'bid_amt_', 'vbd_')
    
    Parameters:
    -----------
    df_columns : pd.Index or list
        Column names from a DataFrame
    metric_prefix : str
        The prefix to search for (e.g., 'bid_amt_', 'vbd_', 'projected_pos_rank_')
    exclude_year : int, optional
        Year to exclude from results (e.g., current league year)
    
    Returns:
    --------
    List[int]
        Sorted list of years found for the metric
    """
    pattern = re.compile(rf'{re.escape(metric_prefix)}(\d{{4}})$')
    years = []
    for col in df_columns:
        match = pattern.search(col)
        if match:
            year = int(match.group(1))
            if exclude_year is None or year != exclude_year:
                years.append(year)
    return sorted(years)
