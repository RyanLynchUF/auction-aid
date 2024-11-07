import pandas as pd
import numpy as np
from typing import List
import re


from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR


def create_league_settings(league):
    """
    Create a dictionary of key league settings
    
    Parameters:
    -----------
    league : League
        A League object
    """
    scoring_format_dict = {item['id']: item for item in league.settings.scoring_format}
    
    league_settings = {
        'league_id': [league.league_id][0],
        'league_size': [len(league.teams)][0],
        'scoring_format_' + scoring_format_dict[53]['abbr'] : [scoring_format_dict[53]['points']][0], # Points per Reception
        'scoring_format_' + scoring_format_dict[4]['abbr'] : [scoring_format_dict[4]['points']][0], # Passing TDs
        'team_composition': {
            'QB': league.settings.position_slot_counts['QB'],
            'TQB': league.settings.position_slot_counts['TQB'],
            'RB': league.settings.position_slot_counts['RB'],
            'RBWR': league.settings.position_slot_counts['RB/WR'],
            'WR': league.settings.position_slot_counts['WR'],
            'WRTE': league.settings.position_slot_counts['WR/TE'],
            'TE': league.settings.position_slot_counts['TE'],
            'OP': league.settings.position_slot_counts['OP'],
            'DT': league.settings.position_slot_counts['DT'],
            'DE': league.settings.position_slot_counts['DE'],
            'LB': league.settings.position_slot_counts['LB'],
            'DL': league.settings.position_slot_counts['DL'],
            'CB': league.settings.position_slot_counts['CB'],
            'S': league.settings.position_slot_counts['S'],
            'DB': league.settings.position_slot_counts['DB'],
            'DP': league.settings.position_slot_counts['DP'],
            'DST': league.settings.position_slot_counts['D/ST'],
            'K': league.settings.position_slot_counts['K'],
            'P': league.settings.position_slot_counts['P'],
            'HC': league.settings.position_slot_counts['HC'],
            'BE': league.settings.position_slot_counts['BE'],
            'IR': league.settings.position_slot_counts['IR'],
            'RBWRTE': league.settings.position_slot_counts['RB/WR/TE'],
            'ER': league.settings.position_slot_counts['ER']
        }
    }

    return league_settings

def generate_player_features(input_features:pd.DataFrame, included_past_seasons:List[int], statistic_for_vorp_calculation:str):
    """
    Generate player-specific features to be used in modelling auction values
    
    Parameters:
    -----------
    input_features : DataFrame
        Set of data to use to create the player features
    include_past_seasons: List[int]
        List of years to include in creating of features.  Used to help create training vs. prediction features
    statistic_for_vorp_calculation
        Determine if PPG or Total Points should be used for certain calculations
    """

    # Calculate the ratio of Expert VBD to Past Auction Values for each year
    for year in included_past_seasons:
        vbd_col = f'vbd_{year}'
        auction_value_col = f'bid_amt_{year}'
        feature_name = f'ratio_of_vbd_to_auction_value_for_{year}'
        # Avoid division by zero or NaN values
        input_features[feature_name] = input_features.apply(
            lambda row: row[vbd_col] / row[auction_value_col] if pd.notna(row[vbd_col]) and pd.notna(row[auction_value_col]) and row[auction_value_col] != 0 else np.nan,
            axis=1)
        
        
    # Calculate the average Total Points or PPG for the player for the past 
    total_years = len(included_past_seasons)
    number_of_years_for_average = min(len(included_past_seasons), 5)
    years_for_average = [included_past_seasons[i] for i in range((total_years - 1), total_years - number_of_years_for_average, -1)]
    if statistic_for_vorp_calculation == 'ppg':
        columns = [f'ppg_{year}' for year in included_past_seasons]
    elif statistic_for_vorp_calculation == 'total_points':
        columns = [f'total_points_{year}' for year in years_for_average]
    else:
        raise ValueError("Invalid metric selected. Choose either 'ppg' or 'total_points'.")
    input_features[f'avg_{statistic_for_vorp_calculation}_from_league_history'] = input_features[columns].mean(axis=1)

    # Calculate average bid_amt for the past
    columns = [f'bid_amt_{year}' for year in years_for_average]
    input_features[f'avg_bid_amt_from_league_history'] = input_features[columns].mean(axis=1)

    # Calculate average ratio of Expert VBD to Past Auction Values
    columns = [f'ratio_of_vbd_to_auction_value_for_{year}' for year in years_for_average]
    input_features[f'avg_ratio_of_vbd_to_auction_value_from_league_history'] = input_features[columns].mean(axis=1)

    # One-hot encoding the 'pos' and 'team' columns
    pos_encodings = pd.get_dummies(input_features['pos'], prefix='pos')
    team_encodings = pd.get_dummies(input_features['team'], prefix='team')
    input_features = pd.concat([input_features, pos_encodings, team_encodings], axis=1)

    # Select the previous year's bid_amt
    input_features['prev_year_bid_amt'] = input_features['bid_amt_' + str(max(included_past_seasons))]

    # Determine the max year in the dataset to see if we are working with training or prediction data
    df_columns = input_features.columns  
    years = [int(re.search(r'(\d{4})$', col).group(0)) for col in df_columns if re.search(r'(\d{4})$', col) and str(CURR_LEAGUE_YR) not in col]
    max_year_in_data = max(years) if years else None

    if max(included_past_seasons) == max_year_in_data:
        input_features['curr_year_bid_amt'] = np.nan
        input_features['curr_year_vbd'] = input_features['vbd_' + str(CURR_LEAGUE_YR)]
        input_features['curr_year_projected_pos_rank'] = input_features['projected_pos_rank_' + str(CURR_LEAGUE_YR)]
    else: 
        input_features['curr_year_bid_amt'] = input_features['bid_amt_' + str(max(included_past_seasons) + 1)]
        input_features['curr_year_vbd'] = input_features['vbd_' + str(max(included_past_seasons) + 1)]
        input_features['curr_year_projected_pos_rank'] = input_features['projected_pos_rank_' + str(max(included_past_seasons) + 1)]

    
    # Select only columns needed for modelling
    player_features = input_features[['pos', 'curr_year_bid_amt', 'prev_year_bid_amt',
        'curr_year_vbd', 'curr_year_projected_pos_rank',
       f'avg_{statistic_for_vorp_calculation}_from_league_history', 
       f'avg_bid_amt_from_league_history',
       f'avg_ratio_of_vbd_to_auction_value_from_league_history',
       'pos_DST', 'pos_K', 'pos_QB', 'pos_RB', 'pos_TE',
       'pos_WR', 'team_ARI', 'team_ATL', 'team_BAL', 'team_BUF', 'team_CAR',
       'team_CHI', 'team_CIN', 'team_CLE', 'team_DAL', 'team_DEN', 'team_DET',
       'team_GB', 'team_HOU', 'team_IND', 'team_JAC', 'team_KC', 'team_LAC',
       'team_LAR', 'team_LV', 'team_MIA', 'team_MIN', 'team_NE', 'team_NO',
       'team_NYG', 'team_NYJ', 'team_PHI', 'team_PIT', 'team_SEA', 'team_SF',
       'team_TB', 'team_TEN', 'team_WAS']]

    return player_features


def generate_projected_position_rank_features(input_player_features:pd.DataFrame, included_past_seasons:List[int]):
    """
    Generate position rank (i.e., QB1, WR4, etc.) features to be used in modelling auction values
    
    Parameters:
    -----------
    input_features : DataFrame
        Set of data to use to create the player features
    include_past_seasons: List[int]
        List of years to include in creating of features.  Used to help create training vs. prediction features
    """

    projected_pos_rank_bid_amts_by_year = []
    for year in included_past_seasons:
        temp_df = input_player_features[[f'bid_amt_{year}', f'projected_pos_rank_{year}', 'pos']].copy()
        temp_df = temp_df.rename(
            columns={f'bid_amt_{year}': 'bid_amt', f'projected_pos_rank_{year}': 'projected_pos_rank'}
        )
        projected_pos_rank_bid_amts_by_year.append(temp_df)

    # Concatenate all years into one DataFrame
    merged_df = pd.concat(projected_pos_rank_bid_amts_by_year)

    # Group by 'pos' and 'projected_pos_rank' and calculate average bid amount
    projected_pos_rank_bid_amts = merged_df.groupby(['pos', 'projected_pos_rank'])['bid_amt'].mean().reset_index()
    projected_pos_rank_bid_amts.columns = ['pos', 'projected_pos_rank', 'projected_pos_rank_avg_bid_amt']

    return projected_pos_rank_bid_amts



