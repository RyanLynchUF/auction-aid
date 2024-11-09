
from collections import defaultdict
from typing import List, Dict
import pandas as pd
import numpy as np

from config import settings
from utils.helper import denormalize

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR

def transform_latest_player_projections(league_size, latest_projections, latest_auction_value):
    """
    Transform the latest player projection data to format for models and addition
    
    Parameters:
    -----------
    scoring_format : str
        The league scoring format: ppr, half-ppr, or standard (no points per reception)
    s3 : bool
        Boolean value to indicate if teh data should be read from s3
    """

    # Players expected to be drafted in the current year
    player_names = [name for name in latest_auction_value.keys()]

    # Player Fantasy Pros Metrics
    ecr_dict = {item['player_name']: item for item in latest_projections['ecr']['players']} 
    ecr_dict = defaultdict(lambda: {'player_position_id': '', 'rank_ecr': np.nan, 'rank_ave': np.nan}, ecr_dict) 
    adp_dict = defaultdict(lambda: {'adp_avg': np.nan}, latest_projections['adp']) 
    vbd_dict = latest_projections['vbd'][str(league_size)]
    vbd_dict = defaultdict(lambda: {'rank': np.nan, 'vbd': np.nan, 'vorp': np.nan, 'vols': np.nan}, vbd_dict) 

    # Input data from upcoming year metrics from FantasyPros
    player_projections_input_features = {
        'player_name': player_names,
        'team': [latest_auction_value[player]['team'] for player in player_names],
        'pos': [ecr_dict[player]['player_position_id'] for player in player_names],
        'proj_points': [latest_auction_value[player]['points'] for player in player_names],
        'orig_auction_value': [latest_auction_value[player]['value'] for player in player_names],
        'ecr': [ecr_dict[player]['rank_ecr'] for player in player_names], 
        'ecr_avg': [ecr_dict[player]['rank_ave'] for player in player_names],
        'vbd_' + str(CURR_LEAGUE_YR): [vbd_dict[player]['vbd'] for player in player_names],
        'vorp': [vbd_dict[player]['vorp'] for player in player_names],
        'vols': [vbd_dict[player]['vols'] for player in player_names],
        'adp': [adp_dict[player]['adp_avg'] for player in player_names]
        }

    latest_player_projections_df = pd.DataFrame(player_projections_input_features)

    # Update names of defensive players to match other data sources
    latest_player_projections_df.loc[latest_player_projections_df['pos'] == 'DST', 'player_name'] \
                                    = latest_player_projections_df.loc[latest_player_projections_df['pos'] == 'DST', 'player_name'] \
                                        .apply(lambda name: f"{name.split()[-1]} D/ST")
    
    latest_player_projections_df = latest_player_projections_df.dropna()
    latest_player_projections_df = latest_player_projections_df.drop(labels=['ecr', 'vorp', 'vols', 'adp'], axis=1)
    latest_player_projections_df['projected_pos_rank_' + str(CURR_LEAGUE_YR)] = latest_player_projections_df.groupby('pos')['ecr_avg'].rank(ascending=True)    


    return latest_player_projections_df.set_index('player_name')

def transform_past_player_stats(past_player_stats:Dict, league_settings:Dict, years:List[int]):

    past_player_stats = {year: past_player_stats[str(year)] for year in years if str(year) in past_player_stats}

    # Input data based on player's performance in past years in the league
    records = [
        {**player_data, 'year': year}
        for year, players in past_player_stats.items()
        for player_data in players
    ]

    past_player_stats_df = pd.DataFrame.from_records(records).set_index('player_name')

    past_player_stats_df = past_player_stats_df[past_player_stats_df['team'].notna()]

    past_player_stats_df['actual_pos_rank_total_points'] =  past_player_stats_df.groupby(['pos', 'year'])['total_points'].rank(method='first', ascending=False)
    past_player_stats_df['actual_pos_rank_ppg'] = past_player_stats_df.groupby(['pos', 'year'])['ppg'].rank(method='first', ascending=False)

    past_player_stats_df = denormalize(past_player_stats_df, 'year', ['actual_pos_rank_ppg', 'ppg', 'actual_pos_rank_total_points', 'total_points', 'team'],
                                       index=['player_name', 'pos'], groupby=['player_name', 'year', 'pos'])
    past_player_stats_df = past_player_stats_df.reset_index()
    
    # Replace actual_pos_rank of 0 with max((actual_pos_rank)) grouped by pos
    actual_pos_rank_columns = [col for col in past_player_stats_df.columns if col.startswith('actual_pos_rank_')]
    for col in actual_pos_rank_columns:
        past_player_stats_df[col] = past_player_stats_df.groupby('pos')[col].transform(
            lambda x: x.replace(0, x.max())
        )

    # Remove any players from positions that are not used for the league
    valid_positions = [pos for pos, count in league_settings['team_composition'].items() if count > 0]
    past_player_stats_df = past_player_stats_df[past_player_stats_df['pos'].isin(valid_positions)]

    return past_player_stats_df.set_index('player_name')

def transform_past_auction_values(past_leagues:Dict, years:List[int]):

    past_leagues = {year: past_leagues[str(year)] for year in years if str(year) in past_leagues}
    
    # Input data based on results of previous drafts in the league
    past_auction_values_df = get_auction_draft_data(past_leagues)

    # Determine if any of the leagues have no bid data, which means it was likely a snake draft
    years_with_all_zero_bids = past_auction_values_df.groupby('year')['bid_amt'].apply(lambda x: (x == 0).all())
    years_with_all_zero_bids = years_with_all_zero_bids[years_with_all_zero_bids].index.tolist()
    if years_with_all_zero_bids:
        return years_with_all_zero_bids

    past_auction_values_df = denormalize(past_auction_values_df, 'year', ['bid_amt'])

    auction_value_columns = [col for col in past_auction_values_df.columns if col.startswith('bid_amt_')]
    past_auction_values_df[auction_value_columns] = past_auction_values_df[auction_value_columns].replace(np.nan, 0)

    return past_auction_values_df

def transform_past_player_projections(league_size:int, past_player_projections:Dict, years:List[int]):
    
    years_string = [str(year) for year in years]

    # Input data from historical years of metrics from FantasyPros
    past_player_projections_df = pd.DataFrame(past_player_projections)
    past_player_projections_df = past_player_projections_df[past_player_projections_df['year'].isin(years_string)]
    past_player_projections_df = past_player_projections_df[past_player_projections_df['league_size'] == str(league_size)]
    
    # Update name of defensive players to match other data sources
    past_player_projections_df.loc[past_player_projections_df['pos'] == 'DST', 'player_name'] = \
        past_player_projections_df.loc[past_player_projections_df['pos'] == 'DST', 'player_name'].apply(
            lambda x: f"{x.split()[-1]} D/ST")
    
    # Denormalize data into format needed to train models
    past_player_projections_df = denormalize(past_player_projections_df, "year", ['adp_avg', 'vbd', 'projected_pos_rank'], columns_to_keep=['pos'])

    # Remove players no longer likely to be draft (nan value for each of the past 3 years)
    adp_columns = [col for col in past_player_projections_df.columns if col.startswith('adp_avg_')]
    vbd_columns = [col for col in past_player_projections_df.columns if col.startswith('vbd_')]
    adp_last_3_years = sorted(adp_columns, reverse=True)[:3] # Sort to select past 3 years
    vbd_last_3_years = sorted(vbd_columns, reverse=True)[:3] # Sort to select past 3 years
    columns_to_keep = adp_last_3_years + vbd_last_3_years
    past_player_projections_df = past_player_projections_df.dropna(subset=columns_to_keep, how='all')

    return past_player_projections_df

def get_auction_draft_data(leagues):
    draft_history_df = pd.DataFrame()

    for year, league in leagues.items():
        draft_history = {"year": [year] * len(league.draft),
            "player_name": [pick.playerName for pick in league.draft],
            "bid_amt": [pick.bid_amount for pick in league.draft]}

        draft_history_df = pd.concat([draft_history_df, pd.DataFrame(draft_history)])

    return draft_history_df


