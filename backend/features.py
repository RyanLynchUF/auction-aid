import pandas as pd
import numpy as np

from config import settings
from utils.helper import normalize, min_max_scale
from utils.logger import get_logger

logger = get_logger(__name__)

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR


def generate_league_features(league):

    scoring_format_dict = {item['id']: item for item in league.settings.scoring_format}
    
    league_features = {
        'league_id': [league.league_id][0],
        'league_size': [len(league.teams)][0],
        'scoring_format_' + scoring_format_dict[53]['abbr'] : [scoring_format_dict[53]['points']][0], # Points per Reception
        'scoring_format_' + scoring_format_dict[4]['abbr'] : [scoring_format_dict[4]['points']][0], # Passing TDs
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

    return pd.DataFrame(league_features, index=[0])

def generate_player_features(input_features, included_past_seasons, vorp_configuration:dict):
    
    #  Select statistics to use for features
    statistic_for_vorp_calculation = vorp_configuration['statistic_for_vorp_calculation'].lower()

    # player_features_df = input_features.dropna(subset=['ecr_avg']) # TODO: Add back when we have historical ecr_avg
    
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
    
    if max(included_past_seasons) + 1 == CURR_LEAGUE_YR:
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

def generate_actual_position_rank_features(league_features, input_player_features, latest_player_projections, included_past_seasons,
                                    auction_dollars_per_team, vorp_configuration):
    #TODO: Read this and go over this again with fresh eyes https://pitcherlist.com/fantasy-101-how-to-turn-projections-into-rankings-and-auction-values/
    #TODO: Find a better approach for addressing $1 players and filling out the bench
    #TODO: Ensure $1 is alotted to each remaining bench spot

    #TODO: Remove concatenated column names and use variable in all of code base
    league_size = league_features['league_size'][0]

    # Parameters used in calculation of league-specific VORP
    statistic_for_vorp_calculation = vorp_configuration['statistic_for_vorp_calculation'].lower()
    team_draft_strategy = vorp_configuration['team_draft_strategy']
    baseline_player_strategy = vorp_configuration['baseline_player_strategy']
    parameters_for_find_baseline_player = {'baseline_player_strategy':baseline_player_strategy, 
                                        'statistic_for_vorp_calculation':statistic_for_vorp_calculation}

    actual_pos_rank_column = 'actual_pos_rank_' + statistic_for_vorp_calculation

    # Determine the number of slots remaining to be drafted and the total number of auction dollars remaining
    #TODO: NEW FEATURE - REAL-TIME UPDATES: update to consider live values "# of teams * budget - number of player slots"
    team_position_counts = {
        'QB': league_features['QB'],
        'TQB': league_features['TQB'],
        'RB': league_features['RB'],
        'RBWR': league_features['RBWR'],
        'WR': league_features['WR'],
        'WRTE': league_features['WRTE'],
        'TE': league_features['TE'],
        'OP': league_features['OP'],
        'DT': league_features['DT'],
        'DE': league_features['DE'],
        'LB': league_features['LB'],
        'DL': league_features['DL'],
        'CB': league_features['CB'],
        'S': league_features['S'],
        'DB': league_features['DB'],
        'DP': league_features['DP'],
        'DST': league_features['DST'],
        'K': league_features['K'],
        'P': league_features['P'],
        'HC': league_features['HC'],
        'BE': league_features['BE'],
        'IR': league_features['IR'],
        'RBWRTE': league_features['RBWRTE'],
        'ER': league_features['ER']
    }
    
    starting_player_counts, bench_player_counts = count_of_positions_for_auction_dollars(league_size, team_position_counts, latest_player_projections)
    total_player_counts = starting_player_counts + bench_player_counts

    actual_position_ranks_with_vorp = calculate_value_over_replacement_player(input_player_features, included_past_seasons, vorp_configuration)

    count_of_positions_to_spend_on = actual_position_ranks_with_vorp[actual_position_ranks_with_vorp['vorp'] > 0].groupby('pos').size()
    count_of_one_dollar_positions = (total_player_counts['count'] - count_of_positions_to_spend_on).fillna(total_player_counts['count'])
    total_remaining_auction_dollars = league_size * int(auction_dollars_per_team) - count_of_one_dollar_positions.sum()
    
    # Create a final dataframe of all positions ranks who should be drafted
    drafted_pos_ranks = pd.DataFrame()
    for pos, count in total_player_counts['count'].items():
        # Get the top 'count' rows for each 'pos' in df1, sorted by 'actual_pos_rank_ppg' in descending order
        top_rows = actual_position_ranks_with_vorp[actual_position_ranks_with_vorp['pos'] == pos].nsmallest(count, 'actual_pos_rank_ppg')
        drafted_pos_ranks = pd.concat([drafted_pos_ranks, top_rows])
    drafted_pos_ranks = drafted_pos_ranks.set_index(['pos', actual_pos_rank_column])

    # Drop all players below baseline player's position rank.  They are assumed to be worth 1 or 0 dollars.
    actual_position_ranks_with_vorp = actual_position_ranks_with_vorp[actual_position_ranks_with_vorp[actual_pos_rank_column] < actual_position_ranks_with_vorp['baseline_pos_rank']]
    actual_position_ranks_with_vorp = actual_position_ranks_with_vorp[actual_position_ranks_with_vorp['pos'].isin(['QB', 'RB', 'WR', 'TE'])]
    actual_position_ranks_with_vorp = actual_position_ranks_with_vorp.set_index(['pos', actual_pos_rank_column])

    # Determine what proportion of points are from each position group (what are the most valuable positions)
    actual_position_ranks_with_vorp[statistic_for_vorp_calculation + '_percentage'] = actual_position_ranks_with_vorp.groupby(level='pos').apply(
        lambda x: x[statistic_for_vorp_calculation] / x[statistic_for_vorp_calculation].sum()
    ).reset_index(level=0, drop=True)
    
    # Calculate the total VORP for each position
    statistics_by_position = actual_position_ranks_with_vorp.groupby(level='pos').apply(
        lambda x: x['vorp'].sum())
    statistics_by_position = statistics_by_position.to_frame(name='total_vorp')

    # Calculate the proportion of total VORP for each position
    statistics_by_position['proportion_of_total_vorp'] = statistics_by_position / statistics_by_position.sum()

    # Calculate total value for each position group based on available auction dollars
    statistics_by_position['position_auction_value_by_vorp'] = statistics_by_position['proportion_of_total_vorp'] * total_remaining_auction_dollars
    
    # Based on players expected to be drafted, deteremine the total value for that position
    statistics_by_position['total_value_to_spend_by_pos'] = statistics_by_position['position_auction_value_by_vorp'] * starting_player_counts['count']
    # Total value across all positions
    total_value_to_spend = statistics_by_position['total_value_to_spend_by_pos'].sum()
    # Determine the proportion of total value to dollars to spend
    vorp_value_factor = total_value_to_spend / total_remaining_auction_dollars

    # Adjust vorp values based value and dollars available in league
    statistics_by_position['league_adjusted_vorp'] = statistics_by_position['total_value_to_spend_by_pos'] / vorp_value_factor
    statistics_by_position = statistics_by_position.reset_index(names='pos')
    actual_position_ranks_with_vorp = actual_position_ranks_with_vorp.reset_index()
    actual_position_ranks_with_vorp = pd.merge(actual_position_ranks_with_vorp, statistics_by_position[['pos', 'league_adjusted_vorp']], on='pos').set_index(['pos', actual_pos_rank_column])

    # Calculate true auction values for players based on team_draft_strategy
    if team_draft_strategy == 'Balanced':
        # Determine true auction values
        actual_position_ranks_with_vorp['true_auction_value'] = actual_position_ranks_with_vorp[statistic_for_vorp_calculation + '_percentage'] * actual_position_ranks_with_vorp['league_adjusted_vorp']
    else:
        # The crossover parameter sets the point at which the sigmoid curve crosses over from increasing to decreasing slope
        crossover = max(actual_position_ranks_with_vorp['vorp']) * .50 # Higher number means more "valuable" players (quantity)
        sigmoid_denominator = 20 # High number means lower value of top players
        
        # Calculate vorp value to use in the sigmoid function
        actual_position_ranks_with_vorp['vorp_for_sigmoid'] = ((actual_position_ranks_with_vorp['vorp']) - crossover) / sigmoid_denominator
        actual_position_ranks_with_vorp['sigmoid'] = 1 / (1 + np.exp(-actual_position_ranks_with_vorp['vorp_for_sigmoid']))
        # Determine true auction values
        actual_position_ranks_with_vorp['true_auction_value'] = total_remaining_auction_dollars / actual_position_ranks_with_vorp['sigmoid'].sum() * actual_position_ranks_with_vorp['sigmoid']
    
    # Create final Dataframe, which contains all drafted players
    drafted_pos_ranks = drafted_pos_ranks.join(actual_position_ranks_with_vorp['true_auction_value'], how='left')
    drafted_pos_ranks['true_auction_value'] = drafted_pos_ranks['true_auction_value'].fillna(1)

    return drafted_pos_ranks[['vorp', 'actual_pos_rank_avg_bid_amt', 'true_auction_value']]

def calculate_value_over_replacement_player(input_player_features, included_past_seasons, vorp_configuration):
    # Parameters used in calculation of league-specific VORP
    statistic_for_vorp_calculation = vorp_configuration['statistic_for_vorp_calculation'].lower()
    team_draft_strategy = vorp_configuration['team_draft_strategy']
    baseline_player_strategy = vorp_configuration['baseline_player_strategy']
    parameters_for_find_baseline_player = {'baseline_player_strategy':baseline_player_strategy, 
                                        'statistic_for_vorp_calculation':statistic_for_vorp_calculation}
    
    actual_pos_rank_column = 'actual_pos_rank_' + statistic_for_vorp_calculation

    # Normalize data to prep for VORP calculations
    input_player_features = normalize(input_player_features, 
                                  columns_to_normalize=['ppg', actual_pos_rank_column, 'total_points', 'bid_amt'],
                                  columns_to_keep=['pos'])

    # Filter the input dataframe to only include years needed (training data vs. current year prediction data)
    input_player_features = input_player_features[input_player_features['year'].isin([str(year) for year in included_past_seasons])]               
    input_player_features = input_player_features.set_index(['player_name', 'year'])
    
    # Replace nan with 0 for actual_auction_value column to tie non-drafted players to a an auction value of 0.
    input_player_features['bid_amt'] = input_player_features['bid_amt'].fillna(0)

    # Create new dataframe to calculate statistcs based on the actual_pos_rank
    statistics_by_position_rank = input_player_features[['pos', actual_pos_rank_column, 'ppg', 'total_points', 'bid_amt']]

    # Filter out all players with nan values in ['ppg', actual_pos_rank_column, 'total_points'], but keep DST players
    statistics_by_position_rank = statistics_by_position_rank.dropna(how='all', subset=['ppg', actual_pos_rank_column, 'total_points'])
    statistics_by_position_rank[['ppg', 'total_points']] = statistics_by_position_rank[['ppg', 'total_points']].fillna(1)
    statistics_by_position_rank = statistics_by_position_rank.set_index(['pos', actual_pos_rank_column])

    # Get average values for the statistical performance and actual_auction_value amounts for each position rank
    statistics_by_position_rank = statistics_by_position_rank.groupby(['pos', actual_pos_rank_column]).agg('mean')
    statistics_by_position_rank = statistics_by_position_rank.rename(columns={'bid_amt':'actual_pos_rank_avg_bid_amt'})

    # Determine the baseline player.  Then, add a column for the baseline player's PPG or Total Points
    statistics_by_position_rank[['baseline_' + statistic_for_vorp_calculation, 'baseline_pos_rank']] = statistics_by_position_rank.groupby('pos', group_keys=False) \
                                        .apply(find_baseline_player, **parameters_for_find_baseline_player)[['baseline_' + statistic_for_vorp_calculation, 'baseline_pos_rank']]

    # Determine VORP using the baseline player's PPG or Total Points
    statistics_by_position_rank['vorp'] = (statistics_by_position_rank[statistic_for_vorp_calculation] - statistics_by_position_rank['baseline_' + statistic_for_vorp_calculation]).clip(lower=0)
    
    return statistics_by_position_rank.reset_index()

# Function to find the baseline players
def find_baseline_player(group, baseline_player_strategy, statistic_for_vorp_calculation):
    
    if baseline_player_strategy == 'First $1 Player':
        # Find the first player with a actual_pos_rank_avg_bid_amt < 1
        #TODO: Less than or equal?
        mask = group['actual_pos_rank_avg_bid_amt'] <= 1
        if mask.any():
            baseline = group.loc[mask, statistic_for_vorp_calculation].iloc[0]
            pos_rank = group.loc[mask, statistic_for_vorp_calculation].index[0][1]
        else: # Handle case where no actual_pos_rank_avg_bid_amt < 1
            baseline = 0  
            pos_rank = 1
        group['baseline_' + statistic_for_vorp_calculation] = baseline
        group['baseline_pos_rank'] = pos_rank

    return group


def generate_projected_position_rank_features(input_player_features, included_past_seasons):

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

def generate_player_draft_score(draft_features:pd.DataFrame):
    # Determine the difference between the predicted value of the player and the true value of the player to determine which players are overvalued by the league.
    draft_features['auction_value_difference'] = draft_features['expected_auction_value'] - draft_features['true_auction_value']
    
    # Normalize the auction_value_difference among a position group to compare the players on an even scale.
    draft_features['normalized_auction_value_difference'] = draft_features.groupby('pos')['auction_value_difference'].transform(min_max_scale, invert_min_and_max=True)
    
    # Determine the percent dropoff in VORP to determine relative value to the next best available player
    # Sort by position and descending VORP within each position
    draft_features = draft_features.sort_values(['pos', 'vorp'], ascending=[True, False])

    # Shift VORP within each position group to get the next highest VORP
    draft_features['next_highest_vorp'] = draft_features.groupby('pos')['vorp'].shift(-1)
    draft_features['next_highest_vorp'] = draft_features['next_highest_vorp'].fillna(0)

    # Calculate the % difference between each player's VORP and the next highest in the same position
    draft_features['vorp_pct_difference'] = (
        (draft_features['vorp'] - draft_features['next_highest_vorp']) / draft_features['vorp'].abs()
    )
    draft_features['vorp_pct_difference'] = draft_features['vorp_pct_difference'].fillna(1)

    # Create a final score that weighs both the value of the player and the scarcity of the position via vorp_pct_difference
    draft_features['draft_score'] = draft_features['normalized_auction_value_difference'] * draft_features['vorp_pct_difference']

    draft_features['normalized_draft_score'] = draft_features['draft_score'].transform(min_max_scale)
    
    return draft_features

# TODO: NEW FEATURE - REAL-TIME UPDATES: Might need to move this to a utility function run in main().  
# The results would be passed into other functions and originally calculated values would be updated based on real-time draft updates.
def count_of_positions_for_auction_dollars(league_size, team_position_counts, latest_player_projections):

    qb_count_starter = qb_count_dedicated = team_position_counts['QB'][0] * league_size
    rb_count_starter = rb_count_dedicated = team_position_counts['RB'][0] * league_size
    wr_count_starter = wr_count_dedicated = team_position_counts['WR'][0] * league_size
    te_count_starter = te_count_dedicated = team_position_counts['TE'][0] * league_size
    dst_count_starter = dst_count_dedicated = team_position_counts['DST'][0] * league_size
    k_count_starter = k_count_dedicated = team_position_counts['K'][0] * league_size
    bench_count = team_position_counts['BE'][0] * league_size

    rbwr_flex_count = [team_position_counts['RBWR'][0] * league_size] 
    rbwrte_flex_count = [team_position_counts['RBWRTE'][0] * league_size] 
    wrte_flex_count = [team_position_counts['WRTE'][0] * league_size] 
    super_flex_count = [team_position_counts['OP'][0] * league_size] 

    qb_flexes = [super_flex_count]
    rb_flexes = [super_flex_count, rbwr_flex_count, rbwrte_flex_count]
    wr_flexes = [super_flex_count, rbwr_flex_count, rbwrte_flex_count, wrte_flex_count]
    te_flexes = [super_flex_count, rbwrte_flex_count, rbwrte_flex_count, wrte_flex_count]
    dst_flexes = []
    k_flexes = []

    latest_player_projections = latest_player_projections.sort_values(by='ecr_avg')
    i = 0

    while any(num > 0 for num in [qb_count_dedicated, rb_count_dedicated, wr_count_dedicated, te_count_dedicated, 
                                  rbwr_flex_count[0], rbwrte_flex_count[0], wrte_flex_count[0], super_flex_count[0]]):
        if latest_player_projections.iloc[i]['pos'] == 'QB': 
            if qb_count_dedicated > 0:
                qb_count_dedicated -= 1
            elif max(qb_flexes)[0] > 0:
                qb_count_starter += 1
                max_index = qb_flexes.index(max(qb_flexes))  # Find index of the maximum value
                qb_flexes[max_index][0] -= 1
        elif latest_player_projections.iloc[i]['pos'] == 'RB': 
            if rb_count_dedicated > 0:
                rb_count_dedicated -= 1
            elif max(rb_flexes)[0] > 0:
                rb_count_starter += 1
                max_index = rb_flexes.index(max(rb_flexes))  # Find index of the maximum value
                rb_flexes[max_index][0] -= 1
        elif latest_player_projections.iloc[i]['pos'] == 'WR': 
            if wr_count_dedicated > 0:
                wr_count_dedicated -= 1
            elif max(wr_flexes)[0] > 0:
                wr_count_starter += 1
                max_index = wr_flexes.index(max(wr_flexes))  # Find index of the maximum value
                wr_flexes[max_index][0] -= 1
        elif latest_player_projections.iloc[i]['pos'] == 'TE': 
            if te_count_dedicated > 0:
                te_count_dedicated -= 1
            elif max(te_flexes)[0] > 0:
                te_count_starter += 1
                max_index = te_flexes.index(max(te_flexes))  # Find index of the maximum value
                te_flexes[max_index][0] -= 1
        elif latest_player_projections.iloc[i]['pos'] == 'DST': 
            if dst_count_dedicated > 0:
                dst_count_dedicated -= 1
            elif max(dst_flexes)[0] > 0:
                dst_count_starter += 1
                max_index = dst_count_starter.index(max(dst_count_starter))  # Find index of the maximum value
                dst_count_starter[max_index][0] -= 1
        elif latest_player_projections.iloc[i]['pos'] == 'K': 
            if k_count_dedicated > 0:
                k_count_dedicated -= 1
            elif max(k_flexes)[0] > 0:
                k_count_starter += 1
                max_index = k_count_starter.index(max(k_count_starter))  # Find index of the maximum value
                k_count_starter[max_index][0] -= 1
        
        i += 1
    
    starter_counts = pd.DataFrame([qb_count_starter, rb_count_starter, wr_count_starter, te_count_starter,
                                        dst_count_dedicated, k_count_starter], 
                                    index=['QB', 'RB', 'WR', 'TE', 'DST', 'K'], columns=['count'])

    qb_bench = 0
    rb_bench = 0
    wr_bench = 0
    te_bench = 0
    dst_bench = 0
    k_bench = 0

    # Calculate make up of league wide benches based on projections
    while bench_count > 0:
        # A try is used here to catch the situation where all projections are counted, but there are still bench spots.
        try:
            if latest_player_projections.iloc[i]['pos'] == 'QB': 
                qb_bench += 1
            elif latest_player_projections.iloc[i]['pos'] == 'RB': 
                rb_bench += 1
            elif latest_player_projections.iloc[i]['pos'] == 'WR': 
                wr_bench += 1
            elif latest_player_projections.iloc[i]['pos'] == 'TE': 
                te_bench += 1
            elif latest_player_projections.iloc[i]['pos'] == 'DST': 
                i += 1
                continue
            elif latest_player_projections.iloc[i]['pos'] == 'K': 
                i += 1
                continue
        except:
            logger.info('Roster and bench counts exceeded available player projections.  Returning truncated bench count results.')
            break
        i += 1
        bench_count -= 1

    bench_counts = pd.DataFrame([qb_bench, rb_bench, wr_bench, te_bench,
                                        dst_bench, k_bench], 
                                    index=['QB', 'RB', 'WR', 'TE', 'DST', 'K'], columns=['count'])
    
    if sum(bench_counts['count']) < team_position_counts['BE'][0] * league_size:
        bench_counts.loc['RB', 'count'] = bench_counts.loc['RB', 'count'] + (team_position_counts['BE'][0] * league_size - sum(bench_counts['count']))
   
    return  starter_counts, bench_counts