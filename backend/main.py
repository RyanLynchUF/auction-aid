from fastapi import Cookie, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from functools import reduce
import pandas as pd

from data.espn.league_api import get_league_data, get_past_leagues, post_leagues, aggregate_and_post_player_stats, get_league_player_stats
from data.metrics_api import get_fantasypros_projections_daily, get_past_fantasypros_projections, scrape_fantasypros_auction_values
from data.preprocessing import transform_latest_player_projections, transform_past_expert_player_projections, transform_past_auction_values, transform_past_player_stats
import features as features
import models as models

import config.settings as settings
from utils.helper import scale_features_min_max


"""
TODO: 

- After initial Commit:
    - Check what features are actually used in model and make sure there aren't any redundnace training / predictions dfs being calculated
    - Review final response to ensure enough players are sent

- Frontend
    - Address no past leagues use cases
    - Invalid league data info    
    - Information hovers for input fields

- Check if 2021 ADP is back for half-ppr

- Line by Line code step through:
    - Determine any code that can be used for utils
    - Define typing for all functions
    - Make variable names NOT source specific (i.e., fp...)
    - Limit redundnat API calls
    - Improve logging
        - Update all print statements to logger statements
        - Think through error handling of someone setting up their own AWS bucket and update logging
    - Think through minor data validations (duplicates?)
    - Address yucky names
        - S3 buckets - historical metrics
        - redundant key names
    - Add clean comments throughout
    - Add input / output style comments to functions
    - Address warnings in terminal
    - Create instruction and add links
        - Landing Login details links

- Potential Bugs:
    - Does pos_rank needs to be min max?  do we need normal pos_rank column and min_max pos rank column?

    
- Generate README: https://www.reddit.com/r/SideProject/s/Pj6jbnUG7w
"""

"""
TODO (tests): 
- New league sign-in - start to finish
    - S3 storage and AppData storage
        - Check if League objects are working in AppData version with using json package instead of jsonpickle
    - Test while changing S3 bucket name
- New league sign-in - stop and restart at different steps
- New league sign-in - use seasonsToInclude features
- League in its first year, add leagues with <3 year experience warning
- Test on public leagues
- Test with league that is NOT an auction draft
- Cron daily download
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LeagueForm(BaseModel):
    leagueId: str
    swid: str
    espnS2: str

class GenerateAuctionAidForm(BaseModel):
    leagueId: str
    swid: str
    espnS2: str
    scoringFormat: str
    auctionDollarsPerTeam: str
    keepers: str
    projectionsSource: str
    baselinePlayerStrategy: str
    statisticForVorpCalculation: str
    teamDraftStrategy: str
    seasonsToInclude: List[int]

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR

@app.post("/api/league/")
async def get_league(form_data: LeagueForm):
    league_id = form_data.leagueId
    swid = form_data.swid
    espn_s2 = form_data.espnS2

    curr_league = get_league_data(league_id=league_id, year=CURR_LEAGUE_YR, swid=swid, espn_s2=espn_s2)

    response = {
        'league_id': curr_league.league_id,
        'league_name': curr_league.settings.name,
        'season_year': curr_league.year,
        'league_size': curr_league.settings.team_count,
        'previous_seasons': curr_league.previousSeasons
        } 

    return response


@app.post("/api/generate-auction-aid")
async def generate_auction_aid(auction_aid_form_data: GenerateAuctionAidForm):
    #Assign frontend data to variables
    league_id = auction_aid_form_data.leagueId
    swid = auction_aid_form_data.swid
    espn_s2 = auction_aid_form_data.espnS2
    scoring_format = auction_aid_form_data.scoringFormat.lower()
    auction_dollars_per_team = auction_aid_form_data.auctionDollarsPerTeam
    keepers = auction_aid_form_data.keepers # TODO: NEW FEATURE- KEEPERS: Consider keeper values in calculations
    projections_source = auction_aid_form_data.projectionsSource # TODO: NEW FEATURE- PROJECTION SOURCES: Allow users to request from other projection sources
    baseline_player_strategy = auction_aid_form_data.baselinePlayerStrategy
    statistic_for_vorp_calculation = auction_aid_form_data.statisticForVorpCalculation.lower()
    team_draft_strategy = auction_aid_form_data.teamDraftStrategy
    included_past_seasons = auction_aid_form_data.seasonsToInclude
    vorp_configuration = {'baseline_player_strategy': baseline_player_strategy,
        'statistic_for_vorp_calculation': statistic_for_vorp_calculation,
        'team_draft_strategy': team_draft_strategy}
    
    # Define dynamic column names
    actual_pos_rank_column = 'actual_pos_rank_' + statistic_for_vorp_calculation

    # Get current league data and available past league data
    curr_league = get_league_data(league_id=league_id, year=CURR_LEAGUE_YR, swid=swid, espn_s2=espn_s2)
    past_leagues = get_past_leagues(league_id=league_id)
        
    # Load current and past league data
    # If there is no past_leagues, or if the last season is not in past_leagues, refresh past_leagues and player stats
    if past_leagues is None or int(max(past_leagues.keys())) + 1 < CURR_LEAGUE_YR:
        post_leagues(league_id=league_id, years=curr_league.previousSeasons, swid=swid, espn_s2=espn_s2)
        past_leagues = get_past_leagues(league_id)
        aggregate_and_post_player_stats(past_leagues)
        past_player_stats = get_league_player_stats(league_id)
    else:
        past_player_stats = get_league_player_stats(league_id)

    # Load projection data 
    league_size = curr_league.settings.team_count
    if projections_source == 'FantasyPros':
        expert_auction_valuation = scrape_fantasypros_auction_values(scoring_format=scoring_format, year=CURR_LEAGUE_YR, 
                                                        league_size=league_size, 
                                                        team_balance=auction_dollars_per_team) # TODO: Add team decomposition if we start using this data
        player_projections = get_fantasypros_projections_daily(scoring_format=scoring_format)
        past_player_projections = get_past_fantasypros_projections(scoring_format=scoring_format)
    else:
        print(f"Unavailable projection source: {projections_source}")

    # Create dictionary which contains league setting
    league_settings = features.create_league_settings(curr_league)

    # Transform input data into format for further processing and feature engineering
    latest_player_projections = transform_latest_player_projections(league_size, player_projections, expert_auction_valuation)
    past_player_projections = transform_past_expert_player_projections(league_size, past_player_projections, years=included_past_seasons)
    past_player_stats = transform_past_player_stats(past_player_stats, league_settings, years=included_past_seasons)
    past_player_auction_values = transform_past_auction_values(past_leagues, years=included_past_seasons)

    # Combine input data into one DataFrame for feature engineering and analysis
    input_dataframes = [latest_player_projections, past_player_auction_values,
                        past_player_stats, past_player_projections]
    input_player_features = reduce(lambda left, right: left.join(right, on='player_name', how='outer', lsuffix='to_merge1', rsuffix='to_merge2'), input_dataframes)
    # Merge position column from different input sources into one consistent column
    input_player_features['pos'] = input_player_features.filter(like='to_merge').bfill(axis=1).iloc[:, 0]
    input_player_features = input_player_features.drop(columns=input_player_features.filter(like='to_merge').columns)
    input_player_features = input_player_features.set_index('player_name')
    # Create actual_pos_rank column values for DST using projections, since actual data is not available 
    actual_pos_rank_columns = [f'{actual_pos_rank_column}_{year}' for year in included_past_seasons]
    input_player_features.loc[input_player_features['pos'] == 'DST', actual_pos_rank_columns] = \
    input_player_features.loc[input_player_features['pos'] == 'DST', 'projected_pos_rank_2024']
    
    

# Feature Engineering
    # Determine years to use for training data vs. prediction data for upcoming draft
    past_seasons_for_training = included_past_seasons.copy()
    past_seasons_for_training.remove(max(included_past_seasons))
    past_seasons_for_prediction = included_past_seasons.copy()
    
    # Generate features for players based on past league(s) and  projections
    training_player_features = features.generate_player_features(input_player_features, past_seasons_for_training, vorp_configuration=vorp_configuration).reset_index()
    prediction_player_features = features.generate_player_features(input_player_features, past_seasons_for_prediction, vorp_configuration=vorp_configuration).reset_index()

    # Generate features for actual position ranks (e.g., RB1, RB2, etc.) based on past league(s)
    training_actual_position_rank_features = features.generate_actual_position_rank_features(league_settings, input_player_features, latest_player_projections,
                                                                               past_seasons_for_training, auction_dollars_per_team, vorp_configuration).reset_index()
    prediction_actual_position_rank_features = features.generate_actual_position_rank_features(league_settings, input_player_features, latest_player_projections,
                                                                                 past_seasons_for_prediction, auction_dollars_per_team, vorp_configuration).reset_index()
    
    # Generate features for projected position ranks (e.g., RB1, RB2, etc.) based on FP projections
    training_projected_position_rank_features = features.generate_projected_position_rank_features(input_player_features, past_seasons_for_training)
    prediction_projected_position_rank_features = features.generate_projected_position_rank_features(input_player_features, past_seasons_for_prediction)

    #TODO: Make a DF with ALL potentially drafted players as the base


    training_features = pd.merge(training_player_features, training_actual_position_rank_features,
                                                left_on=['pos', 'curr_year_projected_pos_rank'], 
                                                right_on=['pos', actual_pos_rank_column], how='left')
    training_features = pd.merge(training_features, training_projected_position_rank_features,
                                                left_on=['pos', 'curr_year_projected_pos_rank'], 
                                                right_on=['pos', 'projected_pos_rank'], how='left').set_index('player_name')
    
    prediction_features = pd.merge(prediction_player_features, prediction_actual_position_rank_features,
                                                left_on=['pos', 'curr_year_projected_pos_rank'], 
                                                right_on=['pos', actual_pos_rank_column], how='left')
    prediction_features = pd.merge(prediction_features, prediction_projected_position_rank_features,
                                                left_on=['pos', 'curr_year_projected_pos_rank'], 
                                                right_on=['pos', 'projected_pos_rank'], how='left').set_index('player_name')
    
    min_max_scaling = False
    if min_max_scaling:
        features_to_scale = [col for col in training_features.columns if col not in 
                             ['player_name', 'pos', 'curr_year_bid_amt']]
        training_features[features_to_scale] = scale_features_min_max(training_features, features_to_scale)
        prediction_features[features_to_scale] = scale_features_min_max(prediction_features, features_to_scale)
        
    # Filter out players unlikely to get drafted, but make sure to keep DST players that may not show up in projections
    training_features = training_features[(training_features['curr_year_projected_pos_rank'].notna()) | (training_features['pos'] == 'DST')]
    prediction_features = prediction_features[(prediction_features['curr_year_projected_pos_rank'].notna()) | (prediction_features['pos'] == 'DST')]

    # Replace nan values with 0
    columns_for_nan_replacement = ['curr_year_bid_amt', 'prev_year_bid_amt', 'curr_year_vbd', 'actual_pos_rank_avg_bid_amt',
                    'vorp', 'true_auction_value', 'avg_ppg_from_league_history',  'avg_bid_amt_from_league_history',
                    'avg_ratio_of_vbd_to_auction_value_from_league_history', 'projected_pos_rank_avg_bid_amt']
    training_features.loc[:,columns_for_nan_replacement] = training_features.loc[:,columns_for_nan_replacement].fillna(0)
    prediction_features.loc[:,columns_for_nan_replacement] = prediction_features.loc[:,columns_for_nan_replacement].fillna(0)

    training_features.loc[:,'curr_year_projected_pos_rank'] = training_features.loc[:,'curr_year_projected_pos_rank'].fillna(training_features.groupby('pos')['curr_year_projected_pos_rank'].transform('max'))
    prediction_features.loc[:,'curr_year_projected_pos_rank'] = prediction_features.loc[:,'curr_year_projected_pos_rank'].fillna(prediction_features.groupby('pos')['curr_year_projected_pos_rank'].transform('max'))

    # Remove unneccesary columns 
    training_features = training_features.drop(['avg_bid_amt_from_league_history',
                                                'true_auction_value', actual_pos_rank_column, 'actual_pos_rank_avg_bid_amt', 'vorp',
                                                'projected_pos_rank',
                                                'team_ARI', 'team_ATL',
                                                'team_BAL', 'team_BUF', 'team_CAR', 'team_CHI', 'team_CIN', 'team_CLE',
                                                'team_DAL', 'team_DEN', 'team_DET', 'team_GB', 'team_HOU', 'team_IND',
                                                'team_JAC', 'team_KC', 'team_LAC', 'team_LAR', 'team_LV', 'team_MIA',
                                                'team_MIN', 'team_NE', 'team_NO', 'team_NYG', 'team_NYJ', 'team_PHI',
                                                'team_PIT', 'team_SEA', 'team_SF', 'team_TB', 'team_TEN', 'team_WAS'], axis=1)
    prediction_features = prediction_features.drop(['curr_year_bid_amt',
                                                'avg_bid_amt_from_league_history',
                                                'true_auction_value', actual_pos_rank_column, 'actual_pos_rank_avg_bid_amt', 'vorp',
                                                'projected_pos_rank',
                                                'team_ARI', 'team_ATL',
                                                'team_BAL', 'team_BUF', 'team_CAR', 'team_CHI', 'team_CIN', 'team_CLE',
                                                'team_DAL', 'team_DEN', 'team_DET', 'team_GB', 'team_HOU', 'team_IND',
                                                'team_JAC', 'team_KC', 'team_LAC', 'team_LAR', 'team_LV', 'team_MIA',
                                                'team_MIN', 'team_NE', 'team_NO', 'team_NYG', 'team_NYJ', 'team_PHI',
                                                'team_PIT', 'team_SEA', 'team_SF', 'team_TB', 'team_TEN', 'team_WAS'], axis=1)

    # Models
    model, multicollinearity_columns_to_remove = models.train_model(training_features, model_type='random_forest', mode='testing')
    
    # Drop identifed columns with multi-collinearity
    training_features = training_features.drop(multicollinearity_columns_to_remove, axis=1)  
    prediction_features = prediction_features.drop(multicollinearity_columns_to_remove, axis=1)  

    # Predict auction values for upcoming draft  
    auction_value_predictions = models.predict_auction_value(model, prediction_features)

    # Select the final set of fields to include in draft_features and display on User Interface
    draft_features = pd.merge(auction_value_predictions.reset_index(), input_player_features[['team', 'ecr_avg', 'pos']].reset_index(), 
                        how='left', on='player_name')
    draft_features = pd.merge(draft_features, prediction_actual_position_rank_features, how='left',
                        left_on=['pos', 'curr_year_projected_pos_rank'], right_on=['pos', actual_pos_rank_column])
    draft_features = draft_features[['ecr_avg', 'player_name', 'team', 
                'pos', 'curr_year_projected_pos_rank', 'vorp',
                'prev_year_bid_amt', 'projected_pos_rank_avg_bid_amt',
                'expected_auction_value', 'true_auction_value']]
    draft_features['true_auction_value'] = draft_features['true_auction_value'].fillna(1)
    draft_features['vorp'] = draft_features['vorp'].fillna(0)

    # Create draft score to help user identify draft targets
    draft_features = features.generate_player_draft_score(draft_features)

    draft_features = draft_features.drop_duplicates(subset=['player_name'])
    sorted_response = draft_features.sort_values(by='ecr_avg')

    sorted_response = sorted_response.to_dict(orient='records')

    return JSONResponse(content=sorted_response)

    """
    Not Used - Alternative approach which creates a model for each position group.
    model_performance, overall_performance = models.train_model_by_position(training_features, model_type='random_forest')
    predictions = models.predict_by_position(prediction_features, models=model_performance)
    """

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
