import os
import re
import json

from data.s3Interface import S3Reader
from data.fantasypros.scrape_fp import FPScraper
import config.settings as settings
from utils.logger import get_logger

logger = get_logger(__name__)

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME_LEAGUE=os.getenv('AWS_S3_BUCKET_NAME_LEAGUE')


reader = S3Reader(aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        bucket_name=AWS_S3_BUCKET_NAME_LEAGUE)

#TODO: Make configurable to read from S3 or AppData

def get_fantasypros_projections_daily(scoring_format:str='standard'):
    """
    Read latest FantasyPros projections from either S3 or local storage.
    
    Parameters:
    -----------
    scoring_format : str
        The league scoring format: ppr, half-ppr, or standard (no points per reception)
    s3 : bool
        Boolean value to indicate if teh data should be read from s3
    """

    response = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE) 
    matching_objects = [obj for obj in response['Contents'] if re.search(rf'/daily-metrics/fp-metrics-{scoring_format}-', obj['Key'])]
    most_recent_object = sorted(matching_objects, key=lambda obj: obj['LastModified'], reverse=True)[0]
    obj_response = reader.read_from_s3(object_key=most_recent_object['Key'])
    json_string = obj_response['Body'].read().decode('utf-8')
    fp_metrics = json.loads(json_string)
    return fp_metrics

def get_past_fantasypros_projections(scoring_format:str='standard'):
    """
    Read latest FantasyPros projections from either S3 or local storage.
    
    Parameters:
    -----------
    scoring_format : str
        The league scoring format: ppr, half-ppr, or standard (no points per reception)
    s3 : bool
        Boolean value to indicate if the data should be read from s3
    """

    response = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE)
    
    matching_objects = [obj for obj in response['Contents'] if re.search(rf'/historical-metrics/fp-metrics-{scoring_format}-allyearsasof-{CURR_LEAGUE_YR}.json', obj['Key'])]

    for obj in matching_objects:

        obj_response = reader.read_from_s3(object_key=obj['Key'])

        json_string = obj_response['Body'].read().decode('utf-8')
        fp_metrics = json.loads(json_string)
        #TODO: add error handling to download latest metrics if this doesnt work
        if obj['Key'][-9:-5] == str(CURR_LEAGUE_YR):
            fp_metrics_history = fp_metrics

    return fp_metrics_history


def scrape_fantasypros_auction_values(scoring_format:str='standard', year:str=CURR_LEAGUE_YR, league_size:str="12", team_balance:int=200,
                                        team_composition:dict= {'QB':1, 'RB':2, 'WR':3, 'TE':1, 'DST':1, 'K':1, 'BN':6,
                                                                'WR/RB':0, 'WR/RB/TE':0, 'WR/TE':0, 'RB/TE':0, 'QB/WR/RB/TE':0, 'DL':0,
                                                                'LB':0, 'DB':0, 'IDP':0}):
    """
    Scrape the latest FantasyPros auction valaues from either S3 or local storage.
    
    Parameters:
    -----------
    scoring_format : str
        The league scoring format: ppr, half-ppr, or standard (no points per reception)
    year: str
        The year of the league
    league_size : str
        The total number of teams in the league
    team_balance : int
        The starting number of auction dollars for a team during the auction draft
    team_composition : dict
        The positions and counts that makeup a team roster for the league
    """

    auction_scraper = FPScraper('auction', scoring_format=scoring_format, league_size=league_size, 
                                team_balance=team_balance, team_composition=team_composition)
    
    auction_data = auction_scraper.scrape_auction_values()

    return auction_data