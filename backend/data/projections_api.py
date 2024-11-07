import os
import glob
import re
import json
from datetime import datetime
from typing import List

from data.data_pipline import scrape_daily_fantasypros_projections, scrape_historical_fantasypros_projections
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

def get_fantasypros_projections_daily(scoring_format:str='standard', s3:bool=settings.S3):
    """
    Read latest FantasyPros projections from either S3 or local storage.
    
    Parameters:
    -----------
    scoring_format : str
        The league scoring format: ppr, half-ppr, or standard (no points per reception)
    s3 : bool
        Boolean value to indicate if the data should be read from s3
    """
    if s3:
        response = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE) 
        matching_objects = [obj for obj in response['Contents'] if re.search(rf'/daily-projections/fp-projections-{scoring_format}-', obj['Key'])]
        if not matching_objects:
            scrape_daily_fantasypros_projections(scoring_formats=[scoring_format])
            response = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE) 
        matching_objects = [obj for obj in response['Contents'] if re.search(rf'/daily-projections/fp-projections-{scoring_format}-', obj['Key'])]
        most_recent_object = sorted(matching_objects, key=lambda obj: obj['LastModified'], reverse=True)[0]
        obj_response = reader.read_from_s3(object_key=most_recent_object['Key'])
        json_string = obj_response['Body'].read().decode('utf-8')
        daily_player_projections = json.loads(json_string)
    else:
        date_string = datetime.now().strftime('%Y%m%d')

        appdata_path = os.path.join(os.getcwd(), "backend/AppData/daily-projections/")

        # Download latest projections 
        appdata_matching_league_objects = [os.path.join(root, file)
                                            for root, dirs, files in os.walk(appdata_path)
                                            for file in files
                                            if re.search(rf'fp-projections-{scoring_format}-{date_string}', file)]
        
        if not appdata_matching_league_objects:
            # Delete old daily projections
            files = glob.glob(os.path.join(appdata_path, '*'))
            for file in files:
                try:
                    os.remove(file)
                    logger.info(f"Deleted old projection data: {file}")
                except Exception as e:
                    logger.error(f"Failed to delete {file}. Reason: {e}")
            scrape_daily_fantasypros_projections(scoring_formats=[scoring_format])
            appdata_matching_league_objects = [os.path.join(root, file)
                                            for root, dirs, files in os.walk(appdata_path)
                                            for file in files
                                            if re.search(rf'fp-projections-{scoring_format}-{date_string}', file)]
    

        for file_path in appdata_matching_league_objects:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r') as f:
                    daily_player_projections = json.loads(f.read())
                    logger.info(f'Successfully read {file_name}')
            except json.JSONDecodeError:
                logger.info(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                logger.info(f"Error reading file {file_name}: {e}")

    return daily_player_projections
def get_past_fantasypros_projections(years:List[int], scoring_format:str='standard', s3:bool=settings.S3):
    """
    Read latest FantasyPros projections from either S3 or local storage.
    
    Parameters:
    -----------
    scoring_format : str
        The league scoring format: ppr, half-ppr, or standard (no points per reception)
    """

    if s3:
        response = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE)
        matching_objects = [obj for obj in response['Contents'] if re.search(rf'/historical-projections/fp-projections-{scoring_format}-allyearsasof-{CURR_LEAGUE_YR}.json', obj['Key'])]
        if not matching_objects:
            logger.info(f"Historical projection data not yet downloaded.  Scraping historical projection data...")
            scrape_historical_fantasypros_projections(years=years, scoring_formats=[scoring_format], s3=settings.S3)
            response = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE)
            matching_objects = [obj for obj in response['Contents'] if re.search(rf'/historical-projections/fp-projections-{scoring_format}-allyearsasof-{CURR_LEAGUE_YR}.json', obj['Key'])]

        for obj in matching_objects:
            obj_response = reader.read_from_s3(object_key=obj['Key'])
            json_string = obj_response['Body'].read().decode('utf-8')
            past_player_projections = json.loads(json_string)
            if obj['Key'][-9:-5] == str(CURR_LEAGUE_YR):
                latest_past_player_projections = past_player_projections
    else:
        appdata_path = os.path.join(os.getcwd(), "backend/AppData/historical-projections/")
        appdata_matching_league_objects = [os.path.join(root, file)
                                            for root, dirs, files in os.walk(appdata_path)
                                            for file in files
                                            if re.search(rf'fp-projections-{scoring_format}-allyearsasof-{CURR_LEAGUE_YR}.json', file)]
        
        if not appdata_matching_league_objects:
            logger.info(f"Historical projection data not yet downloaded.  Scraping historical projection data...")
            # Download historical projections 
            scrape_historical_fantasypros_projections(years=years, scoring_formats=[scoring_format], s3=settings.S3)
            appdata_matching_league_objects = [os.path.join(root, file)
                                            for root, dirs, files in os.walk(appdata_path)
                                            for file in files
                                            if re.search(rf'fp-projections-{scoring_format}-allyearsasof-{CURR_LEAGUE_YR}.json', file)]

        for file_path in appdata_matching_league_objects:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r') as f:
                    latest_past_player_projections = json.loads(f.read())
            except json.JSONDecodeError:
                logger.info(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                logger.info(f"Error reading file {file_name}: {e}")

    return latest_past_player_projections


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