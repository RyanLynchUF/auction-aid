from espn_api.football import League #https://github.com/cwendt94/espn-api

from typing import List, Dict
import os
import re
import json
import jsonpickle

from data.s3Interface import S3Uploader, S3Reader
import config.settings as settings
from utils.logger import get_logger

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR
S3 = settings.S3

from dotenv import load_dotenv
load_dotenv(override=True)

AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME_LEAGUE=os.getenv('AWS_S3_BUCKET_NAME_LEAGUE')

uploader = S3Uploader(aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        bucket_name=AWS_S3_BUCKET_NAME_LEAGUE)

reader = S3Reader(aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        bucket_name=AWS_S3_BUCKET_NAME_LEAGUE)

logger = get_logger(__name__)

def get_league_data(league_id:int, year:int, espn_s2:str = None, swid:str = None):
    """
    Get fantasy football league information using the ESPN API.
    
    Parameters:
    -----------
    league_id : int
        Unique ID assigned by ESPN to the league
    year : int
        The year of the league which will be pulled from ESPN.
    espn_s2 : str
        Unique ID assigned to private leagues on ESPN
    swid : str
        Unique ID assigned to private leagues on ESPN
    """
    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
        logger.info(f'Successfully retrieved league {league_id} for year {year} from ESPN.')
        return league
    except Exception as error:
        logger.error(f'Failed to download league: {error}')

def upload_league(league:League, s3:bool=settings.S3):
    """
    Persist data for league to preferred location.
    
    Parameters:
    -----------
    league : League object
        Custom Python object that contains data for the league
    s3 : bool
        Boolean value to indicate if the data will be persisted to an S3 location.  
        If not, data is persisted to local storage in the AppData folder.
    """

    league_id = league.league_id
    year = league.year
    league_json = jsonpickle.encode(league, max_depth=30,
                                 separators=(',', ': '))
    
    object_name = f"espn-{league_id}-{year}.json"
    if s3:
        uploader.upload_json_to_s3(league_json, f'espn/league/{object_name}')
    else:
        appdata_path = os.path.join(os.getcwd(), "AppData/league")
        file_path = os.path.join(appdata_path, object_name)
        with open(file_path, 'w') as f:
            json.dump(league_json, f, indent=4)
    
def post_leagues(league_id:int=None, years=List[int], espn_s2:str=None, swid:str=None):
    """
    Post multiple years of fantasy football league information using the ESPN API and persist to storage.
    
    Parameters:
    -----------
    league_id : int
        Unique ID assigned by ESPN to the league
    years : List(int)
        The years of the league which will be pulled from ESPN.
    espn_s2 : str
        Unique ID assigned to private leagues on ESPN
    swid : str
        Unique ID assigned to private leagues on ESPN
    """
    leagues = {}
    for year in years:
        try:
            leagues[year] = get_league_data(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
            upload_league(leagues[year])
        except Exception as error:
            logger.error(f'Failed to download past leagues: {error}')

def get_past_leagues(league_id:int, s3:bool=settings.S3):
    """
    Read all past league data available in storage for a given league.
    
    Parameters:
    -----------
    league_id : int
        Unique ID assigned by ESPN to the league
    s3 : bool
        Boolean value to indicate if the data will be read from an S3 location.  
        If not, data is read from local storage in the AppData folder.
    """

    leagues = {}

    if s3:
        s3_league_objects = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE)
        try:
            s3_matching_league_objects = [obj['Key'] for obj in s3_league_objects['Contents'] if re.search(fr'^espn\/league\/espn-{league_id}-\d+\.json$', obj['Key'])]
        except:
            logger.info("No past leagues currently in S3.")
            return None
        for object_key in s3_matching_league_objects:
            obj_response = reader.read_from_s3(object_key=object_key)
            obj = jsonpickle.decode(obj_response['Body'].read())
            if object_key[-9:-5] != str(CURR_LEAGUE_YR):
                leagues[object_key[-9:-5]] = obj
    else:
        appdata_path = os.path.join(os.getcwd(), "AppData/league")
        appdata_matching_league_objects = [os.path.join(root, file)
                                            for root, dirs, files in os.walk(appdata_path)
                                            for file in files
                                            if re.search(fr'^espn-{league_id}-\d+\.json$', file)]
        if not appdata_matching_league_objects:
            logger.info("No past leagues currently in AppData folder.")
            return None
        
        for file_path in appdata_matching_league_objects:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r') as f:
                    leagues[file_name[-9:-5]] = jsonpickle.decode(json.load(f))
                    logger.info(f'Successfully read {file_name}')   
            except json.JSONDecodeError:
                logger.info(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                logger.info(f"Error reading file {file_name}: {e}")

    return leagues
        
def aggregate_and_post_player_stats(past_leagues:Dict, s3=settings.S3):
    """
    Use data from past leagues to aggrgate player stats data. Then, upload the data to preferred location.
    
    Parameters:
    -----------
    past_leagues : Dict
        Dictionary contain keys of years and each value is the League object for that year.
    s3 : bool
        Boolean value to indicate if the data will be persisted to an S3 location.  
        If not, data is persisted to local storage in the AppData folder.
    """
    
    league_id = past_leagues[list(past_leagues.keys())[0]].league_id
    player_history_stats = {}

    for year, league in past_leagues.items():
        player_stats = []
        player_ids = [int(k) for k, v in league.player_map.items() if k.isdigit()]
        # Define the batch size
        batch_size = 500
        # Loop through the list in batches
        for i in range(0, len(player_ids), batch_size):
            # Get the current batch of player IDs
            batch = player_ids[i:i+batch_size]
            
            # Make the API call with the current batch
            players = league.player_info(playerId=batch)
            
            for player in players:  
                if any(var == [] for var in (player.position)): continue
                player_stat = {
                    'player_name': player.name,
                    'team': player.proTeam,
                    'ppg': float(player.avg_points),
                    'pos': player.position,
                    'actual_pos_rank': int(player.posRank) if player.posRank else None,
                    'total_points': float(player.total_points)
                    }
                player_stats.append(player_stat)


        player_history_stats[year] = player_stats

    object_name = f"espn-{league_id}-playerstats.json"
    if s3:
        player_history_stats_json = json.dumps(dict(player_history_stats), indent=4)
        uploader.upload_json_to_s3(player_history_stats_json, f'espn/league/playerstats/{object_name}')
    else:
        appdata_path = os.path.join(os.getcwd(), "AppData/league")
        file_path = os.path.join(appdata_path, object_name)
        with open(file_path, 'w') as f:
            json.dump(player_history_stats, f, indent=4)
  
def get_league_player_stats(league_id:int, s3=settings.S3):
    """
    Read all player stats data available in storage for a given league.
    
    Parameters:
    -----------
    league_id : int
        Unique ID assigned by ESPN to the league
    s3 : bool
        Boolean value to indicate if the data will be read from an S3 location.  
        If not, data is read from local storage in the AppData folder.
    """

    player_history_stats = {}

    if s3:
        s3_league_objects = reader.s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME_LEAGUE)
        s3_matching_league_objects = [obj['Key'] for obj in s3_league_objects['Contents'] if re.search(fr'^espn\/league\/playerstats\/espn-{league_id}-playerstats.json$', obj['Key'])]
        if not s3_matching_league_objects:
            logger.info("No player stats currently in S3.  Downloading player stats from fantasy platform.")
            return None
        for object_key in s3_matching_league_objects:
            obj_response = reader.read_from_s3(object_key=object_key)
            player_history_stats = json.loads(obj_response['Body'].read())
    else:
        appdata_path = os.path.join(os.getcwd(), "AppData/league")
        appdata_matching_league_objects = [os.path.join(root, file)
                                            for root, dirs, files in os.walk(appdata_path)
                                            for file in files
                                            if re.search(fr'^espn-{league_id}-playerstats.json$', file)]
        if not appdata_matching_league_objects:
            logger.info("No player stats currently in AppData folder.  Downloading player stats from fantasy platform.")
            return None
        
        for file_path in appdata_matching_league_objects:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r') as f:
                    player_history_stats = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                logger.error(f"Error reading file {file_name}: {e}")

    return player_history_stats