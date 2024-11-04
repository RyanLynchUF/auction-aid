import os
import json
from datetime import datetime
from typing import List
import pandas as pd
from data.s3Interface import S3Uploader
from data.fantasypros.scrape_fp import FPScraper
import config.settings as settings

from utils.logger import get_logger

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME_LEAGUE=os.getenv('AWS_S3_BUCKET_NAME_LEAGUE')

uploader = S3Uploader(aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        bucket_name=AWS_S3_BUCKET_NAME_LEAGUE)

logger = get_logger(__name__)

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR
SCORING_FORMATS = ['standard', 'ppr', 'half-ppr']
LEAGUE_SIZES = ['10', '12', '14']
    
def scrape_daily_fantasypros_projections():
    """
    Daily process to scrape the latest projection data from FantasyPros.
    """
    for scoring_format in SCORING_FORMATS:
        daily_projections = {}
        date_string = datetime.now().strftime('%Y%m%d')

        ecr_scraper = FPScraper('ecr', scoring_format)
        daily_projections['ecr'] = ecr_scraper.scrape_ecr()

        vbd = {}
        for league_size in LEAGUE_SIZES:
            vbd_scraper = FPScraper('vbd', scoring_format, league_size=league_size)
            vbd[league_size] = vbd_scraper.scrape_vbd()

        daily_projections['vbd'] = vbd

        adp_scraper = FPScraper('adp', scoring_format)
        daily_projections['adp'] = adp_scraper.scrape_adp()

        daily_projections_json = json.dumps(daily_projections)
        object_name = f"fp-metrics-{scoring_format}-{date_string}.json"

        uploader.upload_json_to_s3(daily_projections_json, f'fp/daily-metrics/{object_name}')

#TODO: Refactor this process to allow for only downloading a single year without overwriting the "asoofYYYY" so we don't overwrite it 
def scrape_historical_fantasypros_projections(years:List[int]):
    """
    Downloads histroical FantasyPros projections.
    
    Parameters:
    -----------
    years : List[int]
        List of years to download
    """

    scoring_formats = ['standard', 'ppr', 'half-ppr']
    league_sizes = ['10', '12', '14']

    all_historical_projections = {}

    for scoring_format in scoring_formats:
        historical_projections = []
        for year in years:
            year = str(year)
            yearly_projections = {}

            vbd = {}
            for league_size in league_sizes:
                vbd_scraper = FPScraper('vbd', scoring_format, league_size=league_size, year=year)
                vbd[league_size] = vbd_scraper.scrape_vbd()

            yearly_projections['vbd'] = vbd

            adp_scraper = FPScraper('adp', scoring_format, year=year)
            yearly_projections['adp'] = adp_scraper.scrape_adp()

            historical_projections.append({year: yearly_projections})
            all_historical_projections[scoring_format] = historical_projections

            yearly_projections_json = json.dumps(yearly_projections)
            object_name = f"fp-metrics-{scoring_format}-{year}.json"
            uploader.upload_json_to_s3(yearly_projections_json, f'fp/historical-metrics/{object_name}')

        all_historical_projections[scoring_format] = historical_projections

    for scoring_format, years_data in all_historical_projections.items():
        all_players_data = []

        for year_data in years_data:
            year = list(year_data.keys())[0]
            vbd_data = year_data[year].get('vbd', {})
            adp_data = year_data[year].get('adp', {})

            for league_size, players in vbd_data.items():
                for player_name, stats in players.items():
                    player_data = {
                        'player_name': player_name,
                        'pos': adp_data.get(player_name, {}).get('pos', None),
                        'projected_pos_rank': adp_data.get(player_name, {}).get('projected_pos_rank', None),
                        'year': year,
                        'league_size': league_size,
                        'vbd': stats.get('vbd', None),
                        'vorp': stats.get('vorp', None),
                        'vols': stats.get('vols', None),
                        'adp_avg': adp_data.get(player_name, {}).get('adp_avg', None)
                        
                    }
                    all_players_data.append(player_data)


        all_players_data_json = json.dumps(all_players_data)
        object_name = f"fp-metrics-{scoring_format}-allyearsasof-{CURR_LEAGUE_YR}.json"

        uploader.upload_json_to_s3(all_players_data_json, f'fp/historical-metrics/{object_name}')