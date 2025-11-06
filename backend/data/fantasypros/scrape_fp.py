import requests
from bs4 import BeautifulSoup
import re
import json

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}

DOMAIN_PROJECTIONS = "https://www.fantasypros.com/nfl/"
DOMAIN_AUCTION = "https://draftwizard.fantasypros.com/editor/createFromProjections.jsp?"

URL_SCORING_MAP = {
    "ecr": {
        "standard":"rankings/consensus-cheatsheets.php",
        "ppr":"rankings/ppr-cheatsheets.php",
        "half-ppr":"rankings/half-point-ppr-cheatsheets.php"
    },
    "vbd": {
        "standard":"rankings/vbd.php",
        "ppr":"rankings/ppr-overall.php",
        "half-ppr":"rankings/half-ppr-vbd.php"
    },
    "adp": {
        "standard":"adp/overall.php",
        "ppr":"adp/ppr-overall.php",
        "half-ppr":"adp/half-point-ppr-overall.php"
    },
    "auction": {
         "standard":"STD",
         "ppr":"PPR",
         "half-ppr":"HALF"
    }
}

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR

class FPScraper:
    
    def __init__(self, projection:str, scoring_format:str, year:str=CURR_LEAGUE_YR, league_size:str="12", team_balance:int=200,
                                        team_composition:dict={"QB":"1", "RB":"2", "WR":"3", "TE":"1", "DST":"1", "K":"1", "BN":"6",
                                                               "WR/RB":"0", "WR/RB/TE":"0", "WR/TE":"0", "RB/TE":"0", "QB/WR/RB/TE":"0", "DL":"0",
                                                               "LB":"0", "DB":"0", "IDP":"0"}):
        
        self.projection = projection
        self.scoring_format = scoring_format
        self.year = str(year)
        self.league_size = str(league_size)
        self.team_balance = str(team_balance)
        self.team_composition = team_composition
   
    def scrape_ecr(self):
        """
        Scrape Expert Consensus Rankings (ECR) data from FantasyPros
        """
        url = DOMAIN_PROJECTIONS + URL_SCORING_MAP[self.projection][self.scoring_format]

        results = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(results.text, 'html.parser')

        scripts = soup.find_all("script")

        for script in scripts:
            if (script.string):
                ecr_found = re.search("var ecrData = {.*};", script.string)
                if ecr_found: 
                    ecr_data = ecr_found.group(0).replace("var ecrData = ", "").replace(";", "")
                    ecr_data =  json.loads(ecr_data)
                
        remove_substrings = [" Jr.", " III", " II", " IV", " Sr."]
        # Remove certain substrings from player name 
        for player in ecr_data['players']:
            player_name = player.get('player_name')
    
            # Skip this player if no name exists
            if not player_name:
                continue

            for sub in remove_substrings:
                if player_name in ['Las Vegas Raiders', 'Minnesota Vikings']: continue
                if sub in player_name:
                    player_name = player_name.replace(sub, '')
            player['player_name'] = player_name
            player['rank_ecr'] = float(player['rank_ecr'])
            player['rank_ave'] = float(player['rank_ave'])

        return ecr_data

    def scrape_vbd(self):
        """
        Scrape Value Based Drafting Data (VBD) data from FantasyPros
        """
        url = DOMAIN_PROJECTIONS + URL_SCORING_MAP[self.projection][self.scoring_format] \
                            + "?year=" + self.year \
                            + "&team_size=" + self.league_size

        results = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(results.text, 'html.parser')

        vbd_data = {}
        remove_substrings = [" Jr.", " III", " II", " IV", " Sr."]

        for row in soup.find_all('tr', class_='player-row'):

            # Extract player name
            player_name_elem = row.find('a', class_='player-name')
            if not player_name_elem:
                continue
            player_name = player_name_elem.text.strip()

            for sub in remove_substrings:
                if player_name in ['Las Vegas Raiders', 'Minnesota Vikings']: continue
                if sub in player_name:
                    player_name = player_name.replace(sub, '')

            # Extract other data
            columns = row.find_all('td')
            rank = columns[0].text.strip()
            vbd = columns[3].get('data-value')
            vorp = columns[4].get('data-value')
            vols = columns[5].get('data-value')

            # Store data in dictionary
            vbd_data[player_name] = {
                'rank': int(rank),
                'vbd': int(vbd),
                'vorp': int(vorp),
                'vols': int(vols)
            }

        
        return vbd_data
    
    def scrape_adp(self):
        """
        Scrape Average Draft Position (ADP) data from FantasyPros
        """
        url = DOMAIN_PROJECTIONS + URL_SCORING_MAP[self.projection][self.scoring_format] \
                               + "?year=" + self.year
        

        results = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(results.text, 'html.parser')

        adp_data = {}

        remove_substrings = [" Jr.", " III", " II", " IV", " Sr."]

        for row in soup.find_all('tr'):
            columns = row.find_all('td')
            if len(columns) < 3:  # Skip rows that don't have at least rank, name, and adp_avg
                continue

            # Extract player name
            player_name_elem = row.find('a', class_='player-name')
            if not player_name_elem:
                continue
            player_name = player_name_elem.text.strip()

            for sub in remove_substrings:
                if player_name in ['Las Vegas Raiders DST', 'Minnesota Vikings DST']: continue
                if sub in player_name:
                    player_name = player_name.replace(sub, '')

            position = ''
            rank_str = ''
            
            for char in columns[2].text.strip():
                if char.isdigit():
                    rank_str += char
                else:
                    position += char

            if position == 'DST':
                player_name = player_name.replace(" DST", "")

            # Extract adp_avg (always the last column)
            adp_avg = columns[-1].text.strip()

            # Extract the first number (handles decimals, commas, and ignores extra digits)
            match = re.search(r'[\d,]+\.?\d*', adp_avg)
            if match:
                adp_avg_value = float(match.group().replace(',', ''))
            else:
                adp_avg_value = None

            # Store data in dictionary
            adp_data[player_name] = {
                'pos': position,
                'projected_pos_rank': int(rank_str) if rank_str else None,
                'adp_avg': adp_avg_value
            }


        return adp_data

    def scrape_auction_values(self):
        """
        Scrape Auction Value data from FantasyPros
        """
        sport = "nfl"
        showAuction = "Y"

        team_composition_string = "".join(["&" + position + "=" + str(count) for position, count in self.team_composition.items()])


        url = DOMAIN_AUCTION + 'sport=' + sport \
                            + '&showAuction=' + showAuction \
                            + '&scoringSystem=' + URL_SCORING_MAP['auction'][self.scoring_format] \
                            + '&tb=' + self.team_balance \
                            + '&teams=' + self.league_size \
                            + team_composition_string
        

        results = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(results.text, 'html.parser')

        auction_data = {}

        remove_substrings = [" Jr.", " III", " II", " IV", " Sr."]

        for row in soup.find_all('tr', class_=["PlayerQB", "PlayerRB", "PlayerWR", "PlayerTE", "PlayerDST", "PlayerK"]):
            columns = row.find_all('td')

            # Extract player name
            player_name = columns[1].text.split("(")[0].strip()

            #TODO: Address extra names that are getting parsed
            if ',' in player_name:
                continue

            try:
                team = columns[1].text.split('(')[1].split(' - ')[0]
            except:
                nfl_teams_abbr = {
                    'Arizona Cardinals': 'ARI',
                    'Atlanta Falcons': 'ATL',
                    'Baltimore Ravens': 'BAL',
                    'Buffalo Bills': 'BUF',
                    'Carolina Panthers': 'CAR',
                    'Chicago Bears': 'CHI',
                    'Cincinnati Bengals': 'CIN',
                    'Cleveland Browns': 'CLE',
                    'Dallas Cowboys': 'DAL',
                    'Denver Broncos': 'DEN',
                    'Detroit Lions': 'DET',
                    'Green Bay Packers': 'GB',
                    'Houston Texans': 'HOU',
                    'Indianapolis Colts': 'IND',
                    'Jacksonville Jaguars': 'JAX',
                    'Kansas City Chiefs': 'KC',
                    'Las Vegas Raiders': 'LV',
                    'Los Angeles Chargers': 'LAC',
                    'Los Angeles Rams': 'LAR',
                    'Miami Dolphins': 'MIA',
                    'Minnesota Vikings': 'MIN',
                    'New England Patriots': 'NE',
                    'New Orleans Saints': 'NO',
                    'New York Giants': 'NYG',
                    'New York Jets': 'NYJ',
                    'Philadelphia Eagles': 'PHI',
                    'Pittsburgh Steelers': 'PIT',
                    'San Francisco 49ers': 'SF',
                    'Seattle Seahawks': 'SEA',
                    'Tampa Bay Buccaneers': 'TB',
                    'Tennessee Titans': 'TEN',
                    'Washington Commanders': 'WAS'
                }

                team = nfl_teams_abbr[player_name]

            for sub in remove_substrings:
                if sub in player_name:
                    if player_name in ['Las Vegas Raiders', 'Minnesota Vikings']: continue
                    player_name = player_name.replace(sub, '')


            # Extract other data
            points = columns[2].text
            value = columns[4].text

            # Store data in dictionary
            auction_data[player_name] = {
                'team': team,
                'points': int(points),
                'value': int(value)
            }

        return auction_data

            

