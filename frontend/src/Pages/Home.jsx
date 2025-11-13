// Home.js
import React, { useState } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';
import { FaInfoCircle } from 'react-icons/fa';
import Logo from '../Components/Auction_AId_logo_wide_beta.png'

const InfoBox = ({ title, value }) => (
  <div className="bg-[#272526] p-4 rounded-lg shadow-md flex-1 m-2">
    <h3 className="text-lg font-semibold mb-2 text-gray-300">{title}</h3>
    <p className="text-2xl font-bold text-[#8c52ff]">{value}</p>
  </div>
);

const Dropdown = ({ label, options, value, onChange, name, info }) => (
  <div className="mb-4">
    <div className="flex items-center mb-1">
      <label className="text-sm font-medium text-white">
        {label} 
      </label>
      <div className="relative group">
        <FaInfoCircle className="text-gray-400 cursor-pointer ml-2" />
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 hidden group-hover:block bg-black text-white text-xs px-4 py-2 w-56 max-w-xs rounded-md shadow-lg">
          {info}
        </div>
        </div>
    </div>
    <select
      name={name}
      value={value}
      onChange={onChange}
      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border border-[#7042cc] focus:outline-none focus:ring-[#8c52ff] focus:border-[#8c52ff] bg-[#241c1d] text-white sm:text-sm rounded-md"
    >
      {options.map((option) => (
        <option key={option} value={option}>{option}</option>
      ))}
    </select>
  </div>
);

const Accordion = ({ isOpen, toggle, children }) => (
  <div className="border border-[#7042cc] rounded-lg shadow-md mb-4">
    <button
      onClick={toggle}
      className="w-full px-4 py-2 text-left font-semibold text-white bg-[#312f30] rounded-t-lg focus:outline-none"
    >
      {isOpen ? 'Hide Settings ▲' : 'Show Settings ▼'}
    </button>
    {isOpen && <div className="p-4 bg-[#312f30]">{children}</div>}
  </div>
);

const LoadingScreen = ({ statusMessage }) => (
  <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
    <div className="mx-4 sm:mx-8 md:mx-16 lg:mx-32 xl:mx-48 p-4 text-center flex flex-col items-center">
      <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-[#8c52ff] mb-4"></div>
      <p className="text-white text-xl font-semibold">{statusMessage}</p>
    </div>
  </div>
);


// Function to calculate percentiles for an array of scores
const calculatePercentile = (data, percentile) => {
  const sorted = [...data].sort((a, b) => a - b);
  const index = Math.floor(percentile * sorted.length);
  return sorted[index];
};

// Updated function to calculate percentiles per position
const calculatePositionPercentiles = (auctionData) => {
  const percentilesByPosition = {};
  
  // Group players by position
  const playersByPosition = auctionData.reduce((acc, player) => {
    if (!acc[player.pos]) acc[player.pos] = [];
    acc[player.pos].push(player.normalized_draft_score);
    return acc;
  }, {});
  
  // Calculate percentiles for each position
  Object.keys(playersByPosition).forEach(pos => {
    const scores = playersByPosition[pos].sort((a, b) => a - b);
    percentilesByPosition[pos] = {
      lower: calculatePercentile(scores, 0.15),  // Bottom 15%
      upper: calculatePercentile(scores, 0.85),  // Top 15%
      min: Math.min(...scores),
      max: Math.max(...scores)
    };
  });
  
  return percentilesByPosition;
};

// Updated color function with gradient intensity
const getBackgroundColor = (score, position, percentilesByPosition) => {
  const posPercentiles = percentilesByPosition[position];
  
  if (!posPercentiles) return '#312f30'; // Neutral if position not found
  
  const { lower, upper, min, max } = posPercentiles;
  
  if (score >= upper) {
    // Top 15%: Green gradient (brighter = better)
    // Normalize score within the top 15% range (upper to max)
    const normalizedScore = (score - upper) / (max - upper);
    
    // Green intensity: ranges from medium green to bright green
    // RGB: (34, 197, 94) is a nice bright green, (22, 163, 74) is darker
    const greenR = Math.round(22 + (normalizedScore * 28));      // 22 -> 50
    const greenG = Math.round(163 + (normalizedScore * 34));     // 163 -> 197
    const greenB = Math.round(74 + (normalizedScore * 20));      // 74 -> 94
    
    return `rgb(${greenR}, ${greenG}, ${greenB})`;
    
  } else if (score <= lower) {
    // Bottom 15%: Red gradient (brighter = worse)
    // Normalize score within the bottom 15% range (min to lower)
    const normalizedScore = (score - min) / (lower - min);
    
    // Red intensity: ranges from bright red to medium red
    // Brighter red for worse players (lower scores)
    const redR = Math.round(220 - (normalizedScore * 60));       // 220 -> 160
    const redG = Math.round(38 - (normalizedScore * 8));         // 38 -> 30
    const redB = Math.round(38 - (normalizedScore * 8));         // 38 -> 30
    
    return `rgb(${redR}, ${redG}, ${redB})`;
    
  } else {
    // Middle 70%: Neutral gray
    return '#312f30';
  }
};

const AuctionTable = ({ auctionData }) => {
  // Calculate percentiles per position instead of globally
  const percentilesByPosition = calculatePositionPercentiles(auctionData);
  
  return (
    <div className="mt-8 overflow-x-auto max-h-[500px]">
      <table className="min-w-full divide-y divide-[#312f30]">
        <thead className="bg-[#7042cc] sticky top-0 z-10">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Avg. ECR</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Name</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Team</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Pos.</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Proj. Pos. Rank</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">VORP</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Last Year AV</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Avg. AV for Proj. Pos. Rank</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Expected AV</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">True AV</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#7042cc]">
          {auctionData.map((player, index) => (
            <tr
              key={index}
              style={{
                backgroundColor: getBackgroundColor(
                  player.normalized_draft_score, 
                  player.pos, 
                  percentilesByPosition
                ),
              }}
            >
              <td className="px-6 py-4 whitespace-nowrap text-white">{player.ecr_avg}</td>
              <td className="px-6 py-4 whitespace-nowrap text-white">{player.player_name}</td>
              <td className="px-6 py-4 whitespace-nowrap text-white">{player.team}</td>
              <td className="px-6 py-4 whitespace-nowrap text-white">{player.pos}</td>
              <td className="px-6 py-4 whitespace-nowrap text-white">{player.curr_year_projected_pos_rank}</td>
              <td className="px-6 py-4 whitespace-nowrap text-white">{player.vorp.toFixed(2)}</td>
              <td className="px-6 py-4 whitespace-nowrap text-white">
                {'$' + player.prev_year_bid_amt.toFixed(0)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-white">
                {'$' + player.projected_pos_rank_avg_bid_amt.toFixed(0)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-white">
                {'$' + player.expected_auction_value.toFixed(0)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-white">
                {'$' + player.true_auction_value.toFixed(0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const Home = () => {
  const location = useLocation();
  const apiResult = location.state?.apiResult;
  const { leagueId, swid, espnS2 } = location.state?.formData || {};
  const [accordionOpen, setAccordionOpen] = useState(true);
  const [auctionData, setAuctionData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState(''); 
  const [error, setError] = useState(null);

  const leagueData = typeof apiResult === 'string' ? JSON.parse(apiResult) : apiResult;

  const [formState, setFormState] = useState({
    scoringFormat: 'Standard',
    auctionDollarsPerTeam: '200',
    keepers: 'No',
    projectionsSource: 'FantasyPros',
    baselinePlayerStrategy: 'First $1 Player',
    statisticForVorpCalculation: 'PPG',
    teamDraftStrategy: 'Balanced',
    seasonsToInclude: [...leagueData.previous_seasons],
  });

  const handleDropdownChange = (e) => {
    setFormState({ ...formState, [e.target.name]: e.target.value });
  };

  const handleCheckboxChange = (season) => {
    setFormState(prevState => ({
      ...prevState,
      seasonsToInclude: prevState.seasonsToInclude.includes(season)
        ? prevState.seasonsToInclude.filter(s => s !== season)
        : [...prevState.seasonsToInclude, season]
    }));
  };

  const handleGenerateAuctionAid = async () => {
    setIsLoading(true);
    setStatusMessage('Initialization Auction AId...');
    setError(null);
    try {
      setStatusMessage('Pulling past league data, running models, and talking to the Fantasy Gods...  This may take a few minutes.  Especially if this is your first time running Auction AId.');
      const response = await axios.post('/api/generate-auction-aid', {
        leagueId,
        swid,
        espnS2,
        ...formState,
      });
      setStatusMessage('Finalizing results...');
      setAuctionData(response.data);
      setStatusMessage('');
    } catch (err) {
      if (err.response && err.response.data) {
        setError(err.response.data.detail); // Show custom error message
      } else {
        setError("Failed to generate Auction AId. Please try again.");
      }
      console.error('Error generating Auction AId:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-[#241c1d] min-h-screen flex items-center justify-center">
      {isLoading && <LoadingScreen statusMessage={statusMessage} />}
    <div className="max-w-7xl w-full px-6 py-12 bg-[#312f30]">
      <div className="mb-4">
        <img src={Logo} alt="App Logo" className="h-auto max-w-md mx-auto" />
      </div>

    <div className="flex flex-wrap justify-center mb-8">
      <InfoBox title="League ID" value={leagueData.league_id} />
      <InfoBox title="League Name" value={leagueData.league_name} />
      <InfoBox title="Season" value={leagueData.season_year} />
      <InfoBox title="League Size" value={leagueData.league_size + " teams"} />
    </div>

    <Accordion isOpen={accordionOpen} toggle={() => setAccordionOpen(!accordionOpen)}>
      <div className="flex flex-col md:flex-row gap-8">
        {/* League Settings Section */}
        <div className="flex-1">
          <h2 className="text-xl font-bold mb-4 text-white">League Settings</h2>
          <Dropdown
            label="Scoring Format"
            options={['Standard', 'PPR', 'Half-PPR']}
            value={formState.scoringFormat}
            onChange={handleDropdownChange}
            name="scoringFormat"
            info="Based on the Points-per-Reception (PPR) value for the league.  Standard is 0 PPR.  All other scoring is based on ESPN default scoring settings."
          />
          <Dropdown
            label="Auction $ Per Team"
            options={['200']}
            value={formState.auctionDollarsPerTeam}
            onChange={handleDropdownChange}
            name="auctionDollarsPerTeam"
            info="The starting auction budget for each team in the draft.  Only $200 is supported in Beta version."
          />
          <Dropdown
            label="Keepers"
            options={['No']}
            value={formState.keepers}
            onChange={handleDropdownChange}
            name="keepers"
            info="Indicate if your league has keepers.  Keepers not currently supported in Beta version."
          />

       
        </div>

        {/* Auction Aid Settings Section */}
        <div className="flex-1">
          <h2 className="text-xl font-bold mb-4 text-white">Auction AId Settings</h2>
          <Dropdown
            label="Projections Source"
            options={['FantasyPros']}
            value={formState.projectionsSource}
            onChange={handleDropdownChange}
            name="projectionsSource"
            info="The source for projected player performance (player rankings, expected points, etc).  Only FantasyPros is supported in Beta version."
          />
          <Dropdown
            label="Statistic Used for Value Over Replacement Player (VORP)"
            options={['PPG']}
            value={formState.statisticForVorpCalculation}
            onChange={handleDropdownChange}
            name="vorpCalculation"
            info="The value used to determine a player's actual performance in previous years.  Only PPG is supported in Beta version.  PPG mitigates impact of injuries, but may result in overvaluing players who played in only a few games."
          />
          <Dropdown
            label="Replacement Player for Value Over Replacement Player (VORP)"
            options={['First $1 Player', 'Last Starter']}
            value={formState.baselinePlayerStrategy}
            onChange={handleDropdownChange}
            name="baselinePlayerStrategy"
            info="The player that will be used as the baseline for determining the values of other players."
          />
          <Dropdown
            label="Team Draft Strategy"
            options={['Balanced']}
            value={formState.teamDraftStrategy}
            onChange={handleDropdownChange}
            name="teamDraftStrategy"
            info="Adjust the draft strategy to recommend large amounts on top players ('Spend on Studs') or to build a well-rounded roster ('Balanced').  Only 'Balanced' is support in Beta version."
          />

        </div>
      </div>
              
      <div className="flex items-center mb-1">
        <label className="block text-sm font-medium text-white mt-3 mb-2 mr-2">Past seasons to include in analysis. Only include seasons with auction drafts.  
        Auction AId works best with more history. Note: ESPN removed League data older than 2019, so these seasons will be excluded in calculations.</label>
            <div className="relative group">
              {leagueData.previous_seasons.map((season) => (
                <label key={season} className="inline-flex items-center mr-4">
                  <input
                    type="checkbox"
                    checked={formState.seasonsToInclude.includes(season)}
                    onChange={() => handleCheckboxChange(season)}
                    className="form-checkbox h-5 w-5 text-indigo-600"
                  />
                  <span className="ml-2 text-white">{season}</span>
                </label>
              ))}
              </div>
        </div>
    </Accordion>

    <div className="mt-8 flex justify-center">
      <button
        onClick={handleGenerateAuctionAid}
        disabled={isLoading}
        className="px-6 py-3 bg-[#7042cc] text-white font-semibold rounded-md shadow-sm hover:bg-[#8c52ff] focus:ring-2 focus:ring-offset-2 focus:ring-[#8c52ff] disabled:opacity-50"
      >
        {isLoading ? 'Generating...' : 'Generate Auction Values (AV)'}
      </button>

    </div>

    {error && <div className="mt-4 text-center text-red-600">{error}</div>}
    {auctionData && <AuctionTable auctionData={auctionData} />}
  </div>
</div>
  );
};

export default Home;
