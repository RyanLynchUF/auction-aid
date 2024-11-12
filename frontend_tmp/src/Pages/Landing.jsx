import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Logo from '../Components/Auction_AId_logo_wide_beta.png'

const Landing = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    leagueId: '',
    swid: '',
    espnS2: ''
  });
  const [loading, setLoading] = useState(false);  
  const [error, setError] = useState(''); 

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);  
    setError('');
    try {
      const response = await axios.post('/api/league/', formData);
      navigate('/home', { state: { apiResult: response.data, formData } });
    } catch (error) {
      console.error('Error fetching data:', error);
      setError('Error loading league data.  Please review the submitted credentials and try again.');
    } finally {
      setLoading(false);  
    }
  };

  return (
    <div className="bg-[#241c1d] min-h-screen flex items-center justify-center">
      <div className="flex items-center justify-center bg-gray-100">
        <div className="p-8 bg-[#312f30] shadow-md w-96">
          <div className="mb-4">
            <img src={Logo} alt="App Logo" className="w-full h-auto" />
          </div>
          <p className="block text-white">Enter your ESPN League details below.</p>
          <a className="block text-white mb-4" 
            href="https://example.com/faq.html" 
            rel="noreferrer"
            target="_blank" >
            Need help?  Click <b>here</b>.
          </a>
          <form onSubmit={handleSubmit}>
            <label htmlFor="leagueId" className="block text-white mb-1">League ID:</label>
            <input
              type="text"
              name="leagueId"
              value={formData.leagueId}
              onChange={handleInputChange}
              className="w-full p-2 mb-4 border border-gray-600 rounded bg-gray-800 text-white"
            />
            <label htmlFor="swid" className="block text-white mb-1">SWID (private leagues only):</label>
            <input
              type="text"
              name="swid"
              value={formData.swid}
              onChange={handleInputChange}
              className="w-full p-2 mb-4 border border-gray-600 rounded bg-gray-800 text-white"
            />
            <label htmlFor="espnS2" className="block text-white mb-1">ESPN S2 (private leagues only):</label>
            <input
              type="text"
              name="espnS2"
              value={formData.espnS2}
              onChange={handleInputChange}
              className="w-full p-2 mb-4 border border-gray-600 rounded bg-gray-800 text-white"
            />
            <button
              type="submit"
              className="w-full p-2 bg-[#8c52ff] text-white rounded hover:bg-[#7042cc]"
              disabled={loading}  
            >
              {loading ? "Loading league..." : "Submit"}
            </button>
          </form>  
          {error && <p className="text-red-500 mt-4">{error}</p>}
        </div>
      </div>
    </div>
  );
};

export default Landing;
