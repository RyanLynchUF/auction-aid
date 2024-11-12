import './App.css';
import Landing from './Pages/Landing';
import Home from './Pages/Home'

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// TODO: remove proxy from package.json

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing/>} />
        <Route path="/home" element={<Home/>} />
      </Routes>
    </Router>
  )
}

export default App;
