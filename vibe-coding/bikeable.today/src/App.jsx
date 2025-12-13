import { useState, useEffect } from 'react'
import { fetchWeatherData } from './weatherService'
import { evaluateBikeability } from './bikeabilityEngine'

function App() {
  const [weatherData, setWeatherData] = useState(null);
  const [bikeabilityResult, setBikeabilityResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadWeather() {
      const startTime = Date.now();
      
      try {
        const data = await fetchWeatherData();
        setWeatherData(data);
        
        const result = evaluateBikeability(data);
        setBikeabilityResult(result);
      } catch (error) {
        console.error('Failed to load weather data:', error);
      } finally {
        // Ensure loading shows for at least 2 seconds
        const elapsedTime = Date.now() - startTime;
        const remainingTime = Math.max(0, 2000 - elapsedTime);
        
        setTimeout(() => {
          setLoading(false);
        }, remainingTime);
      }
    }
    
    loadWeather();
  }, []);

  // Update background based on result
  useEffect(() => {
    const body = document.body;
    
    if (loading || !bikeabilityResult) {
      body.className = 'gray-bg';
    } else if (bikeabilityResult.decision === 'YES') {
      body.className = 'sunny-bg';
    } else {
      body.className = 'gray-bg';
    }
    
    // Cleanup function to reset background when component unmounts
    return () => {
      body.className = '';
    };
  }, [loading, bikeabilityResult]);

  const getDecisionClass = () => {
    if (!bikeabilityResult) return 'decision';
    return `decision ${bikeabilityResult.decision.toLowerCase()}`;
  };

  const formatDecodedData = (decodedData, sourceType) => {
    if (!decodedData) return 'no data available';
    
    if (sourceType === 'METAR') {
      const temp = decodedData.temperature ? `${Math.round(decodedData.temperature)}°C` : 'no temp';
      const wind = decodedData.windSpeed ? `${decodedData.windSpeed}kt` : 'calm';
      
      let precip = 'clear';
      if (decodedData.precipitation) {
        if (decodedData.hasThunderstorms) {
          precip = 'thunderstorms';
        } else if (decodedData.hasFreezing) {
          precip = 'freezing precipitation';
        } else if (decodedData.hasSnow) {
          precip = 'snow';
        } else if (decodedData.hasRain) {
          precip = 'rain';
        } else {
          precip = 'precipitation';
        }
      }
      
      const ice = decodedData.ice ? 'icy' : '';
      
      return `${temp}, ${wind} wind, ${precip}${ice ? ', ' + ice : ''}`;
    } else if (sourceType === 'TAF') {
      if (decodedData.periods && decodedData.periods.length > 0) {
        const issues = [];
        let hasThunderstorms = false;
        let hasPrecip = false;
        let hasIce = false;
        
        decodedData.periods.forEach(period => {
          if (period.hasThunderstorms) hasThunderstorms = true;
          if (period.precipitation) hasPrecip = true;
          if (period.ice) hasIce = true;
        });
        
        if (hasThunderstorms) issues.push('thunderstorms');
        else if (hasPrecip) issues.push('precipitation');
        if (hasIce) issues.push('ice');
        
        return issues.length > 0 ? issues.join(', ') + ' forecast' : 'clear forecast';
      }
      return 'no forecast periods';
    }
    
    return 'unknown data format';
  };

  return (
    <div className="app">
      <header>
        <h1>can i bike.today</h1>
        <p>Manhattan, NYC</p>
      </header>
      <main>
        <div className={getDecisionClass()}>
          <h2>
            {loading ? (
              <span className="loading-spinner"></span>
            ) : (
              bikeabilityResult?.message || 'Error loading data'
            )}
          </h2>
          {!loading && bikeabilityResult?.explanation && (
            <p className="explanation">{bikeabilityResult.explanation}</p>
          )}
          {!loading && bikeabilityResult?.airportBreakdown?.[0]?.dataSource && (
            <p className="data-source">{bikeabilityResult.airportBreakdown[0].dataSource}</p>
          )}
        </div>
      </main>
      
      <div className="console-section">
        <h3>Raw Weather Data</h3>
        
        {/* Decoded Data Used for Decision - Always present */}
        <div className="decoded-section">
          <h4>Decision Based On:</h4>
          {loading ? (
            <div className="airport-decoded">
              <div className="airport-name">Loading...</div>
              <div className="commute-data">
                <span className="commute-label">Fetching data...</span>
                <span className="decoded-summary">...</span>
              </div>
            </div>
          ) : (
            bikeabilityResult?.airportBreakdown?.map((airport) => (
              <div key={airport.airport} className="airport-decoded">
                <div className="airport-name">{airport.airport}</div>
                {airport.morning && (
                  <div className="commute-data">
                    <span className="commute-label">Morning ({airport.morning.sourceType}):</span>
                    <span className="decoded-summary">
                      {formatDecodedData(airport.morning.decodedData, airport.morning.sourceType)}
                    </span>
                  </div>
                )}
                {airport.evening && (
                  <div className="commute-data">
                    <span className="commute-label">Evening ({airport.evening.sourceType}):</span>
                    <span className="decoded-summary">
                      {formatDecodedData(airport.evening.decodedData, airport.evening.sourceType)}
                    </span>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
        
        {/* Raw Data */}
        <div className="console">
          {loading ? (
            'Loading weather data...'
          ) : (
            weatherData && Object.entries(weatherData).map(([airport, data]) => (
              <div key={airport} className="airport-group">
                <div className="console-line">
                  <span className="console-prompt">{airport} METAR&gt;</span>
                  <span className="console-text">{data.metar}</span>
                </div>
                <div className="console-line">
                  <span className="console-prompt">{airport} TAF&gt;</span>
                  <span className="console-text">{data.taf}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default App