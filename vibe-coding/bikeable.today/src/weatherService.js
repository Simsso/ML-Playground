const AIRPORTS = {
  LGA: 'KLGA',
  JFK: 'KJFK', 
  EWR: 'KEWR'
};

const PROXY_BASE_URL = 'https://us-central1-timo-websites.cloudfunctions.net';

export async function fetchWeatherData() {
  const weatherData = {};
  
  for (const [name, code] of Object.entries(AIRPORTS)) {
    try {
      const [metar, taf] = await Promise.all([
        fetchMetar(code),
        fetchTaf(code)
      ]);
      
      weatherData[name] = {
        metar: metar || 'No METAR available',
        taf: taf || 'No TAF available'
      };
    } catch (error) {
      console.error(`Error fetching weather for ${name}:`, error);
      weatherData[name] = {
        metar: 'Error loading METAR',
        taf: 'Error loading TAF'
      };
    }
  }
  
  return weatherData;
}

async function fetchMetar(stationCode) {
  const url = `${PROXY_BASE_URL}/metar?station=${stationCode}`;
  
  try {
    const response = await fetch(url);
    const text = await response.text();
    
    // METAR files contain the raw METAR data directly
    return text.trim();
  } catch (error) {
    console.error('METAR fetch error:', error);
    return null;
  }
}

async function fetchTaf(stationCode) {
  const url = `${PROXY_BASE_URL}/taf?station=${stationCode}`;
  
  try {
    const response = await fetch(url);
    const text = await response.text();
    
    // TAF files contain the raw TAF data, may span multiple lines
    return text.trim();
  } catch (error) {
    console.error('TAF fetch error:', error);
    return null;
  }
}