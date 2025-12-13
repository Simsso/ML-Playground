const functions = require('@google-cloud/functions-framework');

// Simple in-memory cache with 15-minute TTL
const cache = new Map();
const CACHE_TTL = 15 * 60 * 1000; // 15 minutes in milliseconds

function getCacheKey(type, station) {
  return `${type}:${station}`;
}

function getCachedData(key) {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  return null;
}

function setCachedData(key, data) {
  cache.set(key, {
    data,
    timestamp: Date.now()
  });
}

// CORS headers for browser requests
function setCorsHeaders(res) {
  res.set('Access-Control-Allow-Origin', '*');
  res.set('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type');
}

// METAR endpoint
functions.http('metar', async (req, res) => {
  setCorsHeaders(res);
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(204).send('');
    return;
  }
  
  try {
    const station = req.query.station || req.params.station;
    if (!station) {
      return res.status(400).json({ error: 'Station parameter required' });
    }
    
    const cacheKey = getCacheKey('metar', station);
    
    // Check cache first
    const cachedData = getCachedData(cacheKey);
    if (cachedData) {
      console.log(`Cache hit for METAR ${station}`);
      return res.send(cachedData);
    }
    
    // Fetch from NOAA if not in cache
    console.log(`Fetching METAR ${station} from NOAA`);
    const url = `https://tgftp.nws.noaa.gov/data/observations/metar/stations/${station}.TXT`;
    
    const response = await fetch(url);
    const text = await response.text();
    
    // Cache the result
    setCachedData(cacheKey, text);
    
    res.send(text);
  } catch (error) {
    console.error('METAR fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch METAR data' });
  }
});

// TAF endpoint
functions.http('taf', async (req, res) => {
  setCorsHeaders(res);
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(204).send('');
    return;
  }
  
  try {
    const station = req.query.station || req.params.station;
    if (!station) {
      return res.status(400).json({ error: 'Station parameter required' });
    }
    
    const cacheKey = getCacheKey('taf', station);
    
    // Check cache first
    const cachedData = getCachedData(cacheKey);
    if (cachedData) {
      console.log(`Cache hit for TAF ${station}`);
      return res.send(cachedData);
    }
    
    // Fetch from NOAA if not in cache
    console.log(`Fetching TAF ${station} from NOAA`);
    const url = `https://tgftp.nws.noaa.gov/data/forecasts/taf/stations/${station}.TXT`;
    
    const response = await fetch(url);
    const text = await response.text();
    
    // Cache the result
    setCachedData(cacheKey, text);
    
    res.send(text);
  } catch (error) {
    console.error('TAF fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch TAF data' });
  }
});

// Cache status endpoint for debugging
functions.http('cacheStatus', async (req, res) => {
  setCorsHeaders(res);
  
  if (req.method === 'OPTIONS') {
    res.status(204).send('');
    return;
  }
  
  const cacheEntries = Array.from(cache.entries()).map(([key, value]) => ({
    key,
    age: Math.round((Date.now() - value.timestamp) / 1000),
    expired: Date.now() - value.timestamp > CACHE_TTL
  }));
  
  res.json({
    totalEntries: cache.size,
    cacheTTL: CACHE_TTL / 1000,
    entries: cacheEntries
  });
});