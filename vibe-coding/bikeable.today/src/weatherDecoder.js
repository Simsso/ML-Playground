// Aviation weather decoder for METAR and TAF data

export function decodeMetar(metarString) {
  if (!metarString || !metarString.trim()) return null;
  
  const decoded = {
    temperature: null,
    dewpoint: null,
    conditions: [],
    precipitation: false,
    ice: false,
    rawText: metarString
  };
  
  // Extract temperature/dewpoint (format: 28/22 or M05/M10 for negative)
  const tempMatch = metarString.match(/\b(M?\d{2})\/(M?\d{2})\b/);
  if (tempMatch) {
    decoded.temperature = parseTempCelsius(tempMatch[1]);
    decoded.dewpoint = parseTempCelsius(tempMatch[2]);
  }
  
  // Check for specific precipitation types (match whole weather codes only)
  const rainCodes = ['RA', 'DZ', 'SH']; // Rain, Drizzle, Showers
  const snowCodes = ['SN', 'PL', 'IC']; // Snow, Ice Pellets, Ice Crystals  
  const thunderCodes = ['TS']; // Thunderstorms
  const freezingCodes = ['FZ']; // Freezing precipitation
  
  // Use word boundaries to match whole weather codes, not substrings
  decoded.hasRain = rainCodes.some(code => new RegExp(`\\b${code}\\b`).test(metarString));
  decoded.hasSnow = snowCodes.some(code => new RegExp(`\\b${code}\\b`).test(metarString));
  decoded.hasThunderstorms = thunderCodes.some(code => new RegExp(`\\b${code}\\b`).test(metarString));
  decoded.hasFreezing = freezingCodes.some(code => new RegExp(`\\b${code}\\b`).test(metarString));
  
  // Legacy precipitation field (any precipitation)
  decoded.precipitation = decoded.hasRain || decoded.hasSnow || decoded.hasThunderstorms || decoded.hasFreezing;
  
  // Check for ice/freezing conditions (use word boundaries)
  const iceCodes = ['FZ', 'IC', 'PL'];
  decoded.ice = iceCodes.some(code => new RegExp(`\\b${code}\\b`).test(metarString));
  
  // Extract visibility
  const visMatch = metarString.match(/\b(\d+)SM\b/);
  if (visMatch) {
    decoded.visibility = parseInt(visMatch[1]);
  }
  
  return decoded;
}

export function decodeTaf(tafString) {
  if (!tafString) return null;
  
  const decoded = {
    periods: [],
    rawText: tafString
  };
  
  // Split TAF into periods (FM, TEMPO, PROB sections)
  const lines = tafString.split('\n').join(' ');
  const periods = lines.split(/(?=FM\d{6}|TEMPO|PROB\d{2})/);
  
  for (const period of periods) {
    if (period.trim()) {
      const periodData = {
        text: period.trim(),
        precipitation: false,
        ice: false,
        temperature: null
      };
      
      // Check for specific precipitation types in this period
      const rainCodes = ['RA', 'DZ', 'SH'];
      const snowCodes = ['SN', 'PL', 'IC'];
      const thunderCodes = ['TS'];
      const freezingCodes = ['FZ'];
      
      periodData.hasRain = rainCodes.some(code => period.includes(code));
      periodData.hasSnow = snowCodes.some(code => period.includes(code));
      periodData.hasThunderstorms = thunderCodes.some(code => period.includes(code));
      periodData.hasFreezing = freezingCodes.some(code => period.includes(code));
      
      // Legacy precipitation field
      periodData.precipitation = periodData.hasRain || periodData.hasSnow || periodData.hasThunderstorms || periodData.hasFreezing;
      
      // Check for ice/freezing
      const iceCodes = ['FZ', 'IC', 'PL'];
      periodData.ice = iceCodes.some(code => period.includes(code));
      
      decoded.periods.push(periodData);
    }
  }
  
  return decoded;
}

function parseTempCelsius(tempString) {
  if (!tempString) return null;
  
  // Handle negative temperatures (M prefix)
  const isNegative = tempString.startsWith('M');
  const temp = parseInt(isNegative ? tempString.slice(1) : tempString);
  
  return isNegative ? -temp : temp;
}

// Convert Celsius to Fahrenheit
export function celsiusToFahrenheit(celsius) {
  return celsius !== null ? Math.round((celsius * 9/5) + 32) : null;
}

// Check if conditions are bikeable based on our thresholds
export function isBikeable(decodedData) {
  if (!decodedData || typeof decodedData.temperature !== 'number') return false;
  
  // Check for ice first (priority over temperature)
  if (decodedData.ice) {
    return { bikeable: false, reason: 'Ice conditions' };
  }
  
  // Check for precipitation with specific types
  if (decodedData.precipitation) {
    if (decodedData.hasThunderstorms) {
      return { bikeable: false, reason: 'Thunderstorms expected' };
    } else if (decodedData.hasFreezing) {
      return { bikeable: false, reason: 'Freezing precipitation expected' };
    } else if (decodedData.hasSnow) {
      return { bikeable: false, reason: 'Snow expected' };
    } else if (decodedData.hasRain) {
      return { bikeable: false, reason: 'Rain expected' };
    } else {
      return { bikeable: false, reason: 'Precipitation expected' };
    }
  }
  
  // Check temperature (must be 42-90°F)
  if (decodedData.temperature !== null) {
    const fahrenheit = celsiusToFahrenheit(decodedData.temperature);
    if (fahrenheit < 42 || fahrenheit > 90) {
      return { bikeable: false, reason: `Temperature ${fahrenheit}°F` };
    }
  }
  
  return { bikeable: true, reason: null };
}