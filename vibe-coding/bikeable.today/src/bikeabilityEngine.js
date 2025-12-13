import { decodeMetar, decodeTaf, isBikeable } from './weatherDecoder.js';

// Ride time constants
const MORNING_COMMUTE_TIME = 9; // 9:00am
const EVENING_COMMUTE_TIME = 18.5; // 6:30pm (18:30)

// Time window definitions
const TIME_WINDOWS = {
  MORNING: { start: 7, end: 10 },
  AFTERNOON: { start: 10, end: 15 },
  LATE_AFTERNOON: { start: 15, end: 19 },
  EVENING_NIGHT: { start: 19, end: 7 } // wraps around midnight
};

function getCurrentTimeWindow() {
  const now = new Date();
  const nycTime = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
  const hour = nycTime.getHours();
  
  if (hour >= TIME_WINDOWS.MORNING.start && hour < TIME_WINDOWS.MORNING.end) {
    return 'MORNING';
  } else if (hour >= TIME_WINDOWS.AFTERNOON.start && hour < TIME_WINDOWS.AFTERNOON.end) {
    return 'AFTERNOON';
  } else if (hour >= TIME_WINDOWS.LATE_AFTERNOON.start && hour < TIME_WINDOWS.LATE_AFTERNOON.end) {
    return 'LATE_AFTERNOON';
  } else {
    return 'EVENING_NIGHT';
  }
}

function getCommutesAndSources() {
  const timeWindow = getCurrentTimeWindow();
  
  switch (timeWindow) {
    case 'MORNING':
      return {
        checkMorning: true,
        checkEvening: true,
        morningSource: 'METAR', // Current conditions for 9am commute
        eveningSource: 'TAF',   // Forecast for 6:30pm return
        dayOffset: 0,
        reason: 'Morning window: checking both commutes today'
      };
    
    case 'AFTERNOON':
      return {
        checkMorning: false,
        checkEvening: true,
        morningSource: null,
        eveningSource: 'TAF',   // Forecast for 6:30pm return
        dayOffset: 0,
        reason: 'Afternoon: checking return home only'
      };
    
    case 'LATE_AFTERNOON':
      return {
        checkMorning: false,
        checkEvening: true,
        morningSource: null,
        eveningSource: 'METAR', // Current conditions for 6:30pm return
        dayOffset: 0,
        reason: 'Late afternoon: checking return home only'
      };
    
    case 'EVENING_NIGHT':
      return {
        checkMorning: true,
        checkEvening: true,
        morningSource: 'TAF',   // Forecast for tomorrow 9am commute
        eveningSource: 'TAF',   // Forecast for tomorrow 6:30pm return
        dayOffset: 1,
        reason: 'Evening/night: checking both commutes tomorrow'
      };
  }
}

export function evaluateAirportConsensus(weatherData) {
  const airports = ['LGA', 'JFK', 'EWR'];
  const results = [];
  const config = getCommutesAndSources();
  
  for (const airport of airports) {
    const data = weatherData[airport];
    if (!data) {
      results.push({
        airport,
        bikeable: false,
        reason: 'No data available',
        dataSource: config.reason
      });
      continue;
    }
    
    let morningResult = null;
    let eveningResult = null;
    let overallBikeable = true;
    let overallReason = null;
    
    // Check morning commute if needed
    if (config.checkMorning) {
      morningResult = evaluateCommute(data, config.morningSource, 'morning');
      if (!morningResult.bikeable) {
        overallBikeable = false;
        overallReason = `Morning: ${morningResult.reason}`;
      }
    }
    
    // Check evening commute if needed
    if (config.checkEvening) {
      eveningResult = evaluateCommute(data, config.eveningSource, 'evening');
      if (!eveningResult.bikeable) {
        overallBikeable = false;
        if (!overallReason) {
          overallReason = `Evening: ${eveningResult.reason}`;
        }
      }
    }
    
    results.push({
      airport,
      bikeable: overallBikeable,
      reason: overallReason,
      dataSource: config.reason,
      morning: morningResult,
      evening: eveningResult,
      dayOffset: config.dayOffset
    });
  }
  
  return results;
}

function evaluateCommute(data, source, period) {
  if (source === 'METAR') {
    // Use current conditions
    const decodedData = decodeMetar(data.metar);
    const result = isBikeable(decodedData);
    return {
      ...result,
      sourceType: 'METAR',
      decodedData,
      rawData: data.metar
    };
  } else if (source === 'TAF') {
    // Use forecast
    const result = evaluateForecastPeriod(data.taf, period);
    return {
      ...result,
      sourceType: 'TAF',
      rawData: data.taf
    };
  } else {
    // Should not happen, but handle gracefully
    return { bikeable: false, reason: 'Invalid data source configuration' };
  }
}

function evaluateForecastPeriod(tafData, period) {
  // For now, check entire TAF for bad conditions
  // TODO: Could be enhanced to parse specific time periods
  const decodedData = decodeTaf(tafData);
  
  if (!decodedData || decodedData.periods.length === 0) {
    return { bikeable: true, reason: null, decodedData };
  }
  
  for (const forecastPeriod of decodedData.periods) {
    if (forecastPeriod.ice) {
      return { bikeable: false, reason: 'Ice conditions in forecast', decodedData };
    }
    if (forecastPeriod.precipitation) {
      // Create precise precipitation message
      let reason = 'Precipitation in forecast';
      if (forecastPeriod.hasThunderstorms) {
        reason = 'Thunderstorms in forecast';
      } else if (forecastPeriod.hasFreezing) {
        reason = 'Freezing precipitation in forecast';
      } else if (forecastPeriod.hasSnow) {
        reason = 'Snow in forecast';
      } else if (forecastPeriod.hasRain) {
        reason = 'Rain in forecast';
      }
      return { bikeable: false, reason, decodedData };
    }
  }
  
  return { bikeable: true, reason: null, decodedData };
}

export function makeConsensusDecision(airportResults) {
  const bikeableCount = airportResults.filter(r => r.bikeable).length;
  const totalAirports = airportResults.length;
  
  // Decision matrix based on our consensus logic
  if (bikeableCount === 3) {
    return {
      decision: 'YES',
      confidence: 'high',
      message: 'YES, BIKE TODAY!',
      explanation: null,
      airportBreakdown: airportResults
    };
  } else if (bikeableCount === 2) {
    const badAirport = airportResults.find(r => !r.bikeable);
    return {
      decision: 'MAYBE',
      confidence: 'medium', 
      message: 'MAYBE - CHECK CONDITIONS',
      explanation: extractUserFriendlyReason(badAirport.reason),
      airportBreakdown: airportResults
    };
  } else {
    // Aggregate the reasons without showing airport codes
    const reasons = airportResults
      .filter(r => !r.bikeable)
      .map(r => r.reason);
    
    const aggregatedReason = aggregateReasons(reasons);
    
    return {
      decision: 'NO',
      confidence: 'high',
      message: 'NO, TAKE THE SUBWAY',
      explanation: aggregatedReason,
      airportBreakdown: airportResults
    };
  }
}

function extractUserFriendlyReason(reason) {
  // Remove "Morning:" or "Evening:" prefix and clean up
  return reason.replace(/^(Morning|Evening):\s*/, '');
}

function aggregateReasons(reasons) {
  // Extract clean reasons without Morning/Evening prefixes
  const cleanReasons = reasons.map(r => extractUserFriendlyReason(r));
  
  // Group similar reasons
  const morningIssues = reasons.filter(r => r.startsWith('Morning:')).map(r => extractUserFriendlyReason(r));
  const eveningIssues = reasons.filter(r => r.startsWith('Evening:')).map(r => extractUserFriendlyReason(r));
  
  // If all reasons are the same, just show one
  const uniqueReasons = [...new Set(cleanReasons)];
  if (uniqueReasons.length === 1) {
    return uniqueReasons[0];
  }
  
  // If we have both morning and evening issues, be more specific
  if (morningIssues.length > 0 && eveningIssues.length > 0) {
    const uniqueMorning = [...new Set(morningIssues)];
    const uniqueEvening = [...new Set(eveningIssues)];
    
    if (uniqueMorning.length === 1 && uniqueEvening.length === 1) {
      if (uniqueMorning[0] === uniqueEvening[0]) {
        return uniqueMorning[0]; // Same issue both times
      } else {
        return `Morning: ${uniqueMorning[0]}, Evening: ${uniqueEvening[0]}`;
      }
    }
  }
  
  // Fallback: show the most common reason
  return uniqueReasons[0];
}

export function evaluateBikeability(weatherData) {
  const airportResults = evaluateAirportConsensus(weatherData);
  const consensusDecision = makeConsensusDecision(airportResults);
  
  return {
    ...consensusDecision,
    timestamp: new Date().toISOString(),
    dataSource: 'NOAA Aviation Weather'
  };
}