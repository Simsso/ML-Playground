// Precise precipitation message formatting

export function formatPrecipitationMessage(decodedData, isCurrent = false) {
  if (!decodedData.precipitation) return null;
  
  const timeContext = isCurrent ? 'currently' : 'expected';
  
  if (decodedData.hasThunderstorms) {
    return isCurrent ? 'Thunderstorms currently' : 'Thunderstorms expected';
  } else if (decodedData.hasFreezing) {
    return isCurrent ? 'Freezing precipitation currently' : 'Freezing precipitation expected';
  } else if (decodedData.hasSnow) {
    return isCurrent ? 'Snow currently' : 'Snow expected';
  } else if (decodedData.hasRain) {
    return isCurrent ? 'Rain currently' : 'Rain expected';
  } else {
    return isCurrent ? 'Precipitation currently' : 'Precipitation expected';
  }
}

export function formatCombinedPrecipitationMessage(currentData, forecastData) {
  const currentMsg = currentData?.precipitation ? formatPrecipitationMessage(currentData, true) : null;
  const forecastMsg = forecastData?.precipitation ? formatPrecipitationMessage(forecastData, false) : null;
  
  if (currentMsg && forecastMsg) {
    // Both current and forecast have precipitation
    const currentType = getPrecipitationType(currentData);
    const forecastType = getPrecipitationType(forecastData);
    
    if (currentType === forecastType) {
      return `${currentType} currently and more ${currentType.toLowerCase()} expected`;
    } else {
      return `${currentMsg} and ${forecastMsg.toLowerCase()}`;
    }
  } else if (currentMsg) {
    return currentMsg;
  } else if (forecastMsg) {
    return forecastMsg;
  }
  
  return null;
}

function getPrecipitationType(decodedData) {
  if (decodedData.hasThunderstorms) return 'Thunderstorms';
  if (decodedData.hasFreezing) return 'Freezing precipitation';
  if (decodedData.hasSnow) return 'Snow';
  if (decodedData.hasRain) return 'Rain';
  return 'Precipitation';
}