import { describe, test, expect } from 'vitest';
import { decodeMetar, decodeTaf, celsiusToFahrenheit, isBikeable } from './weatherDecoder.js';

// Real world METAR examples from major airports
const testMetars = {
  // JFK (NYC) - Good conditions, 28°C/82°F
  KJFK: 'KJFK 171551Z 20009KT 10SM FEW030 FEW090 BKN250 28/22 A2992 RMK AO2 SLP131 T02830217 $',
  
  // LAX (Los Angeles) - Cool conditions, 21°C/70°F
  KLAX: 'KLAX 171553Z 09003KT 10SM OVC020 21/16 A3001 RMK AO2 SLP159 T02110156 $',
  
  // ORD (Chicago) - Warm conditions, 26°C/79°F
  KORD: 'KORD 171606Z 05009KT 10SM BKN017 BKN042 BKN180 BKN230 26/22 A3008 RMK AO2 T02610222 $',
  
  // DEN (Denver) - Hot conditions, 27°C/81°F
  KDEN: 'KDEN 171553Z 30003KT 10SM FEW090 FEW130 27/07 A3015 RMK AO2 SLP124 T02720072 $',
  
  // MIA (Miami) - Very hot, 33°C/91°F (too hot for biking)
  KMIA: 'KMIA 171553Z VRB03KT 10SM BKN041 33/24 A2995 RMK AO2 SLP141 T03330239',
  
  // PHX (Phoenix) - Extremely hot, 33°C/91°F (too hot for biking)
  KPHX: 'KPHX 171551Z 03005KT 10SM FEW140 33/05 A2996 RMK AO2 SLP124 T03280050',
  
  // SEA (Seattle) - Cool conditions, 18°C/64°F
  KSEA: 'KSEA 171553Z 19004KT 10SM SCT019 OVC070 18/13 A3006 RMK AO2 SLP182 T01780133 $',
  
  // ATL (Atlanta) - Hot conditions, 30°C/86°F
  KATL: 'KATL 171552Z 00000KT 10SM CLR 30/20 A3008 RMK AO2 SLP175 T03000200 $',
  
  // BOS (Boston) - Hot conditions, 29°C/84°F
  KBOS: 'KBOS 171554Z 19008KT 10SM FEW050 29/20 A3006 RMK AO2 SLP174 T02890200 $',
  
  // LGA with SCT clouds (should NOT detect snow from "SCT")
  KLGA_SCT: 'KLGA 171951Z 16014KT 10SM FEW039 SCT055 SCT170 BKN250 31/21 A2981 RMK AO2 SLP096 TCU DSNT NE T03110206 $'
};

// Real world TAF examples
const testTafs = {
  KJFK: 'AMD KJFK 171435Z 1715/1818 21013KT P6SM FEW030 SCT060 BKN250 FM171900 18017G21KT P6SM BKN050 FM172200 23008KT P6SM BKN050 PROB30 1723/1803 4SM TSRA BKN040CB FM180400 34008KT P6SM VCSH SCT020 BKN040 FM180600 01013KT P6SM VCSH SCT015 BKN040 FM180700 02015G23KT P6SM SCT015 BKN030 TEMPO 1808/1812 4SM -SHRA BR BKN015 FM181500 06014G20KT P6SM VCSH SCT015 BKN035',
  
  KLAX: 'AMD KLAX 171432Z 1715/1818 12005KT P6SM OVC018 FM171800 23008KT P6SM BKN020 FM172000 25018KT P6SM FEW020 FM180400 25007KT P6SM SCT012 FM181000 VRB03KT P6SM OVC010 FM181700 24006KT P6SM BKN015',
  
  KSEA: 'AMD KSEA 171457Z 1715/1818 18006KT P6SM SCT015 OVC070 TEMPO 1715/1717 BKN015 OVC070 FM171900 22006KT P6SM SCT050 BKN200 FM172200 25006KT P6SM SCT050 BKN200 FM180200 31006KT P6SM BKN200 FM180800 15004KT P6SM BKN200'
};

describe('Weather Decoder', () => {
  describe('METAR Parsing', () => {
    test('decodes JFK METAR correctly', () => {
      const result = decodeMetar(testMetars.KJFK);
      expect(result.temperature).toBe(28);
      expect(result.dewpoint).toBe(22);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
      expect(result.rawText).toBe(testMetars.KJFK);
    });

    test('decodes LAX METAR correctly', () => {
      const result = decodeMetar(testMetars.KLAX);
      expect(result.temperature).toBe(21);
      expect(result.dewpoint).toBe(16);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('decodes ORD METAR correctly', () => {
      const result = decodeMetar(testMetars.KORD);
      expect(result.temperature).toBe(26);
      expect(result.dewpoint).toBe(22);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('decodes DEN METAR correctly', () => {
      const result = decodeMetar(testMetars.KDEN);
      expect(result.temperature).toBe(27);
      expect(result.dewpoint).toBe(7);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('decodes MIA hot METAR correctly', () => {
      const result = decodeMetar(testMetars.KMIA);
      expect(result.temperature).toBe(33);
      expect(result.dewpoint).toBe(24);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('decodes PHX very hot METAR correctly', () => {
      const result = decodeMetar(testMetars.KPHX);
      expect(result.temperature).toBe(33);
      expect(result.dewpoint).toBe(5);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('decodes SEA cool METAR correctly', () => {
      const result = decodeMetar(testMetars.KSEA);
      expect(result.temperature).toBe(18);
      expect(result.dewpoint).toBe(13);
      expect(result.precipitation).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('handles null/empty METAR', () => {
      expect(decodeMetar(null)).toBe(null);
      expect(decodeMetar('')).toBe(null);
      expect(decodeMetar('   ')).toBe(null);
    });

    test('does not detect snow in SCT (scattered) cloud codes', () => {
      const result = decodeMetar(testMetars.KLGA_SCT);
      expect(result.temperature).toBe(31);
      expect(result.dewpoint).toBe(21);
      expect(result.precipitation).toBe(false);
      expect(result.hasSnow).toBe(false);
      expect(result.hasRain).toBe(false);
      expect(result.hasThunderstorms).toBe(false);
      expect(result.hasFreezing).toBe(false);
      expect(result.ice).toBe(false);
    });

    test('handles negative temperatures with M prefix', () => {
      const coldMetar = 'KORD 171606Z 05009KT 10SM BKN017 M15/M20 A3008';
      const result = decodeMetar(coldMetar);
      expect(result.temperature).toBe(-15);
      expect(result.dewpoint).toBe(-20);
    });

    test('detects specific precipitation types correctly', () => {
      // Test rain detection
      const rainyMetar = 'KJFK 171551Z 20009KT 10SM RA 20/18 A2992';
      const rainResult = decodeMetar(rainyMetar);
      expect(rainResult.hasRain).toBe(true);
      expect(rainResult.hasSnow).toBe(false);
      expect(rainResult.precipitation).toBe(true);

      // Test snow detection
      const snowyMetar = 'KJFK 171551Z 20009KT 10SM SN 20/18 A2992';
      const snowResult = decodeMetar(snowyMetar);
      expect(snowResult.hasSnow).toBe(true);
      expect(snowResult.hasRain).toBe(false);
      expect(snowResult.precipitation).toBe(true);

      // Test thunderstorm detection
      const thunderMetar = 'KJFK 171551Z 20009KT 10SM TS 20/18 A2992';
      const thunderResult = decodeMetar(thunderMetar);
      expect(thunderResult.hasThunderstorms).toBe(true);
      expect(thunderResult.hasRain).toBe(false);
      expect(thunderResult.precipitation).toBe(true);

      // Test drizzle (should count as rain)
      const drizzleMetar = 'KJFK 171551Z 20009KT 10SM DZ 20/18 A2992';
      const drizzleResult = decodeMetar(drizzleMetar);
      expect(drizzleResult.hasRain).toBe(true);
      expect(drizzleResult.precipitation).toBe(true);
    });

    test('detects various ice codes', () => {
      const iceTypes = ['FZ', 'IC', 'PL'];
      iceTypes.forEach(code => {
        const metarWithIce = `KJFK 171551Z 20009KT 10SM ${code} 20/18 A2992`;
        const result = decodeMetar(metarWithIce);
        expect(result.ice).toBe(true);
      });
    });
  });

  describe('TAF Parsing', () => {
    test('decodes JFK TAF with precipitation periods', () => {
      const result = decodeTaf(testTafs.KJFK);
      expect(result.periods.length).toBeGreaterThan(0);
      expect(result.rawText).toBe(testTafs.KJFK);
      
      // Should detect thunderstorm (TSRA) and shower (SHRA) periods
      const hasThunderstorms = result.periods.some(p => p.precipitation && p.text.includes('TSRA'));
      const hasShowers = result.periods.some(p => p.precipitation && p.text.includes('SHRA'));
      expect(hasThunderstorms || hasShowers).toBe(true);
    });

    test('decodes LAX TAF correctly', () => {
      const result = decodeTaf(testTafs.KLAX);
      expect(result.periods.length).toBeGreaterThan(0);
      expect(result.rawText).toBe(testTafs.KLAX);
    });

    test('decodes SEA TAF correctly', () => {
      const result = decodeTaf(testTafs.KSEA);
      expect(result.periods.length).toBeGreaterThan(0);
      expect(result.rawText).toBe(testTafs.KSEA);
    });

    test('handles null/empty TAF', () => {
      expect(decodeTaf(null)).toBe(null);
      expect(decodeTaf('')).toBe(null);
    });
  });

  describe('Temperature Conversion', () => {
    test('converts Celsius to Fahrenheit correctly', () => {
      expect(celsiusToFahrenheit(0)).toBe(32);
      expect(celsiusToFahrenheit(28)).toBe(82);
      expect(celsiusToFahrenheit(33)).toBe(91);
      expect(celsiusToFahrenheit(18)).toBe(64);
      expect(celsiusToFahrenheit(-5)).toBe(23);
      expect(celsiusToFahrenheit(null)).toBe(null);
    });
  });

  describe('Bikeability Logic', () => {
    test('approves good conditions (JFK - 82°F, no precip)', () => {
      const decoded = decodeMetar(testMetars.KJFK);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(true);
      expect(result.reason).toBe(null);
    });

    test('approves borderline warm conditions (LAX - 70°F)', () => {
      const decoded = decodeMetar(testMetars.KLAX);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(true);
      expect(result.reason).toBe(null);
    });

    test('approves good warm conditions (ORD - 79°F)', () => {
      const decoded = decodeMetar(testMetars.KORD);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(true);
      expect(result.reason).toBe(null);
    });

    test('approves good conditions (SEA - 64°F)', () => {
      const decoded = decodeMetar(testMetars.KSEA);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(true);
      expect(result.reason).toBe(null);
    });

    test('rejects too hot conditions (MIA - 91°F)', () => {
      const decoded = decodeMetar(testMetars.KMIA);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(false);
      expect(result.reason).toBe('Temperature 91°F');
    });

    test('rejects too hot conditions (PHX - 91°F)', () => {
      const decoded = decodeMetar(testMetars.KPHX);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(false);
      expect(result.reason).toBe('Temperature 91°F');
    });

    test('rejects too cold conditions', () => {
      const coldMetar = 'KORD 171606Z 05009KT 10SM BKN017 M05/M10 A3008 RMK AO2';
      const decoded = decodeMetar(coldMetar);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(false);
      expect(result.reason).toBe('Temperature 23°F');
    });

    test('rejects precipitation conditions with specific messages', () => {
      // Test rain rejection
      const rainyMetar = 'KJFK 171551Z 20009KT 4SM RA BKN030 BKN250 20/18 A2992';
      const rainDecoded = decodeMetar(rainyMetar);
      const rainResult = isBikeable(rainDecoded);
      expect(rainResult.bikeable).toBe(false);
      expect(rainResult.reason).toBe('Rain expected');

      // Test snow rejection  
      const snowyMetar = 'KJFK 171551Z 20009KT 4SM SN BKN030 BKN250 20/18 A2992';
      const snowDecoded = decodeMetar(snowyMetar);
      const snowResult = isBikeable(snowDecoded);
      expect(snowResult.bikeable).toBe(false);
      expect(snowResult.reason).toBe('Snow expected');

      // Test thunderstorm rejection
      const thunderMetar = 'KJFK 171551Z 20009KT 4SM TS BKN030 BKN250 20/18 A2992';
      const thunderDecoded = decodeMetar(thunderMetar);
      const thunderResult = isBikeable(thunderDecoded);
      expect(thunderResult.bikeable).toBe(false);
      expect(thunderResult.reason).toBe('Thunderstorms expected');
    });

    test('rejects ice conditions', () => {
      const icyMetar = 'KJFK 171551Z 20009KT 10SM FZ BKN030 BKN250 M02/M05 A2992';
      const decoded = decodeMetar(icyMetar);
      const result = isBikeable(decoded);
      expect(result.bikeable).toBe(false);
      expect(result.reason).toBe('Ice conditions');
    });

    test('handles null/invalid data', () => {
      expect(isBikeable(null)).toBe(false);
      expect(isBikeable({})).toBe(false);
    });
  });
});