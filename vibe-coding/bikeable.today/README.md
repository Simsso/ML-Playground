# Can I Bike Today

## Brainstorming

### UX & UI of the MVP

**Core Concept:**
- Fixed location: Manhattan only (keeps MVP simple)
- Clear, binary indication of bikeability 
- Focus on visual appeal and immediate clarity
- NOT a weather app - just a simple yes/no decision tool

**Key Requirements:**
- Big, bold answer: "YES, BIKE TODAY" or "NO, TAKE THE SUBWAY" (or "YES, BIKE TOMORROW" if after 6pm)
- Clear indication if showing today vs tomorrow's forecast
- Weather-themed color coding (sunny yellows/greens vs stormy blues/grays)
- Clean, mobile-first design
- Minimal information - just the decision + explanation for "no" decisions
- Simple layout with location badge, date, main verdict, and brief reasoning when needed
- Console-style raw METAR/TAF display at bottom (adds tech appeal without affecting main UX)

**Style Direction:**
- Apple Weather app simplicity
- Clean typography and minimal layout
- Subtle animations
- Touch of NYC identity (yellow accents, bike lane vibes)

### Business Logic of the MVP

**Time-based Logic:**

| Time Window | Start | End | Commute(s) Checked | Data Sources |
|-------------|-------|-----|-------------------|--------------|
| Morning | 7:00am | 10:00am | Both (to work + return home) | METAR (current) for 9am commute + TAF for 6:30pm return |
| Afternoon | 10:00am | 3:00pm | Return home only | TAF for 6:30pm return |
| Late Afternoon | 3:00pm | 7:00pm | Return home only | METAR (current) for 6:30pm return |
| Evening/Night | 7:00pm | 7:00am | Both tomorrow (to work + return home) | TAF for tomorrow 9am commute + TAF for tomorrow 6:30pm return |

**Ride Times:**
- Morning commute: 9:00am
- Return commute: 6:30pm

**Forecast Requirements:**
- Two-part analysis: morning commute (8-10am) + evening return (5-7pm)
- Both parts must be bikeable for overall "YES" decision
- If either part fails, show "NO" with explanation

**Bikeability Thresholds:**
- Temperature: Must be 60°F - 90°F (outside range = not bikeable)
- Precipitation: Any rain, drizzle, or snow = not bikeable
- Ice conditions: Any ice/freezing = not bikeable

**Decision Output:**
- "YES" decisions: No explanation needed
- "NO" decisions: Brief explanation (e.g., "Too cold in morning", "Rain expected", "Ice conditions")

**Multi-Airport Consensus Logic:**
- Use METAR (current conditions) + TAF (forecasts) from LGA, JFK, EWR
- Decode aviation weather data for each airport
- Decision matrix:
  - All 3 airports good: Clear "YES"
  - 2 airports good, 1 bad: "MAYBE" with explanation
  - 1 airport good, 2 bad: "NO" with explanation
  - All 3 airports bad: Clear "NO"

### Tech Stack for the MVP

**Frontend:**
- Vite + React: Modern, fast setup, familiar syntax
- Responsive design with mobile-first approach

**Weather Data:**
- NOAA Aviation Weather Center (METAR + TAF)
- Sources: 
  - METAR: `https://tgftp.nws.noaa.gov/data/observations/metar/stations/KJFK.TXT`
  - TAF: `https://tgftp.nws.noaa.gov/data/forecasts/taf/stations/KJFK.TXT`
- Three airports: LGA, JFK, EWR for Manhattan coverage
- No API key required, reliable government data
- Custom decoder for aviation weather format
- Local Express proxy server to handle CORS restrictions

**Example Data:**
- METAR: `KJFK 171551Z 20009KT 10SM FEW030 FEW090 BKN250 28/22 A2992 RMK AO2 SLP131 T02830217 $`
- TAF: `AMD KJFK 171435Z 1715/1818 21013KT P6SM FEW030 SCT060 BKN250 FM171900 18017G21KT P6SM BKN050`

## Development

### Running the App
```bash
# Install dependencies
npm install

# Start both proxy server and Vite dev server
npm run dev:full

# Or run separately:
npm run proxy    # Starts Express proxy server on :3001
npm run dev      # Starts Vite dev server on :5173
```

### Testing
```bash
npm run test      # Run tests in watch mode
npm run test:run  # Run tests once
```

**Note:** The proxy server is required to bypass CORS restrictions when fetching METAR/TAF data from NOAA servers. It includes a 15-minute cache to reduce load on NOAA servers.

## Deployment

### Production Deployment (Google Cloud Platform)

**Architecture:**
- **Static Files**: Google Cloud Storage bucket with CDN
- **API Proxy**: Google Cloud Functions (serverless)
- **Domain**: canibike.today with HTTPS

**Deployment Plan:**
1. **Cloud Functions**: Convert `proxy-server.js` to Cloud Functions
   - `/metar/{station}` endpoint 
   - `/taf/{station}` endpoint
   - Maintains 15-minute in-memory caching
   - Handles CORS for browser requests

2. **Cloud Storage**: Host static build files
   - `gs://bikeable-today-static` bucket
   - Enable public access and CDN
   - Configure CORS for API calls

3. **Frontend Configuration**: Update API endpoints to Cloud Function URLs
   - Replace `localhost:3001` with Cloud Function trigger URLs
   - Build and upload static files to bucket

**Required GCP Services:**
- Cloud Functions API (✓ enabled)
- Cloud Storage API (✓ enabled)
- Project: `timo-websites` (✓ configured)

**LIVE DEPLOYMENT:**
- **URL**: https://canibike.today
- **Bucket**: `gs://canibike.today`
- **Functions**: https://us-central1-timo-websites.cloudfunctions.net/
- **Status**: Deployed and live

### Architecture
- Aviation weather parsing/decoding library
- Multi-airport data aggregation logic
- Error handling for weather service outages
- Time zone handling (NYC/aviation times)