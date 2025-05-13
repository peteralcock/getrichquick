# NFL Prop Bet Generator 
## CURRENTLY STILL IN DEVELOPMENT
![GetRichQuick](getrichquick.png?raw=true "getrichquick")

This repository provides an AI-driven tool for generating creative NFL prop bets based on real-time NFL news headlines. The system uses RSS feeds, natural language processing (NLP), sentiment analysis, and OpenAI's API to produce prop bets related to NFL players, with accompanying odds and betting recommendations. This system now uses improved artificial intelligence to analyze NFL news headlines, identify player trends, and generate intelligent prop betting recommendations with continuous self-improvement capabilities.


## NEW FEATURES

### 1. User Interface with Point-and-Click Bet Generation
- Modern web interface built with Flask and Bootstrap
- Dashboard displaying player data, news headlines, and betting performance
- One-click prop bet generation from any headline
- Detailed player profiles with performance analytics
- Interactive visualizations of betting success over time

### 2. Bet Accuracy Tracking
- Automated verification of bet outcomes
- Performance analytics by prop type, player position, and team
- Confidence calibration to improve prediction accuracy
- Historical tracking of all bets with success/failure rates
- Visual dashboards showing AI's prediction accuracy over time

### 3. Personal Event Tracking
- Automatic detection of personal events in news headlines
- Categorization by event type (relationship, legal, injury, etc.)
- Correlation analysis between personal events and on-field performance
- Integration of personal event data into betting predictions
- Ongoing learning about event impact through reinforcement

### 4. Self-Assessment Model Training Engine
- Machine learning pipeline for continuous model improvement
- Feature extraction from headlines, player data, and betting history
- Automated model retraining based on betting outcomes
- Performance comparison across model versions
- Advanced pattern recognition for betting success factors

### 5. Betting Performance Tracking
- Comprehensive logging of all generated bets
- Detailed reasoning and confidence for each prediction
- Success rate analytics across different dimensions
- Statistical analysis of factors contributing to accurate predictions
- Trend identification for betting strategy optimization

### 6. Feedback & Reinforcement Learning
- Automated verification of game results
- Detailed explanation logging for each prediction
- Pattern discovery in successful vs. unsuccessful bets
- Bayesian updating of confidence calibration
- Continuous improvement through outcome tracking

### 7. LangChain Integration
- Structured reasoning chains for bet generation
- Named entity recognition for player identification
- Sentiment analysis of news headlines
- Knowledge graph integration for player relationships
- Context-aware prompt templates for consistent outputs

## System Architecture

### Database Schema
The system uses SQLite with the following core tables:
- `players`: Player profiles with metadata and sentiment scores
- `headlines`: News headlines with sentiment analysis and sourcing
- `prop_bets`: Generated betting propositions with predictions and outcomes
- `personal_events`: Personal events detected for players
- `betting_patterns`: Discovered patterns in successful betting
- `model_performance`: Historical performance of ML models
- `bet_explanations`: Detailed reasoning behind each prediction

### Key Components

#### Data Collection
- RSS feed parsing from 20+ NFL news sources
- Named entity recognition to extract player mentions
- Sentiment analysis of headlines
- Team and position inference for players

#### Bet Generation
- LangChain-powered reasoning for prop bet creation
- Dynamic odds calculation based on confidence
- Multi-factor analysis incorporating:
  - Headline sentiment
  - Player history
  - Personal events
  - Position-specific patterns
  - Team context

#### Machine Learning Pipeline
- Random Forest classifier for outcome prediction
- Feature engineering from headline data
- Model version tracking and comparison
- Calibration of confidence scores
- Advanced pattern discovery

#### Visualization & Analytics
- Success rate trends over time
- Confidence vs. actual success rate analysis
- Position-specific performance insights
- Personal event impact visualization
- Betting pattern effectiveness charts

## Getting Started

### Prerequisites
- Python 3.8+
- Flask
- LangChain
- OpenAI API key (for LLM access)
- spaCy with English model
- pandas, matplotlib, seaborn for analytics
- scikit-learn for ML components

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your OpenAI API key: `export OPENAI_API_KEY="your-api-key"`
4. Initialize the database: `python app.py init_db`
5. Run the application: `python app.py`

### Usage
1. Access the web interface at `http://localhost:5000`
2. Click "Update News" to fetch the latest NFL headlines
3. Browse player profiles or view the analytics dashboard
4. Click "Generate Bet" on any headline to create prop bets
5. View betting history and system performance in the Analytics section

## System Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  RSS News Feeds │────▶│ NLP Processing   │────▶│ Sentiment Analysis│
└─────────────────┘     │ (spaCy + LangChain)    └───────────────────┘
                        └──────────────────┘              │
                                │                         ▼
                                ▼                  ┌───────────────┐
┌─────────────────┐     ┌──────────────────┐      │ Personal Event │
│  OpenAI API     │◀───▶│ Prop Bet         │◀────▶│ Detection      │
└─────────────────┘     │ Generation       │      └───────────────┘
                        └──────────────────┘              │
                                │                         │
                                ▼                         ▼
                        ┌──────────────────┐     ┌───────────────────┐
                        │ Database Storage │────▶│ ML Model Training │
                        └──────────────────┘     └───────────────────┘
                                │                         │
                                ▼                         ▼
                        ┌──────────────────┐     ┌───────────────────┐
                        │ Flask Web        │     │ Pattern Discovery │
                        │ Interface        │◀───▶│ & Analysis        │
                        └──────────────────┘     └───────────────────┘
```

## Key Files

- `app.py`: Main application with Flask routes and core functionality
- `templates/`: HTML templates for web interface
- `static/`: CSS, JavaScript, and generated visualizations
- `bet_prediction_model.pkl`: Serialized ML model for prediction
- `calibration_model.json`: Confidence calibration data

## TODO

1. **Expanded Data Sources**: Integration with player statistics APIs and historical game data
2. **Real-time Updating**: WebSocket implementation for live odds updates during games
3. **Multi-sport Expansion**: Extend to NBA, MLB, and other sports leagues
4. **User Customization**: Allow users to adjust confidence thresholds and betting preferences
5. **Advanced Visualization**: Interactive exploration of betting patterns and correlations




## V1 Features

- **Headline Aggregation**: Gathers NFL news headlines from a list of top RSS feeds.
- **Player Identification**: Uses NLP (spaCy) to extract player names from headlines.
- **Sentiment Analysis**: Analyzes sentiment in each headline related to the player to calculate an overall sentiment score.
- **AI-Generated Prop Bets**: Creates unique prop bets using OpenAI's API based on each headline.
- **Database Storage**: Stores player data, headlines, sentiment scores, and generated prop bets in an SQLite database.

## Installation (V1)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/nfl-prop-bet-generator.git
   cd nfl-prop-bet-generator
   ```

2. **Install required dependencies**:
   Ensure you have Python 3.x installed, then run:
   ```bash
   pip install feedparser spacy textblob openai
   python -m spacy download en_core_web_sm
   ```

3. **Set up OpenAI API key**:
   Replace `sk-proj-3c39_W2Vsa...` with your own OpenAI API key in the `client` initialization.

## Usage  (V1)

1. **Run the script**:
   Execute the script to fetch headlines, process player names, analyze sentiment, and generate prop bets.
   ```bash
   python main.py
   ```

2. **Check Output**:
   - **Player profiles**: Displays each player's headlines and overall sentiment score.
   - **Generated Prop Bets**: Prop bets generated by the AI are stored in the SQLite database (`nfl_players.db`), under the `prop_bets` table.

## Database Structure  (V1)

- **`players` Table**: Contains unique player IDs and aggregated sentiment scores.
- **`headlines` Table**: Stores headlines associated with each player.
- **`prop_bets` Table**: Contains AI-generated prop bets linked to each player and headline.

## Example Workflow  (V1)

1. **Fetch Headlines**: Retrieves NFL headlines from specified RSS feeds.
2. **Identify Players**: Extracts player names using spaCy's Named Entity Recognition.
3. **Sentiment Analysis**: Computes a sentiment score for each player based on their headlines.
4. **Generate Prop Bets**: Uses OpenAI API to generate three creative prop bets for each headline related to a player.
5. **Save Results**: Stores data in an SQLite database.

## Warning

This tool is intended for entertainment purposes only and should not be used as a primary betting guide. Gambling carries financial risks, and prop bets generated are speculative.

## License

This project is licensed under the MIT License.
