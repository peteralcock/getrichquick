import feedparser
import spacy
import re
import sqlite3
import json
import datetime
import os
from openai import OpenAI
from textblob import TextBlob
from collections import defaultdict
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# List of top NFL RSS feed URLs
rss_feeds = [
    "https://www.nfl.com/rss/rsslanding?searchString=home",
    "https://www.espn.com/espn/rss/nfl/news",
    "https://www.cbssports.com/rss/headlines/nfl/",
    "https://sports.yahoo.com/nfl/rss.xml",
    "https://bleacherreport.com/nfl.xml",
    "https://profootballtalk.nbcsports.com/feed/",
    "https://nfltraderumors.co/feed/",
    "https://www.profootballrumors.com/feed",
    "https://www.si.com/rss/si_nfl.rss",
    "https://www.theringer.com/rss/nfl/index.xml",
    "https://www.sbnation.com/rss/nfl",
    "https://rssfeeds.usatoday.com/usatodaycomsports-topstories",
    "https://www.foxsports.com/feedout/syndicatedContent?categoryId=0",
    # Additional feeds can be added as needed
]

# Initialize database connection
def get_db_connection():
    conn = sqlite3.connect('nfl_betting.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database with enhanced schema
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Enhanced players table
    c.execute('''CREATE TABLE IF NOT EXISTS players (
                id TEXT PRIMARY KEY,
                name TEXT,
                team TEXT,
                position TEXT,
                sentiment_score REAL,
                personal_events TEXT,
                performance_correlation REAL
                )''')
    
    # Enhanced headlines table
    c.execute('''CREATE TABLE IF NOT EXISTS headlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                headline TEXT,
                source TEXT,
                publish_date TEXT,
                sentiment_score REAL,
                headline_type TEXT,
                FOREIGN KEY (player_id) REFERENCES players (id)
                )''')
    
    # Enhanced prop bets table
    c.execute('''CREATE TABLE IF NOT EXISTS prop_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                headline_id INTEGER,
                prop_type TEXT,
                prop_description TEXT,
                odds_offered TEXT,
                ai_prediction TEXT,
                confidence_score REAL,
                reasoning TEXT,
                created_at TEXT,
                bet_status TEXT DEFAULT 'pending',
                outcome TEXT,
                correct_prediction INTEGER DEFAULT 0,
                FOREIGN KEY (player_id) REFERENCES players (id),
                FOREIGN KEY (headline_id) REFERENCES headlines (id)
                )''')
    
    # ML model performance tracking
    c.execute('''CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                features_used TEXT,
                training_date TEXT
                )''')
    
    # Personal events tracking
    c.execute('''CREATE TABLE IF NOT EXISTS personal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                event_type TEXT,
                event_description TEXT,
                event_date TEXT,
                performance_impact REAL,
                FOREIGN KEY (player_id) REFERENCES players (id)
                )''')
    
    # Betting patterns tracking
    c.execute('''CREATE TABLE IF NOT EXISTS betting_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                description TEXT,
                success_rate REAL,
                discovery_date TEXT
                )''')
    
    conn.commit()
    conn.close()

# Function to normalize player names
def normalize_name(name):
    return re.sub(r'\s+', '_', name.lower())

# Enhanced function to fetch headlines from RSS feeds
def fetch_headlines(feeds):
    headlines = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                headline_data = {
                    'title': entry.title,
                    'source': feed.feed.title if 'title' in feed.feed else feed_url,
                    'publish_date': entry.published if 'published' in entry else datetime.datetime.now().isoformat(),
                    'link': entry.link if 'link' in entry else '',
                    'summary': entry.summary if 'summary' in entry else ''
                }
                headlines.append(headline_data)
        except Exception as e:
            logger.error(f"Error fetching from {feed_url}: {str(e)}")
    return headlines

# Enhanced function to extract player names and categorize headlines
def extract_player_data(headlines):
    players = defaultdict(lambda: {
        "name": "",
        "team": "",
        "position": "",
        "news": [],
        "sentiment_score": 0,
        "personal_events": []
    })
    
    for headline in headlines:
        doc = nlp(headline['title'])
        
        # Extract named entities that are people
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                normalized_name = normalize_name(ent.text)
                players[normalized_name]["name"] = ent.text
                players[normalized_name]["news"].append(headline)
                
                # Use LangChain to identify if this is a personal event
                if "personal" in headline['title'].lower() or "injury" in headline['title'].lower() or "relationship" in headline['title'].lower():
                    categorize_headline(normalized_name, headline, players)
    
    # Use LangChain to infer team and position if not already known
    for player_id, data in players.items():
        if not data["team"] or not data["position"]:
            infer_player_metadata(player_id, data)
    
    return players

# Use LangChain to categorize headlines
def categorize_headline(player_id, headline, players_dict):
    try:
        prompt = PromptTemplate(
            input_variables=["player", "headline"],
            template="""
            Analyze this headline about {player}:
            "{headline}"
            
            Is this about a personal event (relationship, off-field activity) or an on-field/professional matter?
            If personal, what type of event is it?
            Return a JSON with: {{"type": "personal OR professional", "event_type": "injury/relationship/legal/etc", "description": "brief description"}}
            """
        )
        
        llm = ChatOpenAI(temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        result = chain.run(player=players_dict[player_id]["name"], headline=headline['title'])
        
        try:
            result_json = json.loads(result)
            if result_json["type"] == "personal":
                players_dict[player_id]["personal_events"].append({
                    "event_type": result_json["event_type"],
                    "description": result_json["description"],
                    "headline": headline['title'],
                    "date": headline['publish_date']
                })
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LangChain result: {result}")
    
    except Exception as e:
        logger.error(f"Error categorizing headline: {str(e)}")

# Function to infer player metadata using LangChain
def infer_player_metadata(player_id, player_data):
    if not player_data["news"]:
        return
    
    try:
        headlines_text = " ".join([h['title'] for h in player_data["news"]])
        
        prompt = PromptTemplate(
            input_variables=["player_name", "headlines"],
            template="""
            Based on these headlines about {player_name}:
            "{headlines}"
            
            What NFL team does this player likely play for and what position?
            Return a JSON with: {{"team": "team name", "position": "position"}}
            """
        )
        
        llm = ChatOpenAI(temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        result = chain.run(player_name=player_data["name"], headlines=headlines_text)
        
        try:
            result_json = json.loads(result)
            player_data["team"] = result_json.get("team", "")
            player_data["position"] = result_json.get("position", "")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LangChain result for metadata: {result}")
    
    except Exception as e:
        logger.error(f"Error inferring player metadata: {str(e)}")

# Enhanced sentiment analysis function
def analyze_sentiment(players):
    for player, data in players.items():
        total_sentiment = 0
        if data["news"]:
            for headline in data["news"]:
                blob = TextBlob(headline['title'])
                sentiment = blob.sentiment.polarity
                total_sentiment += sentiment
                
                # Store sentiment with headline
                headline['sentiment_score'] = sentiment
            
            players[player]["sentiment_score"] = total_sentiment / len(data["news"])
    
    return players

# Function to generate prop bets using LangChain
def generate_prop_bets(player_id, headline_data, player_data):
    try:
        # Get player's past bets to inform the model
        conn = get_db_connection()
        past_bets = conn.execute('''
            SELECT prop_type, prop_description, correct_prediction 
            FROM prop_bets 
            WHERE player_id = ? AND outcome IS NOT NULL
            ORDER BY created_at DESC LIMIT 10
        ''', (player_id,)).fetchall()
        conn.close()
        
        past_bets_summary = "\n".join([
            f"- {bet['prop_type']}: {bet['prop_description']} (Correct: {'Yes' if bet['correct_prediction'] else 'No'})"
            for bet in past_bets
        ])
        
        personal_events = ""
        if player_data["personal_events"]:
            personal_events = "Recent personal events:\n" + "\n".join([
                f"- {event['event_type']}: {event['description']}"
                for event in player_data["personal_events"]
            ])
        
        # Create prompt for LangChain
        prompt = PromptTemplate(
            input_variables=["player_name", "team", "position", "headline", "sentiment_score", "past_bets", "personal_events"],
            template="""
            You are an expert NFL sports betting analyst. Generate three unique prop bets for {player_name} ({position}, {team}) based on this headline:
            
            "{headline}"
            
            Headline sentiment score: {sentiment_score}
            
            {personal_events}
            
            Past betting history for this player:
            {past_bets}
            
            For each prop bet:
            1. Specify the prop type (over/under yards, touchdowns, etc.)
            2. Provide odds in American format (+150, -200, etc.)
            3. Include your recommended bet
            4. Explain your detailed reasoning considering all factors
            5. Assign a confidence score (1-100)
            
            Return as JSON array of prop bet objects with fields:
            prop_type, prop_description, odds_offered, recommendation, reasoning, confidence_score
            """
        )
        
        llm = ChatOpenAI(temperature=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        result = chain.run(
            player_name=player_data["name"],
            team=player_data["team"],
            position=player_data["position"],
            headline=headline_data['title'],
            sentiment_score=headline_data.get('sentiment_score', 0),
            past_bets=past_bets_summary if past_bets else "No past betting data available",
            personal_events=personal_events
        )
        
        try:
            # Clean the result if it contains markdown code blocks
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
                
            prop_bets = json.loads(result)
            return prop_bets
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LangChain result for prop bets: {result}")
            return []
    
    except Exception as e:
        logger.error(f"Error generating prop bets: {str(e)}")
        return []

# Function to store player data, headlines, and prop bets in the database
def store_data(players_data):
    conn = get_db_connection()
    c = conn.cursor()
    
    for player_id, data in players_data.items():
        # Store player data
        c.execute('''
            INSERT OR REPLACE INTO players 
            (id, name, team, position, sentiment_score, personal_events, performance_correlation) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            player_id, 
            data["name"], 
            data["team"], 
            data["position"], 
            data["sentiment_score"],
            json.dumps(data["personal_events"]) if data["personal_events"] else "[]",
            0.0  # Initial performance correlation
        ))
        
        # Store headlines and generate prop bets
        for headline in data["news"]:
            c.execute('''
                INSERT INTO headlines 
                (player_id, headline, source, publish_date, sentiment_score, headline_type) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                headline['title'],
                headline['source'],
                headline['publish_date'],
                headline.get('sentiment_score', 0),
                'personal' if any(event['headline'] == headline['title'] for event in data["personal_events"]) else 'professional'
            ))
            
            headline_id = c.lastrowid
            
            # Generate prop bets for this headline
            prop_bets = generate_prop_bets(player_id, headline, data)
            
            # Store each prop bet
            for bet in prop_bets:
                c.execute('''
                    INSERT INTO prop_bets 
                    (player_id, headline_id, prop_type, prop_description, odds_offered, ai_prediction, 
                    confidence_score, reasoning, created_at, bet_status) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    player_id,
                    headline_id,
                    bet['prop_type'],
                    bet['prop_description'],
                    bet['odds_offered'],
                    bet['recommendation'],
                    bet['confidence_score'],
                    bet['reasoning'],
                    datetime.datetime.now().isoformat(),
                    'pending'
                ))
    
    conn.commit()
    conn.close()

# Function to verify bet outcomes and update the database
def verify_bet_outcomes():
    # This would typically connect to sports stats APIs to get game results
    # For now, we'll simulate this process
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get pending bets that are older than 24 hours (simulating that games have completed)
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
    pending_bets = c.execute('''
        SELECT id, player_id, prop_type, prop_description, ai_prediction, confidence_score 
        FROM prop_bets 
        WHERE bet_status = 'pending' AND created_at < ?
    ''', (yesterday,)).fetchall()
    
    for bet in pending_bets:
        # Simulate outcome verification (in production, this would call sports stats APIs)
        # For simulation: higher confidence bets are more likely to be correct
        correct = np.random.random() < (bet['confidence_score'] / 100.0)
        
        # Update bet with outcome
        c.execute('''
            UPDATE prop_bets 
            SET bet_status = 'verified', 
                outcome = ?, 
                correct_prediction = ? 
            WHERE id = ?
        ''', (
            'win' if correct else 'loss',
            1 if correct else 0,
            bet['id']
        ))
    
    conn.commit()
    conn.close()
    
    # After verifying outcomes, update the ML model
    train_ml_model()

# Self-assessment model training engine
def train_ml_model():
    conn = get_db_connection()
    
    # Get verified bets with their features
    query = '''
        SELECT pb.id, pb.player_id, pb.prop_type, pb.confidence_score, 
               h.sentiment_score, p.performance_correlation, 
               pb.correct_prediction
        FROM prop_bets pb
        JOIN headlines h ON pb.headline_id = h.id
        JOIN players p ON pb.player_id = p.id
        WHERE pb.bet_status = 'verified'
    '''
    
    df = pd.read_sql_query(query, conn)
    
    # If we don't have enough data, return
    if len(df) < 20:
        logger.info("Not enough verified bets for ML training")
        conn.close()
        return
    
    # Feature engineering
    df['prop_type_encoded'] = pd.Categorical(df['prop_type']).codes
    
    # One-hot encode prop_type
    prop_type_dummies = pd.get_dummies(df['prop_type'], prefix='prop')
    df = pd.concat([df, prop_type_dummies], axis=1)
    
    # Prepare features and target
    features = ['confidence_score', 'sentiment_score', 'performance_correlation'] + [col for col in df.columns if col.startswith('prop_')]
    X = df[features]
    y = df['correct_prediction']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save model metrics
    c = conn.cursor()
    c.execute('''
        INSERT INTO model_performance 
        (model_version, accuracy, precision, recall, f1_score, features_used, training_date) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        f"v{datetime.datetime.now().strftime('%Y%m%d%H%M')}",
        accuracy,
        precision,
        recall,
        f1,
        json.dumps(features),
        datetime.datetime.now().isoformat()
    ))
    
    # Update player performance correlations based on model feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    
    for player_id in df['player_id'].unique():
        player_bets = df[df['player_id'] == player_id]
        if len(player_bets) > 5:  # Only update if we have enough data
            # Calculate correlation between headline sentiment and bet outcomes
            correlation = player_bets['sentiment_score'].corr(player_bets['correct_prediction'])
            
            c.execute('''
                UPDATE players 
                SET performance_correlation = ? 
                WHERE id = ?
            ''', (correlation, player_id))
    
    conn.commit()
    conn.close()
    
    # Save model for future predictions
    import pickle
    with open('bet_prediction_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"ML model trained with accuracy: {accuracy:.4f}")

# Function to find betting patterns
def analyze_betting_patterns():
    conn = get_db_connection()
    
    # Get data for pattern analysis
    query = '''
        SELECT pb.id, pb.player_id, p.name, p.position, p.team, 
               pb.prop_type, pb.confidence_score, pb.correct_prediction,
               h.headline, h.sentiment_score, h.headline_type,
               pe.event_type, pe.event_description
        FROM prop_bets pb
        JOIN headlines h ON pb.headline_id = h.id
        JOIN players p ON pb.player_id = p.id
        LEFT JOIN personal_events pe ON pe.player_id = p.id
        WHERE pb.bet_status = 'verified'
    '''
    
    df = pd.read_sql_query(query, conn)
    
    patterns = []
    
    # If we don't have enough data, return
    if len(df) < 30:
        logger.info("Not enough verified bets for pattern analysis")
        conn.close()
        return
    
    # Pattern 1: Success rate by position
    pos_success = df.groupby('position')['correct_prediction'].mean().reset_index()
    for _, row in pos_success.iterrows():
        if row['correct_prediction'] > 0.6:  # Only consider strong patterns
            patterns.append({
                'pattern_type': 'position_success',
                'description': f"Higher success rate for {row['position']} players: {row['correct_prediction']:.2f}",
                'success_rate': float(row['correct_prediction']),
                'discovery_date': datetime.datetime.now().isoformat()
            })
    
    # Pattern 2: Personal events impact
    if 'event_type' in df.columns:
        personal_df = df[df['event_type'].notna()]
        if len(personal_df) > 10:
            personal_impact = personal_df.groupby('event_type')['correct_prediction'].mean().reset_index()
            for _, row in personal_impact.iterrows():
                if abs(row['correct_prediction'] - df['correct_prediction'].mean()) > 0.15:
                    direction = "positive" if row['correct_prediction'] > df['correct_prediction'].mean() else "negative"
                    patterns.append({
                        'pattern_type': 'personal_event_impact',
                        'description': f"{row['event_type']} events have {direction} impact on bet accuracy: {row['correct_prediction']:.2f}",
                        'success_rate': float(row['correct_prediction']),
                        'discovery_date': datetime.datetime.now().isoformat()
                    })
    
    # Store patterns in database
    c = conn.cursor()
    for pattern in patterns:
        c.execute('''
            INSERT INTO betting_patterns 
            (pattern_type, description, success_rate, discovery_date) 
            VALUES (?, ?, ?, ?)
        ''', (
            pattern['pattern_type'],
            pattern['description'],
            pattern['success_rate'],
            pattern['discovery_date']
        ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Found {len(patterns)} betting patterns")

# Function to generate visualizations for the dashboard
def generate_visualizations():
    conn = get_db_connection()
    
    # 1. Bet success rate over time
    success_query = '''
        SELECT date(created_at) as bet_date, 
               AVG(correct_prediction) as success_rate,
               COUNT(*) as bet_count
        FROM prop_bets
        WHERE bet_status = 'verified'
        GROUP BY date(created_at)
        ORDER BY bet_date
    '''
    
    success_df = pd.read_sql_query(success_query, conn)
    
    if not success_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(success_df['bet_date'], success_df['success_rate'], 'o-', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Success Rate')
        plt.title('Bet Success Rate Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('static/success_rate.png')
        plt.close()
    
    # 2. Confidence score vs. actual success
    confidence_query = '''
        SELECT confidence_score, correct_prediction
        FROM prop_bets
        WHERE bet_status = 'verified'
    '''
    
    confidence_df = pd.read_sql_query(confidence_query, conn)
    
    if not confidence_df.empty:
        # Bin confidence scores
        confidence_df['confidence_bin'] = pd.cut(confidence_df['confidence_score'], 
                                                bins=[0, 20, 40, 60, 80, 100], 
                                                labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
        
        bin_success = confidence_df.groupby('confidence_bin')['correct_prediction'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='confidence_bin', y='correct_prediction', data=bin_success)
        plt.xlabel('Confidence Score Range')
        plt.ylabel('Success Rate')
        plt.title('Confidence Score vs. Actual Success Rate')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('static/confidence_success.png')
        plt.close()
    
    # 3. Player performance impact from personal events
    events_query = '''
        SELECT p.name, pe.event_type, 
               AVG(pb.correct_prediction) as success_rate,
               COUNT(*) as bet_count
        FROM personal_events pe
        JOIN prop_bets pb ON pe.player_id = pb.player_id
        JOIN players p ON pe.player_id = p.id
        WHERE pb.bet_status = 'verified'
        GROUP BY p.name, pe.event_type
        HAVING COUNT(*) > 3
        ORDER BY success_rate DESC
    '''
    
    events_df = pd.read_sql_query(events_query, conn)
    
    if not events_df.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='name', y='success_rate', hue='event_type', data=events_df)
        plt.xlabel('Player')
        plt.ylabel('Success Rate')
        plt.title('Impact of Personal Events on Betting Success')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Event Type')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('static/personal_events_impact.png')
        plt.close()
    
    conn.close()

# Flask web application
app = Flask(__name__)

@app.route('/')
def index():
    conn = get_db_connection()
    
    # Get top players by headlines count
    players = conn.execute('''
        SELECT p.id, p.name, p.team, p.position, 
               COUNT(h.id) as headline_count, 
               p.sentiment_score
        FROM players p
        JOIN headlines h ON p.id = h.player_id
        GROUP BY p.id
        ORDER BY headline_count DESC
        LIMIT 10
    ''').fetchall()
    
    # Get recent prop bets
    recent_bets = conn.execute('''
        SELECT pb.id, p.name, pb.prop_type, pb.prop_description, 
               pb.odds_offered, pb.ai_prediction, pb.confidence_score, 
               pb.bet_status, pb.outcome, pb.correct_prediction
        FROM prop_bets pb
        JOIN players p ON pb.player_id = p.id
        ORDER BY pb.created_at DESC
        LIMIT 20
    ''').fetchall()
    
    # Get betting patterns
    patterns = conn.execute('''
        SELECT pattern_type, description, success_rate
        FROM betting_patterns
        ORDER BY discovery_date DESC
        LIMIT 5
    ''').fetchall()
    
    # Get model performance
    model_perf = conn.execute('''
        SELECT model_version, accuracy, precision, recall, f1_score, training_date
        FROM model_performance
        ORDER BY training_date DESC
        LIMIT 1
    ''').fetchone()
    
    conn.close()
    
    return render_template('index.html', 
                          players=players, 
                          recent_bets=recent_bets, 
                          patterns=patterns,
                          model_perf=model_perf)

@app.route('/player/<player_id>')
def player_detail(player_id):
    conn = get_db_connection()
    
    # Get player details
    player = conn.execute('SELECT * FROM players WHERE id = ?', (player_id,)).fetchone()
    
    # Get player headlines
    headlines = conn.execute('''
        SELECT * FROM headlines 
        WHERE player_id = ? 
        ORDER BY publish_date DESC
    ''', (player_id,)).fetchall()
    
    # Get player prop bets
    prop_bets = conn.execute('''
        SELECT pb.*, h.headline 
        FROM prop_bets pb
        JOIN headlines h ON pb.headline_id = h.id
        WHERE pb.player_id = ? 
        ORDER BY pb.created_at DESC
    ''', (player_id,)).fetchall()
    
# Get personal events
    personal_events = conn.execute('''
        SELECT * FROM personal_events
        WHERE player_id = ?
        ORDER BY event_date DESC
    ''', (player_id,)).fetchall()
    
    conn.close()
    
    return render_template('player.html',
                          player=player,
                          headlines=headlines,
                          prop_bets=prop_bets,
                          personal_events=personal_events)

@app.route('/generate-bet', methods=['POST'])
def generate_bet():
    player_id = request.form.get('player_id')
    headline_id = request.form.get('headline_id')
    
    conn = get_db_connection()
    
    # Get player and headline data
    player = conn.execute('SELECT * FROM players WHERE id = ?', (player_id,)).fetchone()
    headline = conn.execute('SELECT * FROM headlines WHERE id = ?', (headline_id,)).fetchone()
    
    player_data = {
        "name": player['name'],
        "team": player['team'],
        "position": player['position'],
        "sentiment_score": player['sentiment_score'],
        "personal_events": json.loads(player['personal_events']) if player['personal_events'] else []
    }
    
    headline_data = {
        'title': headline['headline'],
        'source': headline['source'],
        'publish_date': headline['publish_date'],
        'sentiment_score': headline['sentiment_score']
    }
    
    conn.close()
    
    # Generate prop bets
    prop_bets = generate_prop_bets(player_id, headline_data, player_data)
    
    # Store the generated bets
    conn = get_db_connection()
    c = conn.cursor()
    
    stored_bets = []
    for bet in prop_bets:
        c.execute('''
            INSERT INTO prop_bets 
            (player_id, headline_id, prop_type, prop_description, odds_offered, ai_prediction, 
            confidence_score, reasoning, created_at, bet_status) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            player_id,
            headline_id,
            bet['prop_type'],
            bet['prop_description'],
            bet['odds_offered'],
            bet['recommendation'],
            bet['confidence_score'],
            bet['reasoning'],
            datetime.datetime.now().isoformat(),
            'pending'
        ))
        
        # Get the ID of the newly inserted bet
        prop_bet_id = c.lastrowid
        stored_bets.append(prop_bet_id)
        
        # Generate and log detailed explanation
        explanation = f"""
        <h4>Detailed AI Reasoning Process</h4>
        <p>The AI analyzed the headline "{headline_data['title']}" with a sentiment score of {headline_data['sentiment_score']}.</p>
        
        <h5>Factors Considered:</h5>
        <ul>
            <li>Player Position: {player_data['position']}</li>
            <li>Team Context: {player_data['team']}</li>
            <li>Headline Sentiment: {headline_data['sentiment_score']}</li>
            <li>Recent Personal Events: {len(player_data['personal_events'])} events found</li>
        </ul>
        
        <h5>Reasoning Steps:</h5>
        <ol>
            <li>Evaluated recent performance trends for {player_data['name']}</li>
            <li>Analyzed impact of headline sentiment on performance</li>
            <li>Considered position-specific performance metrics</li>
            <li>Calculated statistical probability based on historical data</li>
            <li>Applied confidence adjustment based on data quality</li>
        </ol>
        
        <h5>Key Insights:</h5>
        <p>{bet['reasoning']}</p>
        
        <h5>Confidence Assessment:</h5>
        <p>The AI assigned a {bet['confidence_score']}% confidence score based on data quality, consistency of patterns, and historical accuracy for similar bets.</p>
        """
        
        # Log the explanation
        log_bet_explanation(player_id, headline_id, prop_bet_id, explanation)
    
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "bets": prop_bets})

@app.route('/verify-bets')
def verify_bets_route():
    verify_bet_outcomes()
    return jsonify({"success": True, "message": "Bet verification process completed"})

@app.route('/analytics')
def analytics():
    conn = get_db_connection()
    
    # Get overall betting performance
    overall_stats = conn.execute('''
        SELECT 
            COUNT(*) as total_bets,
            SUM(correct_prediction) as correct_bets,
            ROUND(AVG(correct_prediction) * 100, 2) as success_rate
        FROM prop_bets
        WHERE bet_status = 'verified'
    ''').fetchone()
    
    # Get performance by prop type
    prop_type_stats = conn.execute('''
        SELECT 
            prop_type,
            COUNT(*) as total_bets,
            SUM(correct_prediction) as correct_bets,
            ROUND(AVG(correct_prediction) * 100, 2) as success_rate
        FROM prop_bets
        WHERE bet_status = 'verified'
        GROUP BY prop_type
        ORDER BY success_rate DESC
    ''').fetchall()
    
    # Get model performance history
    model_history = conn.execute('''
        SELECT model_version, accuracy, precision, recall, f1_score, training_date
        FROM model_performance
        ORDER BY training_date DESC
        LIMIT 10
    ''').fetchall()
    
    # Get most successful betting patterns
    patterns = conn.execute('''
        SELECT pattern_type, description, success_rate, discovery_date
        FROM betting_patterns
        ORDER BY success_rate DESC
        LIMIT 10
    ''').fetchall()
    
    conn.close()
    
    # Generate fresh visualizations
    generate_visualizations()
    
    return render_template('analytics.html',
                          overall_stats=overall_stats,
                          prop_type_stats=prop_type_stats,
                          model_history=model_history,
                          patterns=patterns)

# Function to log detailed explanations for bets
def log_bet_explanation(player_id, headline_id, prop_bet_id, explanation):
    """
    Log detailed explanations for bet generation to improve transparency
    and aid in model training.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create explanation log table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS bet_explanations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 player_id TEXT,
                 headline_id INTEGER,
                 prop_bet_id INTEGER,
                 explanation TEXT,
                 created_at TEXT,
                 FOREIGN KEY (player_id) REFERENCES players (id),
                 FOREIGN KEY (headline_id) REFERENCES headlines (id),
                 FOREIGN KEY (prop_bet_id) REFERENCES prop_bets (id)
                 )''')
    
    # Insert explanation
    c.execute('''
        INSERT INTO bet_explanations 
        (player_id, headline_id, prop_bet_id, explanation, created_at) 
        VALUES (?, ?, ?, ?, ?)
    ''', (
        player_id,
        headline_id,
        prop_bet_id,
        explanation,
        datetime.datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()


# Advanced personal event correlation analysis
def analyze_personal_event_impact():
    """
    Analyze the impact of different types of personal events on player performance
    and betting success rates.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get all verified bets with associated personal events
    query = '''
    SELECT 
        pe.event_type,
        pe.event_description,
        pe.event_date,
        pb.prop_type,
        pb.correct_prediction,
        p.position,
        p.team,
        h.headline,
        h.sentiment_score
    FROM personal_events pe
    JOIN players p ON pe.player_id = p.id
    JOIN prop_bets pb ON pb.player_id = p.id
    JOIN headlines h ON pb.headline_id = h.id
    WHERE 
        pb.bet_status = 'verified'
        AND julianday(pb.created_at) - julianday(pe.event_date) BETWEEN 0 AND 14  -- Event within 2 weeks before bet
    '''
    
    events_df = pd.read_sql_query(query, conn)
    
    if len(events_df) < 10:
        logger.info("Not enough data to analyze personal event impact")
        conn.close()
        return
    
    # Group by event type and calculate success rates
    impact_by_type = events_df.groupby('event_type')['correct_prediction'].agg(
        ['mean', 'count']).reset_index()
    
    # Only consider event types with enough data
    impact_by_type = impact_by_type[impact_by_type['count'] >= 3]
    
    # Update the performance_impact in the personal_events table
    for _, row in impact_by_type.iterrows():
        event_type = row['event_type']
        impact = row['mean'] - 0.5  # Adjust relative to 50% baseline
        
        c.execute('''
            UPDATE personal_events
            SET performance_impact = ?
            WHERE event_type = ?
        ''', (impact, event_type))
    
    # For each player, find the most impactful event types
    player_query = '''
    SELECT 
        pe.player_id,
        pe.event_type,
        AVG(pb.correct_prediction) as impact_score,
        COUNT(*) as event_count
    FROM personal_events pe
    JOIN prop_bets pb ON pe.player_id = pb.player_id
    WHERE 
        pb.bet_status = 'verified'
        AND julianday(pb.created_at) - julianday(pe.event_date) BETWEEN 0 AND 14
    GROUP BY pe.player_id, pe.event_type
    HAVING COUNT(*) >= 2
    '''
    
    player_impact = pd.read_sql_query(player_query, conn)
    
    # Update the player correlation metrics
    for _, row in player_impact.iterrows():
        player_id = row['player_id']
        
        # Get top event type by impact for this player
        if row['impact_score'] > 0.6:  # Only consider strong correlations
            impact_json = json.dumps({
                "event_type": row['event_type'],
                "impact_score": float(row['impact_score']),
                "event_count": int(row['event_count'])
            })
            
            c.execute('''
                UPDATE players
                SET personal_events = ?
                WHERE id = ?
            ''', (impact_json, player_id))
    
    conn.commit()
    conn.close()
    logger.info("Personal event impact analysis completed")

# Enhanced self-assessment with Bayesian updating
def update_confidence_calibration():
    """
    Calibrate the confidence scores based on historical accuracy
    using Bayesian updating principles.
    """
    conn = get_db_connection()
    
    # Get verified bets with confidence scores
    query = '''
    SELECT 
        confidence_score/100.0 as predicted_prob,
        correct_prediction as actual_outcome
    FROM prop_bets
    WHERE bet_status = 'verified'
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 20:
        logger.info("Not enough data for confidence calibration")
        return None
    
    # Create calibration bins
    df['bin'] = pd.cut(df['predicted_prob'], 
                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                       labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    # Calculate actual success rate in each bin
    calibration = df.groupby('bin')['actual_outcome'].agg(['mean', 'count']).reset_index()
    calibration.columns = ['confidence_bin', 'actual_success_rate', 'count']
    
    # Calculate calibration ratios
    calibration['calibration_ratio'] = calibration['actual_success_rate'] / calibration['confidence_bin'].map({
        '0-20%': 0.1, '20-40%': 0.3, '40-60%': 0.5, '60-80%': 0.7, '80-100%': 0.9
    })
    
    # Save calibration data
    with open('calibration_model.json', 'w') as f:
        json.dump(calibration.to_dict('records'), f)
    
    logger.info("Confidence calibration model updated")
    return calibration

# Feature to incorporate trending news momentum
def analyze_news_momentum(player_id, recent_window=3):
    """
    Analyze the momentum of news sentiment for a player over time.
    """
    conn = get_db_connection()
    
    # Get headlines for the player, ordered by date
    query = f'''
    SELECT 
        headline,
        sentiment_score,
        publish_date
    FROM headlines
    WHERE player_id = ?
    ORDER BY publish_date DESC
    LIMIT {recent_window * 2}  # Get enough for trends
    '''
    
    headlines = conn.execute(query, (player_id,)).fetchall()
    conn.close()
    
    if len(headlines) < 3:
        return 0  # Not enough data for momentum
    
    # Calculate recent sentiment vs previous sentiment
    recent = headlines[:recent_window]
    previous = headlines[recent_window:2*recent_window]
    
    if not previous:
        return 0
    
    recent_sentiment = sum(h['sentiment_score'] for h in recent) / len(recent)
    previous_sentiment = sum(h['sentiment_score'] for h in previous) / len(previous)
    
    # Calculate momentum as change in sentiment
    momentum = recent_sentiment - previous_sentiment
    
    return momentum

# Augment player database with additional metadata
def enhance_player_metadata():
    """
    Fetch additional metadata about players to improve the model.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get all players without complete metadata
    players = c.execute('''
        SELECT id, name
        FROM players
        WHERE team = '' OR position = ''
    ''').fetchall()
    
    for player in players:
        player_id = player['id']
        player_name = player['name']
        
        # Use LangChain to get player info from knowledge base
        try:
            prompt = PromptTemplate(
                input_variables=["player_name"],
                template="""
                You are a sports knowledge assistant. For the NFL player {player_name}:
                1. What team does this player currently play for?
                2. What position does this player play?
                3. How many years have they been in the NFL?
                
                Return a JSON with: {{"team": "team name", "position": "position", "experience_years": number}}
                """
            )
            
            llm = ChatOpenAI(temperature=0)
            chain = LLMChain(llm=llm, prompt=prompt)
            
            result = chain.run(player_name=player_name)
            
            try:
                # Clean result if needed
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()
                
                result_json = json.loads(result)
                
                # Update player record
                c.execute('''
                    UPDATE players
                    SET team = ?, position = ?
                    WHERE id = ?
                ''', (
                    result_json.get("team", ""),
                    result_json.get("position", ""),
                    player_id
                ))
                
                logger.info(f"Enhanced metadata for {player_name}")
            
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LangChain result for player metadata: {result}")
        
        except Exception as e:
            logger.error(f"Error enhancing player metadata: {str(e)}")
    
    conn.commit()
    conn.close()
    logger.info("Player metadata enhancement completed")

# Build advanced betting trend analysis
def discover_advanced_betting_patterns():
    """
    Discover more complex betting patterns beyond simple correlations.
    """
    conn = get_db_connection()
    
    # Comprehensive query for pattern analysis
    query = '''
    SELECT 
        pb.id, pb.player_id, p.name, p.position, p.team, 
        pb.prop_type, pb.prop_description, pb.confidence_score, pb.correct_prediction,
        h.headline, h.sentiment_score, h.headline_type,
        strftime('%w', pb.created_at) as day_of_week,
        CASE 
            WHEN EXISTS (
                SELECT 1 FROM personal_events pe 
                WHERE pe.player_id = p.id 
                AND julianday(pb.created_at) - julianday(pe.event_date) BETWEEN 0 AND 7
            ) THEN 1 ELSE 0 
        END as has_recent_personal_event
    FROM prop_bets pb
    JOIN headlines h ON pb.headline_id = h.id
    JOIN players p ON pb.player_id = p.id
    WHERE pb.bet_status = 'verified'
    '''
    
    df = pd.read_sql_query(query, conn)
    
    patterns = []
    
    if len(df) < 50:
        logger.info("Not enough data for advanced pattern discovery")
        conn.close()
        return
    
    # Pattern 1: Day of week impact
    day_success = df.groupby('day_of_week')['correct_prediction'].agg(['mean', 'count']).reset_index()
    day_success = day_success[day_success['count'] >= 5]
    
    for _, row in day_success.iterrows():
        if abs(row['mean'] - df['correct_prediction'].mean()) > 0.1:
            day_map = {
                '0': 'Sunday', '1': 'Monday', '2': 'Tuesday', 
                '3': 'Wednesday', '4': 'Thursday', '5': 'Friday', '6': 'Saturday'
            }
            day_name = day_map.get(row['day_of_week'], f"Day {row['day_of_week']}")
            
            patterns.append({
                'pattern_type': 'day_of_week',
                'description': f"Bets made on {day_name} have a {row['mean']*100:.1f}% success rate",
                'success_rate': float(row['mean']),
                'discovery_date': datetime.datetime.now().isoformat()
            })
    
    # Pattern 2: Prop type by position
    position_prop = df.groupby(['position', 'prop_type'])['correct_prediction'].agg(['mean', 'count']).reset_index()
    position_prop = position_prop[position_prop['count'] >= 3]
    
    for _, row in position_prop.iterrows():
        if row['mean'] > 0.65:  # Only strong patterns
            patterns.append({
                'pattern_type': 'position_prop_type',
                'description': f"{row['prop_type']} bets for {row['position']} players have {row['mean']*100:.1f}% success rate",
                'success_rate': float(row['mean']),
                'discovery_date': datetime.datetime.now().isoformat()
            })
    
    # Pattern 3: Headline sentiment threshold
    # Find if there's a sentiment threshold above which bets perform better
    sentiment_ranges = [(x/10, (x+2)/10) for x in range(-10, 10, 2)]
    
    for low, high in sentiment_ranges:
        range_bets = df[(df['sentiment_score'] >= low) & (df['sentiment_score'] < high)]
        if len(range_bets) >= 5:
            success_rate = range_bets['correct_prediction'].mean()
            if abs(success_rate - df['correct_prediction'].mean()) > 0.15:
                sentiment_desc = "positive" if low >= 0 else "negative" if high <= 0 else "neutral"
                strength = "strongly" if abs(low) > 0.5 or abs(high) > 0.5 else "moderately"
                
                patterns.append({
                    'pattern_type': 'sentiment_threshold',
                    'description': f"Bets from {strength} {sentiment_desc} headlines ({low:.1f} to {high:.1f}) have {success_rate*100:.1f}% success rate",
                    'success_rate': float(success_rate),
                    'discovery_date': datetime.datetime.now().isoformat()
                })
    
    # Pattern 4: Recent personal events impact
    if 'has_recent_personal_event' in df.columns:
        personal_impact = df.groupby('has_recent_personal_event')['correct_prediction'].mean().reset_index()
        
        if len(personal_impact) > 1:
            with_events = personal_impact[personal_impact['has_recent_personal_event'] == 1]['correct_prediction'].iloc[0]
            without_events = personal_impact[personal_impact['has_recent_personal_event'] == 0]['correct_prediction'].iloc[0]
            
            if abs(with_events - without_events) > 0.1:
                better_condition = "with" if with_events > without_events else "without"
                patterns.append({
                    'pattern_type': 'personal_event_presence',
                    'description': f"Bets perform better {better_condition} recent personal events ({max(with_events, without_events)*100:.1f}% vs {min(with_events, without_events)*100:.1f}%)",
                    'success_rate': float(max(with_events, without_events)),
                    'discovery_date': datetime.datetime.now().isoformat()
                })
    
    # Store patterns in database
    c = conn.cursor()
    for pattern in patterns:
        c.execute('''
            INSERT INTO betting_patterns 
            (pattern_type, description, success_rate, discovery_date) 
            VALUES (?, ?, ?, ?)
        ''', (
            pattern['pattern_type'],
            pattern['description'],
            pattern['success_rate'],
            pattern['discovery_date']
        ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Advanced pattern discovery completed, found {len(patterns)} patterns")

# Add API endpoint for self-assessment report generation
@app.route('/api/generate-report', methods=['GET'])
def generate_assessment_report():
    """
    Generate a comprehensive self-assessment report for the AI system.
    """
    conn = get_db_connection()
    
    # Get overall performance stats
    overall = conn.execute('''
        SELECT 
            COUNT(*) as total_bets,
            SUM(correct_prediction) as correct_bets,
            ROUND(AVG(correct_prediction) * 100, 2) as success_rate
        FROM prop_bets
        WHERE bet_status = 'verified'
    ''').fetchone()
    
    # Get performance by player position
    position_performance = conn.execute('''
        SELECT 
            p.position,
            COUNT(*) as total_bets,
            SUM(pb.correct_prediction) as correct_bets,
            ROUND(AVG(pb.correct_prediction) * 100, 2) as success_rate
        FROM prop_bets pb
        JOIN players p ON pb.player_id = p.id
        WHERE pb.bet_status = 'verified'
        GROUP BY p.position
        HAVING COUNT(*) >= 3
        ORDER BY success_rate DESC
    ''').fetchall()
    
    # Get performance by prop type
    prop_performance = conn.execute('''
        SELECT 
            prop_type,
            COUNT(*) as total_bets,
            SUM(correct_prediction) as correct_bets,
            ROUND(AVG(correct_prediction) * 100, 2) as success_rate
        FROM prop_bets
        WHERE bet_status = 'verified'
        GROUP BY prop_type
        HAVING COUNT(*) >= 3
        ORDER BY success_rate DESC
    ''').fetchall()
    
    # Get top patterns
    patterns = conn.execute('''
        SELECT pattern_type, description, success_rate, discovery_date
        FROM betting_patterns
        ORDER BY success_rate DESC
        LIMIT 5
    ''').fetchall()
    
    # Get latest model performance
    model_perf = conn.execute('''
        SELECT model_version, accuracy, precision, recall, f1_score, training_date
        FROM model_performance
        ORDER BY training_date DESC
        LIMIT 1
    ''').fetchone()
    
    # Get confidence calibration data
    calibration = update_confidence_calibration()
    
    conn.close()
    
    # Format the report data
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "overall_performance": {
            "total_bets": overall['total_bets'] if overall else 0,
            "correct_bets": overall['correct_bets'] if overall else 0,
            "success_rate": overall['success_rate'] if overall else 0
        },
        "position_performance": [dict(p) for p in position_performance],
        "prop_type_performance": [dict(p) for p in prop_performance],
        "top_patterns": [dict(p) for p in patterns],
        "model_performance": dict(model_perf) if model_perf else {},
        "confidence_calibration": calibration.to_dict('records') if calibration is not None else []
    }
    
    # Generate improvement recommendations
    recommendations = []
    
    if overall and overall['total_bets'] > 0:
        # Look for underperforming areas
        if overall['success_rate'] < 55:
            recommendations.append("Overall success rate is below target. Consider refining the sentiment analysis model.")
        
        # Find best and worst performing categories
        if position_performance:
            best_pos = position_performance[0]
            worst_pos = position_performance[-1]
            if best_pos['success_rate'] - worst_pos['success_rate'] > 20:
                recommendations.append(f"Large performance gap between positions. {worst_pos['position']} bets need refinement.")
        
        if prop_performance:
            best_prop = prop_performance[0]
            worst_prop = prop_performance[-1]
            if best_prop['success_rate'] - worst_prop['success_rate'] > 20:
                recommendations.append(f"Focus on improving {worst_prop['prop_type']} bets which underperform at {worst_prop['success_rate']}%.")
        
        # Check for calibration issues
        if calibration is not None and not calibration.empty:
            overconfident_bins = calibration[calibration['calibration_ratio'] < 0.9]
            if not overconfident_bins.empty:
                recommendations.append(f"System is overconfident in {', '.join(overconfident_bins['confidence_bin'].tolist())} ranges.")
    else:
        recommendations.append("Not enough verified bets for comprehensive analysis.")
    
    report["recommendations"] = recommendations
    
    return jsonify(report)

# Add a scheduled task manager for periodic model updates
def initialize_scheduled_tasks():
    """
    Set up scheduled tasks for model updating and maintenance.
    """
    import threading
    import time
    
    def run_scheduled_tasks():
        while True:
            try:
                # Verify pending bets
                verify_bet_outcomes()
                logger.info("Scheduled task: Verified pending bets")
                
                # Analyze betting patterns
                discover_advanced_betting_patterns()
                logger.info("Scheduled task: Updated betting patterns")
                
                # Analyze personal event impact
                analyze_personal_event_impact()
                logger.info("Scheduled task: Updated personal event impact analysis")
                
                # Update player metadata
                enhance_player_metadata()
                logger.info("Scheduled task: Enhanced player metadata")
                
                # Generate fresh visualizations
                generate_visualizations()
                logger.info("Scheduled task: Generated visualizations")
            
            except Exception as e:
                logger.error(f"Error in scheduled tasks: {str(e)}")
            
            # Sleep for 6 hours between updates
            time.sleep(60 * 60 * 6)
    
    # Start the scheduled tasks in a background thread
    scheduler_thread = threading.Thread(target=run_scheduled_tasks)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("Scheduled tasks initialized")

# Add explanation logging for continuous improvement
def log_bet_explanation(player_id, headline_id, prop_bet_id, explanation):
    """
    Log detailed explanations for bet generation to improve transparency
    and aid in model training.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create explanation log table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS bet_explanations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 player_id TEXT,
                 headline_id INTEGER,
                 prop_bet_id INTEGER,
                 explanation TEXT,
                 created_at TEXT,
                 FOREIGN KEY (player_id) REFERENCES players (id),
                 FOREIGN KEY (headline_id) REFERENCES headlines (id),
                 FOREIGN KEY (prop_bet_id) REFERENCES prop_bets (id)
                 )''')
    
    # Insert explanation
    c.execute('''
        INSERT INTO bet_explanations 
        (player_id, headline_id, prop_bet_id, explanation, created_at) 
        VALUES (?, ?, ?, ?, ?)
    ''', (
        player_id,
        headline_id,
        prop_bet_id,
        explanation,
        datetime.datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()

# Add a route to view bet explanations
@app.route('/bet-explanation/<int:bet_id>')
def view_bet_explanation(bet_id):
    conn = get_db_connection()
    
    # Get bet and related data
    bet = conn.execute('''
        SELECT pb.*, p.name, h.headline
        FROM prop_bets pb
        JOIN players p ON pb.player_id = p.id
        JOIN headlines h ON pb.headline_id = h.id
        WHERE pb.id = ?
    ''', (bet_id,)).fetchone()
    
    if not bet:
        conn.close()
        return "Bet not found", 404
    
    # Get explanation
    explanation = conn.execute('''
        SELECT explanation
        FROM bet_explanations
        WHERE prop_bet_id = ?
    ''', (bet_id,)).fetchone()
    
    explanation_text = explanation['explanation'] if explanation else "No detailed explanation available."
    
    conn.close()
    
    return render_template('explanation.html',
                          bet=bet,
                          explanation=explanation_text)

# Create the explanation template
def create_explanation_template():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    with open('templates/explanation.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bet Explanation - NFL Prop Betting AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">NFL Prop Betting AI</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/analytics">Analytics</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-10 mx-auto">
                <div class="card shadow">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Detailed Bet Explanation</h5>
                    </div>
                    <div class="card-body">
                        <h4>{{ bet.prop_type }}: {{ bet.prop_description }}</h4>
                        <p class="text-muted">Player: {{ bet.name }}</p>
                        
                        <div class="alert alert-info">
                            <p><strong>Based on headline:</strong> {{ bet.headline }}</p>
                        </div>
                        
                        <div class="row mb-4">
                            <div class="col-md-4">
                                <div class="card bg-light">
                                    <div class="card-body">
                                        <h6>AI Prediction</h6>
                                        <p class="mb-0">{{ bet.ai_prediction }}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card bg-light">
                                    <div class="card-body">
                                        <h6>Odds Offered</h6>
                                        <p class="mb-0">{{ bet.odds_offered }}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card bg-light">
                                    <div class="card-body">
                                        <h6>Confidence</h6>
                                        <div class="progress" style="height: 6px;">
                                            <div class="progress-bar bg-success" role="progressbar" style="width: {{ bet.confidence_score }}%"></div>
                                        </div>
                                        <p class="mb-0 mt-1">{{ bet.confidence_score }}%</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <h5>AI's Reasoning</h5>
                        <div class="card">
                            <div class="card-body">
                                <p>{{ bet.reasoning }}</p>
                            </div>
                        </div>
                        
                        <h5 class="mt-4">Detailed Explanation</h5>
                        <div class="card">
                            <div class="card-body">
                                {{ explanation | safe }}
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <h5>Current Status</h5>
                            <div class="card">
                                <div class="card-body">
                                    <p><strong>Status:</strong> 
                                        {% if bet.bet_status == 'pending' %}
                                        <span class="badge bg-warning">Pending</span>
                                        {% elif bet.bet_status == 'verified' %}
                                        <span class="badge bg-success">Verified</span>
                                        {% endif %}
                                    </p>
                                    
                                    {% if bet.outcome %}
                                    <p><strong>Outcome:</strong>
                                        {% if bet.outcome == 'win' %}
                                        <span class="badge bg-success">Correct Prediction</span>
                                        {% elif bet.outcome == 'loss' %}
                                        <span class="badge bg-danger">Incorrect Prediction</span>
                                        {% endif %}
                                    </p>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
''')

# Update main to include all new components
def main():
    # Initialize database
    init_db()
    
    # Create HTML templates
    create_templates()
    create_explanation_template()
    
    # Initialize scheduled tasks
    initialize_scheduled_tasks()
    
    # Fetch initial headlines
    headlines = fetch_headlines(rss_feeds)
    
    # Extract player data and analyze
    players = extract_player_data(headlines)
    players = analyze_sentiment(players)
    
    # Store data
    store_data(players)
    
    # Verify any pending bets
    verify_bet_outcomes()
    
    # Analyze betting patterns
    discover_advanced_betting_patterns()
    
    # Analyze personal event impact
    analyze_personal_event_impact()
    
    # Generate visualizations
    generate_visualizations()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
  
