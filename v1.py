import feedparser
import spacy
import re
import sqlite3
from openai import OpenAI
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
from textblob import TextBlob
from collections import defaultdict
import json

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# List of top NFL RSS feed URLs
rss_feeds = [
    "https://www.nfl.com/rss/rsslanding?searchString=home",
    "https://www.espn.com/espn/rss/nfl/news",
    # Add additional feeds as required
]

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
    "https://theathletic.com/nfl/feed/",
    "https://nflspinzone.com/feed/",
    "https://www.yardbarker.com/rss/sport/1",
    "https://www.rotoballer.com/category/nfl?feed=atom",
    "https://dknation.draftkings.com/rss/nfl",
    "https://www.fantasypros.com/feed/",
    "https://walterfootball.com/rss.xml",
    "https://www.pff.com/pff-rss",
    "https://www.footballoutsiders.com/rss.xml",
    "https://thedraftnetwork.com/feed/",
    "https://nflmocks.com/feed/",
    "https://touchdownwire.usatoday.com/feed/",
    "https://thespun.com/category/nfl/feed",
    "https://clutchpoints.com/rss/nfl",
    "https://www.totalprosports.com/category/nfl/feed/",
    "https://www.thebiglead.com/nfl/feed",
]


# Initialize SQLite database
conn = sqlite3.connect('nfl_players.db')
c = conn.cursor()

# Create tables for players, headlines, and prop bets
c.execute('''CREATE TABLE IF NOT EXISTS players (
             id TEXT PRIMARY KEY,
             sentiment_score REAL
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS headlines (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             player_id TEXT,
             headline TEXT,
             FOREIGN KEY (player_id) REFERENCES players (id)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS prop_bets (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             player_id TEXT,
             headline TEXT,
             prop_bet TEXT,
             FOREIGN KEY (player_id) REFERENCES players (id)
             )''')

conn.commit()

# Function to normalize player names to snake_case
def normalize_name(name):
    return re.sub(r'\s+', '_', name.lower())

# Function to fetch headlines from RSS feeds
def fetch_headlines(feeds):
    headlines = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            headlines.append(entry.title)
    return headlines

# Function to extract player names using NER
def extract_player_names(headlines):
    players = defaultdict(lambda: {"news": [], "sentiment_score": 0})
    for headline in headlines:
        doc = nlp(headline)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                normalized_name = normalize_name(ent.text)
                players[normalized_name]["news"].append(headline)
    return players

# Function to analyze sentiment of headlines
def analyze_sentiment(players):
    for player, data in players.items():
        for headline in data["news"]:
            blob = TextBlob(headline)
            sentiment = blob.sentiment.polarity
            players[player]["sentiment_score"] += sentiment
    return players

# Function to generate prop bets using OpenAI and store them in the database
def analyze_with_openai(player_id, headline):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that generates creative sports prop bets."},
            {
                "role": "user",
                "content": f"Generate three unique prop bets for the following NFL headline:\n\n{headline}\n\nOutput each prop bet and the odds and the recommended bet to make and your justification for it in JSON format."
            }
        ]
    )

    # Extract the generated prop bets from OpenAI's response
    prop_bets = completion.choices[0].message.content.strip()
    print(f"Generated prop bets for {player_id}: {prop_bets}")

    # Insert each generated prop bet into the database
    c.execute("INSERT INTO prop_bets (player_id, headline, prop_bet) VALUES (?, ?, ?)",
              (player_id, headline, prop_bets))
    conn.commit()

# Function to insert player data and their headlines into the database
def insert_player_data(players_data):
    for player, data in players_data.items():
        player_id = player
        sentiment_score = data["sentiment_score"]
        headlines = data["news"]

        # Insert or update player sentiment score
        c.execute("INSERT OR REPLACE INTO players (id, sentiment_score) VALUES (?, ?)", 
                  (player_id, sentiment_score))

        # Insert each headline for the player and generate prop bets
        for headline in headlines:
            c.execute("INSERT INTO headlines (player_id, headline) VALUES (?, ?)", 
                      (player_id, headline))
            # Call OpenAI to generate prop bets for each headline
            analyze_with_openai(player_id, headline)
    
    conn.commit()

# Main workflow
def main():
    headlines = fetch_headlines(rss_feeds)
    players = extract_player_names(headlines)
    player_profiles = analyze_sentiment(players)
    insert_player_data(player_profiles)
    
    # Display player profiles
    for player, profile in player_profiles.items():
        print(f"Player: {player}")
        print(f"Headlines: {profile['news']}")
        print(f"Sentiment Score: {profile['sentiment_score']}")
        print("----------")

# Run the main function
if __name__ == "__main__":
    main()

# Close the database connection
conn.close()

