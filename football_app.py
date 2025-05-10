import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys


model = joblib.load('ball_model.pkl')
le = joblib.load('ble.pkl')

st.title('Expected Goals Prediction')
st.header('Any Team From Europe Top 5 Leagues')

home = st.selectbox('Home Team', options=['Athletic Club', 'Club Atlético de Madrid', 'CA Osasuna','FC Barcelona', 'Getafe CF', 'Granada CF', 'Real Madrid CF','Rayo Vallecano de Madrid', 'RCD Mallorca', 'Real Betis Balompié','Real Sociedad de Fútbol', 'Villarreal CF', 'Valencia CF','Deportivo Alavés', 'Cádiz CF', 'UD Almería', '1. FC Köln','TSG 1899 Hoffenheim', 'Bayer 04 Leverkusen', 'Borussia Dortmund','FC Bayern München', 'VfB Stuttgart', 'VfL Wolfsburg','SV Werder Bremen', '1. FSV Mainz 05', 'FC Augsburg','SC Freiburg', 'Borussia Mönchengladbach', 'Eintracht Frankfurt','1. FC Union Berlin', 'VfL Bochum 1848', '1. FC Heidenheim 1846',
       'SV Darmstadt 98', 'RB Leipzig', 'AC Milan', 'ACF Fiorentina',
       'AS Roma', 'Atalanta BC', 'Bologna FC 1909', 'Cagliari Calcio',
       'Genoa CFC', 'FC Internazionale Milano', 'Juventus FC', 'SS Lazio',
       'SSC Napoli', 'Udinese Calcio', 'Empoli FC', 'Arsenal',
       'Aston Villa', 'Chelsea', 'Everton', 'Fulham', 'Liverpool',
       'Manchester City', 'Manchester United', 'Newcastle United',
       'Tottenham Hotspur', 'Wolverhampton Wanderers', 'Burnley',
       'Nottingham Forest', 'Crystal Palace', 'Sheffield United',
       'Luton Town', 'Brighton & Hove Albion', 'Brentford',
       'West Ham United', 'AFC Bournemouth', 'Hellas Verona FC',
       'US Salernitana 1919', 'Frosinone Calcio', 'US Sassuolo Calcio',
       'Torino FC', 'US Lecce', 'AC Monza', 'UD Las Palmas', 'Girona FC',
       'RC Celta de Vigo', 'Sevilla FC', 'Toulouse FC',
       'Stade Brestois 29', 'Olympique de Marseille', 'Montpellier HSC',
       'Lille OSC', 'OGC Nice', 'Olympique Lyonnais',
       'Paris Saint-Germain FC', 'FC Lorient', 'Stade Rennais FC 1901',
       'Le Havre AC', 'Clermont Foot 63', 'FC Nantes', 'FC Metz',
       'Racing Club de Lens', 'Stade de Reims', 'AS Monaco FC',
       'RC Strasbourg Alsace'])
away = st.selectbox('Away Team', options=['RC Strasbourg Alsace', 'AS Monaco FC', 'Stade de Reims',
       'Racing Club de Lens', 'FC Metz', 'FC Nantes', 'Clermont Foot 63',
       'Le Havre AC', 'Stade Rennais FC 1901', 'FC Lorient',
       'Paris Saint-Germain FC', 'Olympique Lyonnais', 'OGC Nice',
       'Lille OSC', 'Montpellier HSC', 'Olympique de Marseille',
       'Stade Brestois 29', 'Toulouse FC', 'Sevilla FC',
       'RC Celta de Vigo', 'Girona FC', 'UD Las Palmas', 'AC Monza',
       'US Lecce', 'Torino FC', 'US Sassuolo Calcio', 'Frosinone Calcio',
       'US Salernitana 1919', 'Hellas Verona FC', 'AFC Bournemouth',
       'West Ham United', 'Brentford', 'Brighton & Hove Albion',
       'Luton Town', 'Sheffield United', 'Crystal Palace',
       'Nottingham Forest', 'Burnley', 'Wolverhampton Wanderers',
       'Tottenham Hotspur', 'Newcastle United', 'Manchester United',
       'Manchester City', 'Liverpool', 'Fulham', 'Everton', 'Chelsea',
       'Aston Villa', 'Arsenal', 'Empoli FC', 'Udinese Calcio',
       'SSC Napoli', 'SS Lazio', 'Juventus FC',
       'FC Internazionale Milano', 'Genoa CFC', 'Cagliari Calcio',
       'Bologna FC 1909', 'Atalanta BC', 'AS Roma', 'ACF Fiorentina',
       'AC Milan', 'RB Leipzig', 'SV Darmstadt 98',
       '1. FC Heidenheim 1846', 'VfL Bochum 1848', '1. FC Union Berlin',
       'Eintracht Frankfurt', 'Borussia Mönchengladbach', 'SC Freiburg',
       'FC Augsburg', '1. FSV Mainz 05', 'SV Werder Bremen',
       'VfL Wolfsburg', 'VfB Stuttgart', 'FC Bayern München',
       'Borussia Dortmund', 'Bayer 04 Leverkusen', 'TSG 1899 Hoffenheim',
       '1. FC Köln', 'UD Almería', 'Cádiz CF', 'Deportivo Alavés',
       'Valencia CF', 'Villarreal CF', 'Real Sociedad de Fútbol',
       'Real Betis Balompié', 'RCD Mallorca', 'Rayo Vallecano de Madrid',
       'Real Madrid CF', 'Granada CF', 'Getafe CF', 'FC Barcelona',
       'CA Osasuna', 'Club Atlético de Madrid', 'Athletic Club'])

input_data = {
    'HOME': home,
    'AWAY': away
}

def Predict_Eligibility(input_data):
    input_df = pd.DataFrame([input_data])
    cols = ['HOME', 'AWAY']
    for col in cols:
        input_df[col] = le.fit_transform(input_df[col])

    Prediction = model.predict(input_df)
    return Prediction

if st.button('Predict'):
    Prediction = Predict_Eligibility(input_data)
    Prediction
else:
    st.info('Press button to predict')