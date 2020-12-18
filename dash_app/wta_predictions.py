import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import re
import dash_bootstrap_components as dbc

X_train_log = pd.read_pickle('./X_train_log.data') #the (dummified) feature space used in logistic regression model
y_train = pd.read_pickle('./y_train.data') #target variable
matches3 = pd.read_pickle('./matches3.data') #original data set (post-feature engineering), with all features and all rows
rankings = pd.read_pickle('./rankings.data') #complete player rankings



# convert tournament dates to datetime
matches3['tourney_date'] = pd.to_datetime(matches3['tourney_date'], format = '%Y-%m-%d')




# best logistic regression model obtained from grid search in capstone_final Jupyter Notebook
log_best = LogisticRegression(C=7.585775750291836, solver='liblinear')
log_best.fit(X_train_log, y_train)




pred_cols = X_train_log.columns #features in ML model ***in correct order***




#Column names for dummy variables
level_cols = list(filter(lambda c: 'tourney_level' in c, X_train_log.columns))
surface_cols = list(filter(lambda c: 'surface' in c and 'win' not in c, X_train_log.columns))




def next_match(player1, player2, surface, tourney_level):
	'''
	this function organizes information for a hypothetical "next match" between two players
	
	Arguments:
	player1: one of the players in the match
	player2: the other player in the match
	surface: surface for the match to be played on
	tourney_level: tournament level where the match will take place

	The various features that go into the logistic regression model are imputed by searching through
	the dataframes matches3 and rankings for the most recent occurences of player1 playing on the
	chosen surface or at the selected tournament level, player1 and player2 playing against each other,
	etc.

	If the data contains no instanes of a player on the surface or tourney_level input, the win %
	is imputed with her overall win %
	'''
	global p1_surface_imputed
	global p2_surface_imputed
	global p1_level_imputed
	global p2_level_imputed
	
	s = pd.Series(dtype = 'float64')
	if player1 < player2:
		p1 = player1
		p2 = player2
	else:
		p1 = player2
		p2 = player1
	
	p1_mask = (matches3['player_1'] == p1) | (matches3['player_2'] == p1)
	p2_mask = (matches3['player_1'] == p2) | (matches3['player_2'] == p2)
	surface_mask = matches3['surface'] == surface
	level_mask = matches3['tourney_level'] == tourney_level
	
	p1_last_match = matches3[p1_mask].iloc[-1]
	p2_last_match = matches3[p2_mask].iloc[-1]
	p1_last_match_surface = matches3[p1_mask & surface_mask]
	p2_last_match_surface = matches3[p2_mask & surface_mask]
	p1_last_match_level = matches3[p1_mask & level_mask]
	p2_last_match_level = matches3[p2_mask & level_mask]
	
	if p1 == p1_last_match['player_1']:
		p1_string = 'player_1_'
	else:
		p1_string = 'player_2_'
		
	if p2 == p2_last_match['player_1']:
		p2_string = 'player_1_'
	else:
		p2_string = 'player_2_'
	
	s['player_1_recent_form'] = p1_last_match[p1_string + 'recent_form']
	s['player_2_recent_form'] = p2_last_match[p2_string + 'recent_form']
	
	p1_id = p1_last_match[p1_string + 'id']
	p1_most_recent_ranking = rankings[rankings['player_id'] == p1_id].iloc[-1]
	s['log_player_1_rank'] = np.log(p1_most_recent_ranking['ranking'])
	
	p2_id = p2_last_match[p2_string + 'id']
	p2_most_recent_ranking = rankings[rankings['player_id'] == p2_id].iloc[-1]
	s['log_player_2_rank'] = np.log(p2_most_recent_ranking['ranking'])
	
	years_ago = (matches3['tourney_date'].apply(lambda x: datetime.utcnow()-x).dt.days/365.2422)
	s['player_1_age'] = p1_last_match[p1_string + 'age'] + years_ago[p1_last_match.name]
	s['player_2_age'] = p2_last_match[p2_string + 'age'] + years_ago[p2_last_match.name]
	

	
	if p1_last_match_surface.shape[0] > 0:
		p1_last_match_surface = p1_last_match_surface.iloc[-1]
		if p1 == p1_last_match_surface['player_1']:
			p1_string = 'player_1_'
		else:
			p1_string = 'player_2_'
		
		s['player_1_surface_win_pct'] = p1_last_match_surface[p1_string + 'surface_win_pct']
	else:
		s['player_1_surface_win_pct'] = p1_last_match[p1_string + 'win_pct']
		p1_surface_imputed = True
		
		
	if p2_last_match_surface.shape[0] > 0:
		p2_last_match_surface = p2_last_match_surface.iloc[-1]
		if p2 == p2_last_match_surface['player_1']:
			p2_string = 'player_1_'
		else:
			p2_string = 'player_2_'
		
		s['player_2_surface_win_pct'] = p2_last_match_surface[p2_string + 'surface_win_pct']
	else:
		s['player_2_surface_win_pct'] = p2_last_match[p2_string + 'win_pct']
		p2_surface_imputed = True
	
	if p1_last_match_level.shape[0] > 0:
		p1_last_match_level = p1_last_match_level.iloc[-1]
		if p1 == p1_last_match_level['player_1']:
			p1_string = 'player_1_'
		else:
			p1_string = 'player_2_'
		
		s['player_1_level_win_pct'] = p1_last_match_level[p1_string + 'level_win_pct']
	else:
		s['player_1_level_win_pct'] = p1_last_match[p1_string + 'win_pct']
		p1_level_imputed = True
		
	if p2_last_match_level.shape[0] > 0:
		p2_last_match_level = p2_last_match_level.iloc[-1]
		if p2 == p2_last_match_level['player_1']:
			p2_string = 'player_1_'
		else:
			p2_string = 'player_2_'
		
		s['player_2_level_win_pct'] = p2_last_match_level[p2_string + 'level_win_pct']
	else:
		s['player_2_level_win_pct'] = p2_last_match[p2_string + 'win_pct']
		p2_level_imputed = True
		

	last_match = matches3[(matches3['player_1'] == p1) & (matches3['player_2'] == p2)]
	if last_match.shape[0] > 0:
		last_match = last_match.iloc[-1]
		s['player_1_h2h'] = last_match['player_1_h2h']
		s['player_2_h2h'] = last_match['player_2_h2h']
	else:
		s['player_1_h2h'] = 0
		s['player_2_h2h'] = 0
	
	for idx in surface_cols:
		if surface in idx:
			s[idx] = 1
		else:
			s[idx] = 0
			
	for idx in level_cols:
		if tourney_level in idx:
			s[idx] = 1
		else:
			s[idx] = 0
	
	s = s[pred_cols]
	return s




# IDs of players to include in app dropdown menus.
# Only want to include players who have played in the last 2 seasons
current_ids = rankings[rankings['week'] >= '2018-01-01'].copy()
current_ids = current_ids['player_id']
current_ids




# Get list of player names matching IDs in current_ids
player_list_pt1 = matches3[((matches3['player_1_id'].isin(current_ids))
							 & (matches3['player_2_id'].isin(current_ids)))]['player_1']

player_list_pt2 = matches3[((matches3['player_1_id'].isin(current_ids))
							 & (matches3['player_2_id'].isin(current_ids)))]['player_2']

player_list = pd.concat([player_list_pt1, player_list_pt2])
player_list = sorted(player_list.unique().tolist())
player_list




tourney_dict = {
	'G': 'Grand Slam',
	'F': 'WTA Finals',
	'PM': 'Premier Mandatory',
	'P': 'Premier',
	'I': 'International',
	'C': 'Challenger',
	'D': 'Fed Cup',
	'O': 'Olympics'
}
bar_rows_dict = {
	'recent_form': 'Recent form',
	'surface_win_pct': 'Win percent on ',
	'level_win_pct': 'Win percent at ',
	'h2h': 'Head-to-head matches',
	'ranking': 'Ranking'
}

p1_indices = [
	'player_1_recent_form',
	'player_1_surface_win_pct',
	'player_1_level_win_pct',
	'player_1_h2h',
	'log_player_1_rank'
]

p2_indices = [
	'player_2_recent_form',
	'player_2_surface_win_pct',
	'player_2_level_win_pct',
	'player_2_h2h',
	'log_player_2_rank'
]



# Sets y-axis label for graph in app
def set_ylabel(feature, surface, level):
	if feature == 'recent_form':
		return 'Recent Form'
	if feature == 'surface_win_pct':
		return 'Win Percent on ' + surface
	if feature == 'level_win_pct':
		return 'Win Percent in ' + tourney_dict[level] + ' Matches'
	if feature == 'h2h':
		return 'Head-to-Head Matches Won'
	if feature == 'ranking':
		return 'Player\'s Ranking'





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App gives warning if an imputed value is used in prediction
p1_surface_imputed = False
p2_surface_imputed = False
p1_level_imputed = False
p2_level_imputed = False

app.layout = html.Div([
	html.Div([
		html.H1("Women's Tennis Association Match Predictor"),
		html.Hr(style = {'color': '#CDCDCD'}),
		dbc.Row([
			dbc.Col([
				html.Label([
					"Select first player",
					dcc.Dropdown(
						id='first-player',
						clearable = False,
						value='Serena Williams', options=[
							{'label': name, 'value': name}
							for name in player_list
						], style={'width': 300})
				]),
				html.Br(),
				html.Label([
					"Select second player",
					dcc.Dropdown(
						id='second-player',
						clearable = False,
						value='Venus Williams', options=[
							{'label': name, 'value': name}
							for name in player_list
						], style={'width': 300})
				]),
				html.Br(),
				html.Label([
					"Select Surface",
					dcc.Dropdown(
						id='surface',
						clearable = False,
						value='Hard', options=[
							{'label': surf, 'value': surf}
							for surf in ['Hard', 'Grass', 'Clay', 'Carpet']
						], style={'width': 300})
				]),
				html.Br(),
				html.Label([
					"Select tournament level",
					dcc.Dropdown(
						id='level',
						clearable = False,
						value='G', options=[
							{'label': tourney_dict[key], 'value': key}
							for key in tourney_dict.keys()
						], style={'width': 300})
				])
			],
			width = 5),
			dbc.Col([
				html.Div(id = 'prediction')
			],
			width = 7,
			style={'border': '1px #CDCDCD solid'})
		]),
		html.Hr(style = {'color': '#CDCDCD'}),
		html.H3('Select graph to display.'),
		dcc.RadioItems(
					id='feature',
					value='recent_form',
					inputStyle={"margin-left": "20px"}
				),
		dcc.Graph(id = 'bar-graph')
	],
	style = {'margin-left': 50, 'margin-right': 100})
], style = {'backgroundColor': '#F9F8FF', 'font-family':'sans-serif'})

@app.callback(
	Output('feature', 'options'),
	[Input('surface', 'value'), Input('level', 'value')]
)

def update_radioItems(surface, level):
	'''
	radioItems allow user to select a particular graph to display
	Options are surface win %, head-to-head, level win %, ranking, and recent form
	radioItems options update depending on selected court surface and tournament level
	'''
	options=[{'label': bar_rows_dict[key], 'value': key} for key in bar_rows_dict.keys()]
	for option in options:
		if option['value'] == 'surface_win_pct':
			option['label'] = option['label'] + surface.lower() + ' courts'
		if option['value'] == 'level_win_pct':
			option['label'] = option['label'] + tourney_dict[level] + ' events'
	return options

@app.callback(
	Output('first-player', 'options'),
	Input('second-player', 'value')
)

def update_p1_dropdown(player2):
	'''
	This makes it impossible for the user to select the same player in both dropdown menus.
	Prevents error.
	'''
	updated_player_list = [name for name in player_list if name != player2]
	return [{'label': name, 'value': name} for name in updated_player_list]

@app.callback(
	Output('second-player', 'options'),
	Input('first-player', 'value')
)

def update_p2_dropdown(player1):
	'''
	This makes it impossible for the user to select the same player in both dropdown menus.
	Prevents error.
	'''
	updated_player_list = [name for name in player_list if name != player1]
	return [{'label': name, 'value': name} for name in updated_player_list]

@app.callback(
	Output('prediction', 'children'),
	[Input('first-player', 'value'),
	 Input('second-player', 'value'),
	 Input('surface', 'value'),
	 Input('level', 'value')])

def update_prediction(player1, player2, surface, level):
	'''
	Print the predicted winner of a match between <player1> and <player2>
	on court surface <surface> at tournament at tournament level <level>

	Print the betting odds as predicted by logistic regression model

	Print warning if a value is imputed.
	'''
	
	global p1_surface_imputed, p2_surface_imputed, p1_level_imputed, p2_level_imputed
	p1_surface_imputed = False
	p2_surface_imputed = False
	p1_level_imputed = False
	p2_level_imputed = False
	
	if player2 < player1:
		player1, player2 = player2, player1
	match_series = next_match(player1, player2, surface, level)
	prediction = log_best.predict(np.matrix(match_series))
	probability = log_best.predict_proba(np.matrix(match_series))
	if prediction == 'player_1':
		pred_statement = 'The predicted winner is ' + player1 + '.'
		p = probability[0,0]
		prob_statement = 'Her odds of winning are ' + str(round(p/(1-p), 2)) + ':1.'
	else:
		pred_statement = 'The predicted winner is ' + player2 + '.'
		p = probability[0,1]
		prob_statement = 'Her odds of winning are ' + str(round(p/(1-p), 2)) + ':1.'
		
	child = [html.H2(pred_statement), html.H3(prob_statement)]
	if p1_surface_imputed:
		child.append(html.H6('Note: No data is available for ' + player1 + ' on ' + surface.lower()
			+ ' courts, so her overall win percent is used in place of her win percent on '
			+ surface.lower() + '.'
		))
	if p2_surface_imputed:
		child.append(html.H6('Note: No data is available for ' + player2 + ' on ' + surface.lower()
			+ ' courts, so her overall win percent is used in place of her win percent on '
			+ surface.lower() + '.'
		))
	if p1_level_imputed:
		child.append(html.H6('Note: No data is available for ' + player1 + ' at ' + tourney_dict[level]
			+ ' events, so her overall win percent is used in place of her win percent at '
			+ tourney_dict[level] + ' events.'
		))
	if p2_level_imputed:
		child.append(html.H6('Note: No data is available for ' + player2 + ' at ' + tourney_dict[level]
			+ ' events, so her overall win percent is used in place of her win percent at '
			+ tourney_dict[level] + ' events.'
		))
	
	return child
	

@app.callback(
	Output('bar-graph', 'figure'),
	[Input('first-player', 'value'),
	 Input('second-player', 'value'),
	 Input('surface', 'value'),
	 Input('level', 'value'),
	 Input('feature', 'value')])

def update_figure(player1, player2, surface, level, feature):
	'''
	update graph depending on player1, player2, surface, level, and radioItems selection
	y-axis label updates according to set_ylabel function defined earlier
	actual ranking (not log of ranking) is displayed
	'''
	if player2 < player1:
		player1, player2 = player2, player1
	match_series = next_match(player1, player2, surface, level)
	p1_series = match_series[p1_indices]
	p2_series = match_series[p2_indices]
	p1_series.index = bar_rows_dict.keys()
	p2_series.index = bar_rows_dict.keys()
	p1_series['ranking'] = round(np.e**p1_series['ranking'])
	p2_series['ranking'] = round(np.e**p2_series['ranking'])
	
	match_df = pd.DataFrame({'player_1': p1_series, 'player_2': p2_series}).reset_index()
	match_df = pd.melt(match_df, 'index', ['player_1', 'player_2'])
	match_df.columns = match_df.columns.str.replace('variable', 'player')
	match_df.columns = match_df.columns.str.replace('index', 'feature')
	match_df['player'] = match_df['player'].str.replace('player_1', player1)
	match_df['player'] = match_df['player'].str.replace('player_2', player2)
	match_df = match_df.sort_values(by = 'player')
	
	fig = px.bar(
		data_frame = match_df[match_df['feature'] == feature],
		x = 'player',
		y = 'value',
		color = 'player',
		labels = {'player': 'Player\'s Name', 'value': set_ylabel(feature, surface, level)},
		color_discrete_sequence = ['#82AEFF', '#B582FF'],
		width = 800
	)
	
	fig.update_layout(
		transition_duration=500,
		paper_bgcolor='#F9F8FF',
		#plot_bgcolor='#FFFFFF',
		legend={'orientation': 'h', 'yanchor': 'top', 'xanchor': 'right', 'y': 1.1, 'x': 1},
		legend_title_text=None
	)
	
	return fig

if __name__ == '__main__':
	app.run_server(debug=True)