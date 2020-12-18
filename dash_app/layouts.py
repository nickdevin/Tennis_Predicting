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