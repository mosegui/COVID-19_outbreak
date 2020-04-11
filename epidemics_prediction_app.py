import os
import datetime as dt

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import scipy.stats
import scipy.optimize
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from jupyter_plotly_dash import JupyterDash

from read_data_spain import get_data_datadista


class LogisticPredictorRegional:

    num_projections = 29

    num_data_curves = 2  # number of data sets per projection/per display (one for cumulative and one for non-cumulative)
    num_projection_curves = 2  # number of projection curves per projection/per display (one for cumulative and one for non-cumulative)

    def __init__(self, cases, log_y=False):

        self.cases = cases
        self.log_y = log_y

        self._cases_diff_max = 0

        self._create_fig()

        self.add_figure_traces()

        self.update_fig_layout()

    def _create_fig(self):
        self.fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])
        self.fig_title_prefix = "Predicción COVID-19"

    def logistic_model(self, x, a, b, c):
        return c / (1 + np.exp(-a * (x - b)))

    def generalised_logistic_model(self, x, a, b, nu):
        return self.coeff_c / (1 + np.exp(-a * (x - b))) ** (1 / nu)

    def _get_relative_time_axis(self, df_index):
        _get_days_axis = lambda ts: (ts - df_index[0]).days
        return np.array(list(map(_get_days_axis, df_index)))

    def _get_cases_fit(self):

        time_axis = self._get_relative_time_axis(self.cases.index)

        try:
            return scipy.optimize.curve_fit(self.logistic_model, time_axis, self.cases, p0=[2, 30, 1e4])
        except:
            return np.array([np.nan] * 3), None

    def _get_visibility_mask(self, idx):

        if self.num_projections % 2 == 0:
            # the + 1 accomodates the logistic projection, which is calculated separatedly from the other generalized logistic projections
            mask = ([True] * self.num_data_curves) + ([False] * (self.num_projections + 1) * self.num_projection_curves)
        else:
            mask = ([True] * self.num_data_curves) + ([False] * (self.num_projections + 1 ) * self.num_projection_curves)

        mask[(self.num_projection_curves * idx + self.num_data_curves):
             (self.num_projection_curves * idx + self.num_data_curves + self.num_projection_curves)] = [True] * self.num_projection_curves

        return mask

    @property
    def cases_diff_max(self):
        return self._cases_diff_max

    @cases_diff_max.setter
    def cases_diff_max(self, value):
        if value > self._cases_diff_max:
            self._cases_diff_max = value


    def add_figure_traces(self):  # refactor into methods

        self.prediction_coeffs = []

        self.fig.append_trace(go.Scatter(x=self.cases.index,
                                         y=self.cases,
                                         name=f"Casos totales",
                                         mode="lines"), 1, 1)

        self.fig.add_trace(go.Scatter(x=self.cases.index[1:],
                                      y=np.diff(self.cases),
                                      name=f"Nuevos casos diarios",
                                      mode="lines"), 1, 1, secondary_y=True)
        self.cases_diff_max = np.nanmax(np.diff(self.cases))

        logistic_fit, _ = self._get_cases_fit()

        extended_time_index = [self.cases.index[0] + dt.timedelta(days=x) for x in range(2 * len(self.cases.index))]
        logistic_prediction = self.logistic_model(self._get_relative_time_axis(extended_time_index),
                                                  logistic_fit[0],
                                                  logistic_fit[1],
                                                  logistic_fit[2])

        max_infected = 1.25 * logistic_fit[2]
        min_infected = 0.75 * logistic_fit[2]

        if self.num_projections % 2 == 0:
            self.coeffs_c = sorted(np.concatenate((np.array([logistic_fit[2]]), np.linspace(min_infected, max_infected, self.num_projections))))
        else:
            # if the chosen number of projections is even, its c value will naturally appear at the center of the linspace. No need to add it manually
            self.coeffs_c = np.linspace(min_infected, max_infected, self.num_projections)

        for coeff_c in self.coeffs_c:

            try:

                self.coeff_c = int(coeff_c)

                ref_width = 1.5
                ref_color = 'green'

                if self.coeff_c == int(logistic_fit[2]):

                    self.prediction_coeffs.append((logistic_fit[0], logistic_fit[1], int(logistic_fit[2]), None))

                    self.fig.append_trace(go.Scatter(x=extended_time_index,
                                                     y=logistic_prediction,
                                                     name="Proyección logística",
                                                     line=dict(color=ref_color, width=ref_width,
                                                               dash='dash'),
                                                     visible=True), 1, 1)

                    self.fig.add_trace(go.Scatter(x=extended_time_index[1:],
                                                  y=np.diff(logistic_prediction),
                                                  name=f"Modelo logístico tasa",
                                                  line=dict(color=ref_color, width=ref_width,
                                                            dash='dash'),
                                                  visible=True), 1, 1, secondary_y=True)
                    self.cases_diff_max = np.nanmax(np.diff(logistic_prediction))



                else:

                    gen_logistic_fit, _ = scipy.optimize.curve_fit(self.generalised_logistic_model,
                                                                   self._get_relative_time_axis(self.cases.index),
                                                                   self.cases,
                                                                   p0=[logistic_fit[0], logistic_fit[1], 1])

                    self.prediction_coeffs.append((gen_logistic_fit[0], gen_logistic_fit[1], self.coeff_c, gen_logistic_fit[2]))

                    gen_logistic_prediction = self.generalised_logistic_model(self._get_relative_time_axis(extended_time_index),
                                                                              gen_logistic_fit[0],
                                                                              gen_logistic_fit[1],
                                                                              gen_logistic_fit[2])

                    self.fig.append_trace(go.Scatter(x=extended_time_index,
                                                     y=gen_logistic_prediction,
                                                     name="Proyección logística<br>generalizada",
                                                     line=dict(color='black', width=1,
                                                               dash='dash'),
                                                     visible=False), 1, 1)
                    self.fig.add_trace(go.Scatter(x=extended_time_index[1:],
                                                  y=np.diff(gen_logistic_prediction),
                                                  name=f"Modelo logístico<br>generalizado tasa",
                                                  line=dict(color='black', width=1,
                                                            dash='dash'),
                                                  visible=False), 1, 1, secondary_y=True)
                    self.cases_diff_max = np.nanmax(np.diff(gen_logistic_prediction))

            except:

                self.prediction_coeffs.append((None,) * 4)

                self.fig.append_trace(go.Scatter(x=extended_time_index,
                                                 y=np.ones_like(extended_time_index) * np.nan,
                                                 line=dict(color='black', width=1,
                                                           dash='dash'),
                                                 visible=False), 1, 1)

                self.fig.append_trace(go.Scatter(x=extended_time_index,
                                                 y=np.ones_like(extended_time_index) * np.nan,
                                                 line=dict(color='black', width=1,
                                                           dash='dash'),
                                                 visible=False), 1, 1)

    def _get_projections_slider(self):

        steps = []

        # TODO: extract
        for i in range(self.num_projections):

            speed, inflection, total_infected, nu = self.prediction_coeffs[i]

            if inflection:
                inflection = (self.cases.index[0] + dt.timedelta(days=np.round(inflection))).strftime('%Y-%m-%d')
            else:
                inflection = self.cases.index[0]

            if i == self.num_projections // 2:
                label = "Evolución logística"
                title_text = "Logistic:<br>infection speed: {:.2f}\tworst day: {}\ttotal infected: {}".format(speed, inflection, total_infected)
            else:
                label = ""
                title_text = "Generalized Logistic:<br>infection speed: {:.2f}\tworst day: {}\ttotal infected: {}\t(nu: {:.2f})".format(speed, inflection, total_infected, nu)

            step = dict(
                method="update",
                # args=[],
                args=[{"visible": self._get_visibility_mask(i)}, {"title.text": title_text}],
                label=label,
            )

            steps.append(step)

        sliders = [dict(
            active=self.num_projections // 2,  # corresponds to logistic projection if projection multipliers stay 0.75 and 1.25
            pad={"t": 50},
            steps=steps,
        )]

        return sliders


    def update_fig_layout(self):

        self.buttons_annotations = [
            dict(text="Escenario favorable", x=0.2, xref="paper", y=-0.6, yref="paper", align="left", showarrow=False),
            dict(text="Logístico", x=0.5, xref="paper", y=-0.6, yref="paper", showarrow=False),
            dict(text="Escenario desfavorable", x=0.8, xref="paper", y=-0.6, yref="paper", showarrow=False),
        ]

        yaxis_max = np.nanmax(self.coeffs_c)

        if self.log_y:
            yaxis_dict = dict(title="Número de casos", range=[0, np.log10(yaxis_max)* 1.1], type='log')
        else:
            yaxis_dict = dict(title="Número de casos", range=[-yaxis_max * 0.1, yaxis_max * 1.1])

        self.fig.update_layout(showlegend=True,
                               xaxis=dict(title="Tiempo (días)"),
                               yaxis=yaxis_dict,
                               yaxis2=dict(title="Tasa de casos", range=[-self.cases_diff_max * 0.1, self.cases_diff_max * 1.1]),
                               sliders=self._get_projections_slider(),
                               annotations=self.buttons_annotations,
                               title="Dummy Title"
                               )

##############################################

# get data
CASES = get_data_datadista(r"https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_casos.csv")


def generate_figure(data_set, region_name, scaling="Linear"):
    """Generate Figure of Region
    """
    if scaling == 'Logarithmic':
        regional_figure = LogisticPredictorRegional(data_set[region_name], log_y=True)
    else:
        regional_figure = LogisticPredictorRegional(data_set[region_name])

    return regional_figure.fig


def generate_regions_dropdown_options(data_set):

    options_list = []

    for region in data_set.columns:
        options_list.append({'label': region, 'value': region})

    return options_list

def generate_scaling_dropdown_options():

    options_list = []

    for scaling in ["Linear","Logarithmic"]:
        options_list.append({'label': scaling, 'value': scaling})

    return options_list


# create application
# app = dash.Dash(__name__)
app = JupyterDash(__name__)

# define application layout
app.layout = html.Div(children=[
    # html.H1(children='Predicciones COVID-19'),
    #
    # html.Div(children='''
    #     Descubre como el Coronavirus destruye tu sistema sanitario favorito!
    # '''),

    html.Div(
            [
                html.P('\n'),
                dcc.Dropdown(
                        id = 'Regions',
                        options=generate_regions_dropdown_options(CASES),
                        value='Total',
                ),
            ],
            className='two columns',
            style={'display': 'inline-block', 'width': '30%'},
        ),

    html.Div(
        [
            html.P('\n'),
            dcc.Dropdown(
                id='Scaling',
                options=generate_scaling_dropdown_options(),
                value='Linear',
            ),
        ],
        className='two columns',
        style={'display': 'inline-block', 'width': '30%'},
    ),

    dcc.Graph(id='graph')
])

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('Regions', 'value'),
     dash.dependencies.Input('Scaling', 'value')])
def update_image_src(region, scaling):
    """
    """
    if not region:
        return generate_figure(CASES, 'Total', scaling=scaling)
    return generate_figure(CASES, region, scaling=scaling)





if __name__ == '__main__':
    app.run_server(debug=True)

    # foo = LogisticPredictorRegional(CASES['Total'])



# class LogisticPredictor:
#     multiplicity = 7  # How many curves per data entry are generated considered all data sets
#
#     def __init__(self, cases):
#
#         # check all datasets column are the same
#
#         self.cases = cases
#
#         self.regions = list(self.cases.columns)
#
#         self.visible_initially = False
#
#         self._create_fig()
#
#         self.add_figure_traces()
#
#         self.add_figure_widgets()
#
#         self.update_fig_layout()
#
#     def _create_fig(self):
#         self.fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])
#         self.fig_title_prefix = "Predicción COVID-19"
#
#     def _get_visibility_mask(self, region):
#
#         mask = [False] * self.multiplicity * len(self.regions)
#
#         true_index = list(self.regions).index(region)
#
#         mask[(self.multiplicity * true_index):(self.multiplicity * true_index + self.multiplicity)] = [True] * self.multiplicity
#
#         return mask
#
#     def logistic_model(self, x, a, b, c):
#         return c / (1 + np.exp(-(x - b) / a))
#
#     def generalised_logistic_model(self, x, a, b, nu):
#         return self.coeff_c / (1 + np.exp(-(x - b) / a)) ** (1 / nu)
#
#     def _get_relative_time_axis(self, df_index):
#         _get_days_axis = lambda ts: (ts - df_index[0]).days
#         return np.array(list(map(_get_days_axis, df_index)))
#
#     def _get_cases_fit(self, region):
#
#         time_axis = self._get_relative_time_axis(self.cases.index)
#
#         try:
#             return scipy.optimize.curve_fit(self.logistic_model, time_axis, self.cases[region], p0=[2, 30, 5000])
#         except:
#             return np.array([np.nan] * 3), None
#
#     def add_figure_traces(self):
#
#         for region in self.regions:
#
#             if region == 'Total':
#                 self.visible_initially = True
#
#             self.fig.append_trace(go.Scatter(x=self.cases.index,
#                                              y=self.cases[region],
#                                              name=f"Casos totales {region}",
#                                              mode="lines",
#                                              visible=self.visible_initially), 1, 1)
#
#             logistic_fit, _ = self._get_cases_fit(region)
#
#             extended_time_index = [self.cases.index[0] + dt.timedelta(days=x) for x in range(2 * len(self.cases.index))]
#             logistic_prediction = self.logistic_model(self._get_relative_time_axis(extended_time_index),
#                                                       logistic_fit[0],
#                                                       logistic_fit[1],
#                                                       logistic_fit[2])
#
#             self.fig.append_trace(go.Scatter(x=extended_time_index,
#                                              y=logistic_prediction,
#                                              name=f"Modelo logístico acumulado {region}",
#                                              line=dict(color='black', width=1,
#                                                        dash='dash'),
#                                              visible=self.visible_initially), 1, 1)
#
#             max_infected = 1.5 * logistic_fit[2]
#             min_infected = 0.75 * logistic_fit[2]
#
#             for coeff_c in np.linspace(min_infected, max_infected, self.multiplicity - 2):
#                 try:
#                     self.coeff_c = int(coeff_c)
#                     gen_logistic_fit, _ = scipy.optimize.curve_fit(self.generalised_logistic_model,
#                                                                    self._get_relative_time_axis(self.cases.index),
#                                                                    self.cases[region],
#                                                                    p0=[logistic_fit[0], logistic_fit[1], 1])
#
#                     gen_logistic_prediction = self.generalised_logistic_model(self._get_relative_time_axis(extended_time_index),
#                                                                               gen_logistic_fit[0],
#                                                                               gen_logistic_fit[1],
#                                                                               gen_logistic_fit[2])
#                 except:
#                     gen_logistic_prediction = np.ones_like(extended_time_index) * np.nan
#
#                 self.fig.append_trace(go.Scatter(x=extended_time_index,
#                                                  y=gen_logistic_prediction,
#                                                  name=f"Modelo logístico acumulado {self.coeff_c} {region}",
#                                                  line=dict(color='black', width=1,
#                                                            dash='dash'),
#                                                  visible=self.visible_initially), 1, 1)
#
#     def _get_projections_slider(self):
#
#         # Create and add slider
#         steps = []
#         for i in range(self.multiplicity):
#             step = dict(
#                 method="restyle",
#                 args=["visible", [False] * self.amount_projections],
#             )
#             step["args"][1][i] = True  # Toggle i'th trace to "visible"
#             steps.append(step)
#
#         sliders = [dict(
#             active=10,
#             currentvalue={"prefix": "Frequency: "},
#             pad={"t": 50},
#             steps=steps
#         )]
#
#     def _get_regional_buttons(self):
#
#         regional_buttons = []
#
#         for region in self.regions:
#             button = dict(label=region, method="update", args=[{"visible": self._get_visibility_mask(region)},
#                                                                {"title": f"{self.fig_title_prefix}: {region.upper()}"}])
#
#             regional_buttons.append(button)
#
#         regional_buttons = [regional_buttons[-1]] + regional_buttons[:-1]  # put the global Spain statistics first
#
#         return dict(buttons=regional_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
#                     x=0.895, xanchor="left", y=1.15, yanchor="top", active=0)
#
#     def _get_scaling_widgets(self):
#
#         scaling_buttons = [
#             dict(label='Lineal', method='update', args=[{'visible': []},
#                                                         {'yaxis': {'type': 'linear'}}]),
#             dict(label='Logarítmica', method='update', args=[{'visible': []},
#                                                              {'yaxis': {'type': 'log'}}])
#         ]
#
#         return dict(buttons=scaling_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
#                     x=0.6, xanchor="left", y=1.15, yanchor="top")
#
#     def add_figure_widgets(self):
#
#         #         self._get_projections_slider()
#
#         self.fig.update_layout(updatemenus=[self._get_regional_buttons(), self._get_scaling_widgets()])
#
#     def update_fig_layout(self):
#
#         self.buttons_annotations = [
#             dict(text="Representación", x=0.53, xref="paper", y=1.10, yref="paper", align="left", showarrow=False),
#             dict(text="Comunidad<br>Autónoma", x=0.89, xref="paper", y=1.125, yref="paper", showarrow=False)
#         ]
#
#         # Update remaining layout properties
#         self.fig.update_layout(showlegend=True,
#                                title=f"{self.fig_title_prefix}: TOTAL",
#                                xaxis_title="Tiempo (días)",
#                                yaxis_title="Número de casos",
#                                annotations=self.buttons_annotations,
#                                #                                sliders=sliders
#                                )
#
#         self.fig.show()


# class LogisticModeler:
#     multiplicity = 4  # How many curves per data entry are generated considered all data sets
#
#     def __init__(self, cases, deaths, icu=None, recovered=None):
#
#         # check all datasets column are the same
#
#         self.cases = cases
#         self.deaths = deaths
#         self.icu = icu
#         self.recovered = recovered
#
#         self.regions = list(self.cases.columns)
#         self.visible_initially = False
#
#         self._get_labels_n_dfs()
#
#         self._create_fig()
#
#         self.add_figure_traces()
#
#         self.add_figure_widgets()
#
#         self.update_fig_layout()
#
#     def _get_labels_n_dfs(self):
#
#         self.dfs = [self.cases, self.deaths]
#
#         if self.icu is not None:
#             self.dfs.append(self.icu)
#         if self.recovered is not None:
#             self.dfs.append(self.recovered)
#
#         self.labels = ['cases', 'deaths', 'icu', 'recovered'][:len(self.dfs)]
#
#     def _create_fig(self):
#         self.fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])
#         self.fig_title_prefix = "Evolución COVID-19"
#
#     def _get_visibility_mask(self, region):
#
#         mask = [False] * self.multiplicity * len(self.regions)
#
#         true_index = list(self.regions).index(region)
#
#         mask[(self.multiplicity * true_index):(self.multiplicity * true_index + self.multiplicity)] = [True] * self.multiplicity
#
#         return mask
#
#     def logistic_model(self, x, a, b, c):
#         return c / (1 + np.exp(-(x - b) / a))
#
#     def _get_relative_time_axis(self, df_index):
#         _get_days_axis = lambda ts: (ts - df_index[0]).days
#         return np.array(list(map(_get_days_axis, df_index)))
#
#     def _get_cases_fit(self, region):
#
#         time_axis = self._get_relative_time_axis(self.cases.index)
#
#         try:
#             return scipy.optimize.curve_fit(self.logistic_model, time_axis, self.cases[region], p0=[2, 30, 5000])
#         except:
#             return np.array([np.nan] * 3), None
#
#
#     def add_figure_traces(self):
#
#         for region in self.regions:
#
#             if region == 'Total':
#                 self.visible_initially = True
#
#             self.fig.append_trace(go.Scatter(x=self.cases.index,
#                                              y=self.cases[region],
#                                              name=f"Casos totales",  # " {region}",
#                                              mode="lines",
#                                              visible=self.visible_initially), 1, 1)
#
#             logistic_fit, _ = self._get_cases_fit(region)
#
#             extended_time_index = [self.cases.index[0] + dt.timedelta(days=x) for x in range(2 * len(self.cases.index))]
#             model_evolution = self.logistic_model(self._get_relative_time_axis(extended_time_index), logistic_fit[0], logistic_fit[1], logistic_fit[2])
#
#             self.fig.append_trace(go.Scatter(x=extended_time_index,
#                                              y= model_evolution,
#                                              name=f"Modelo logístico acumulado",  # " {region}",
#                                              line=dict(color='black', width=1,
#                                                        dash='dash'),
#                                              visible=self.visible_initially), 1, 1)
#
#             self.fig.add_trace(go.Scatter(x=extended_time_index[1:],
#                                           y= np.diff(model_evolution),
#                                           name=f"Modelo logístico tasa",  # " {region}",
#                                           line=dict(color='black', width=1,
#                                                     dash='dash'),
#                                           visible=self.visible_initially), 1, 1, secondary_y=True)
#
#             self.fig.add_trace(go.Scatter(x=self.cases.index[1:],
#                                           y= np.diff(self.cases[region]),
#                                           name=f"Nuevos casos diarios",  # " {region}",
#                                           mode="lines",
#                                           visible=self.visible_initially), 1, 1, secondary_y=True)
#
#
#     def _get_regional_buttons(self):
#
#         regional_buttons = []
#
#         for region in self.regions:
#             button = dict(label=region, method="update", args=[{"visible": self._get_visibility_mask(region)},
#                                                                {"title": f"{self.fig_title_prefix}: {region.upper()}"}])
#
#             regional_buttons.append(button)
#
#         regional_buttons = [regional_buttons[-1]] + regional_buttons[:-1]  # put the global Spain statistics first
#
#         return dict(buttons=regional_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
#                     x=0.895, xanchor="left", y=1.15, yanchor="top", active=0)
#
#     def _get_scaling_widgets(self):
#
#         scaling_buttons = [
#             dict(label='Lineal', method='update', args=[{'visible': []},
#                                                         {'yaxis': {'type': 'linear'}}]),
#             dict(label='Logarítmica', method='update', args=[{'visible': []},
#                                                              {'yaxis': {'type': 'log'}}])
#         ]
#
#         return dict(buttons=scaling_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
#                     x=0.6, xanchor="left", y=1.15, yanchor="top")
#
#     def add_figure_widgets(self):
#         self.fig.update_layout(updatemenus=[self._get_regional_buttons(), self._get_scaling_widgets()])
#
#     def update_fig_layout(self):
#
#         self.buttons_annotations = [
#             dict(text="Representación", x=0.53, xref="paper", y=1.10, yref="paper", align="left", showarrow=False),
#             dict(text="Comunidad<br>Autónoma", x=0.89, xref="paper", y=1.125, yref="paper", showarrow=False)
#         ]
#
#         # Update remaining layout properties
#         self.fig.update_layout(showlegend=True,
#                                title=f"{self.fig_title_prefix}: TOTAL",
#                                xaxis_title="Tiempo (días)",
#                                yaxis_title="Número de casos",
#                                annotations=self.buttons_annotations
#                                )
#
#         self.fig.show()