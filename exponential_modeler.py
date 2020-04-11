import os
import datetime as dt

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ExponentialModeler:
    multiplicity = 2  # How many curves per data entry are generated considered all data sets

    def __init__(self, cases, row_width=(0.25, 0.25, 0.5)):

        # check all datasets column are the same

        self.cases = cases
        self.row_width = row_width

        self.regions = list(self.cases.columns)
        self.visible_initially = False

        self._create_fig()

        self.add_figure_traces()
        self.add_figure_traces3()
        self.add_figure_traces2()

        self.add_figure_widgets()

        self.update_fig_layout()


    def _create_fig(self):
        self.fig = make_subplots(rows=3, cols=1, shared_xaxes=False, row_width=self.row_width, vertical_spacing=0.15)
        self.yaxis_domain = self.fig.__dict__.get('_layout_obj').yaxis.domain
        self.fig_title_prefix = "Evolución COVID-19"

    def _get_visibility_mask(self, region):

        mask = [False] * self.multiplicity * len(self.regions)

        true_index = list(self.regions).index(region)

        mask[(self.multiplicity * true_index):(self.multiplicity * true_index + self.multiplicity)] = [True] * self.multiplicity

        return mask + [True] * 2

    def exponential_model(self, x, a, b, c):
        return a * np.exp(b * (x - c))

    def _get_relative_time_axis(self, df_index):
        _get_days_axis = lambda ts: (ts - df_index[0]).days
        return np.array(list(map(_get_days_axis, df_index)))

    def _get_cases_fit(self, region):

        time_axis = self._get_relative_time_axis(self.cases.index)

        amplitude = self.cases[region].max() // 100.

        bounds = ([0, 0, -10], [np.inf, np.inf, 10])

        try:
            return scipy.optimize.curve_fit(self.exponential_model, time_axis, self.cases[region], p0=[amplitude, 2, 0], bounds=bounds)
        except:
            return np.array([np.nan] * 3), None


    def add_figure_traces(self):

        self.exp_models_log = {}

        for region in self.regions:

            if region == 'Total':
                self.visible_initially = True

            self.fig.append_trace(go.Scatter(x=self.cases.index,
                                             y=self.cases[region],
                                             name=f"reported infections",
                                             mode="lines",
                                             visible=self.visible_initially), 1, 1)

            model_pars, cov_matrix = self._get_cases_fit(region)

            self.exp_models_log[region] = {"a":model_pars[0], "b":model_pars[1], "c":model_pars[2]}

            regional_exp_model = self.exponential_model(self._get_relative_time_axis(self.cases.index), *model_pars)

            self.fig.append_trace(go.Scatter(x=self.cases.index,
                                             y=regional_exp_model,
                                             name=f"Modelo exponencial",  # " {region}",
                                             line=dict(color='black', width=1,
                                                       dash='dash'),
                                             visible=self.visible_initially), 1, 1)

    def add_figure_traces3(self):

        points = [(k, v.get("a")) for k, v in self.exp_models_log.items()]

        points = sorted(points, key= lambda x: x[1])[::-1]

        regions = [item[0] for item in points]
        scale_factor = [item[1] for item in points]

        self.fig.append_trace(go.Bar(x=regions,
                                     y=scale_factor,
                                     name=f"",
                                     visible=True), 3, 1)

    def add_figure_traces2(self):

        points = [(k, v.get("b")) for k, v in self.exp_models_log.items()]

        points = sorted(points, key=lambda x: x[1])[::-1]

        regions = [item[0] for item in points]
        infection_speeds = [item[1] for item in points]

        self.fig.append_trace(go.Bar(x=regions,
                                     y=infection_speeds,
                                     name=f"reported infections",
                                     visible=True), 2, 1)


    def _get_regional_buttons(self):

        regional_buttons = []

        for region in self.regions:

            day_zero = (self.cases.index[0] + dt.timedelta(days=np.round(self.exp_models_log[region]['c']))).strftime('%Y-%m-%d')
            title_text = "Infection speed: {:.4f}<br>Estimated day 0: {}".format(self.exp_models_log[region]['b'], day_zero)


            button = dict(label=region, method="update", args=[{"visible": self._get_visibility_mask(region)},
                                                               {"title": title_text}])

            regional_buttons.append(button)

        regional_buttons = [regional_buttons[-1]] + regional_buttons[:-1]  # put the global Spain statistics first

        return dict(buttons=regional_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
                    x=0.895, xanchor="left", y=1.15, yanchor="top", active=0)

    def _get_scaling_widgets(self):

        scaling_buttons = [
            dict(label='Lineal', method='update', args=[{'visible': []},
                                                        {'yaxis': {'type': 'linear', 'domain': self.yaxis_domain}}]),
            dict(label='Logarítmica', method='update', args=[{'visible': []},
                                                             {'yaxis': {'type': 'log', 'domain': self.yaxis_domain}}])
        ]

        return dict(buttons=scaling_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
                    x=0.6, xanchor="left", y=1.15, yanchor="top")

    def add_figure_widgets(self):
        self.fig.update_layout(updatemenus=[self._get_regional_buttons(), self._get_scaling_widgets()])

    def update_fig_layout(self):

        self.buttons_annotations = [
            dict(text="Representación", x=0.53, xref="paper", y=1.10, yref="paper", align="left", showarrow=False),
            dict(text="Comunidad<br>Autónoma", x=0.89, xref="paper", y=1.125, yref="paper", showarrow=False)
        ]

        # Update remaining layout properties
        self.fig.update_layout(showlegend=True,
                               title=f"{self.fig_title_prefix}: TOTAL",
                               height=700,
                               xaxis_title="Tiempo (días)",
                               yaxis_title="Número de casos",
                               yaxis2=dict(title="Índice escala", type="log"),
                               yaxis3=dict(title="Velocidad<br>propagación vírica", type="log"),
                               annotations=self.buttons_annotations
                               )

        self.fig.show()
