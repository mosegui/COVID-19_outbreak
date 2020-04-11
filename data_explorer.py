import os
import datetime as dt

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataExplorer:
    multiplicity = 8  # How many curves per data entry are generated considered all data sets

    def __init__(self, cases, deaths, icu=None, recovered=None, row_width=(0.2, 0.2, 0.6)):

        # check all datasets column are the same

        self.cases = cases
        self.deaths = deaths
        self.icu = icu
        self.recovered = recovered
        self.row_width = row_width

        self.regions = list(self.cases.columns)
        self.visible_initially = False

        self._get_labels_n_dfs()

        self._create_fig()

        self.add_figure_traces()

        self.fig.add_shape(type="line",
                           xref="x3",
                           yref="y3",
                           x0=self.cases.index[0],
                           y0=1, x1=self.cases.index[-1],
                           y1=1,
                           line=dict(color="red", width=1, dash="dashdot"),
                           )

        self.add_figure_widgets()

        self.update_fig_layout()

    def _get_labels_n_dfs(self):

        self.dfs = [self.cases, self.deaths]

        if self.icu is not None:
            self.dfs.append(self.icu)
        if self.recovered is not None:
            self.dfs.append(self.recovered)

        self.labels = ['cases', 'deaths', 'icu', 'recovered'][:len(self.dfs)]

    def _create_fig(self):
        self.fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=self.row_width, vertical_spacing=0.02)
        self.yaxis_domain = self.fig.__dict__.get('_layout_obj').yaxis.domain
        self.fig_title_prefix = "Evolución COVID-19"

    def _get_virus_growth_ratio(self, col):

        regional_new_cases_daily = np.diff(self.cases[col].values)

        foo = regional_new_cases_daily[1:]
        bar = regional_new_cases_daily[:-1]

        return foo / bar

    def _get_criticality_ratio(self, region):
        return 100 * (self.icu[region] / self.cases[region])

    def _get_mortality_ratio(self, region):
        return 100 * (self.deaths[region] / self.cases[region])

    def _get_recovery_ratio(self, region):
        return 100 * (self.recovered[region] / self.cases[region])

    def _get_visibility_mask(self, region):

        mask = [False] * self.multiplicity * len(self.regions)

        true_index = list(self.regions).index(region)

        mask[(self.multiplicity * true_index):(self.multiplicity * true_index + self.multiplicity)] = [True] * self.multiplicity

        return mask

    def add_figure_traces(self):

        for region in self.regions:

            if region == 'Total':
                self.visible_initially = True

            for label, df in zip(self.labels, self.dfs):
                self.fig.append_trace(go.Scatter(x=df.index,
                                                 y=df[region],
                                                 name=f"{label}",  # " {region}",
                                                 mode="lines",
                                                 visible=self.visible_initially), 1, 1)

            self.fig.append_trace(go.Scatter(x=self.dfs[0].index,
                                             y=self._get_criticality_ratio(region),
                                             name=f"Critical",  # " {region}",
                                             mode="lines",
                                             visible=self.visible_initially), 2, 1)

            self.fig.append_trace(go.Scatter(x=self.dfs[0].index,
                                             y=self._get_mortality_ratio(region),
                                             name=f"Mortality",  # " {region}",
                                             mode="lines",
                                             visible=self.visible_initially), 2, 1)

            self.fig.append_trace(go.Scatter(x=self.dfs[0].index,
                                             y=self._get_recovery_ratio(region),
                                             name=f"Recovery",  # " {region}",
                                             mode="lines",
                                             visible=self.visible_initially), 2, 1)

            self.fig.append_trace(go.Scatter(x=self.dfs[0].index,
                                             y=self._get_virus_growth_ratio(region),
                                             name=f"Índice<br>crec. vírico",  # " {region}",
                                             mode="lines",
                                             visible=self.visible_initially), 3, 1)

    def _get_regional_buttons(self):

        regional_buttons = []

        for region in self.regions:
            button = dict(label=region, method="update", args=[{"visible": self._get_visibility_mask(region)},
                                                               {"title": f"{self.fig_title_prefix}: {region.upper()}"}])

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
                               xaxis3_title="Tiempo (días)",
                               yaxis_title="Número de casos",
                               annotations=self.buttons_annotations
                               )

        self.fig.show()
