
import plotly.graph_objs as go
import plotly.express as px

from plotly.subplots import make_subplots


class MultiLineGraph:
    def __init__(self, name, x, y, trend, approxs):
        self.x = x
        self.y = y
        self.trend = trend
        self.approxs = approxs
        self.num_traces = len(approxs)

        self.fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        self.plot_curves()
        self.fig.update_layout(showlegend=True, title_text=name)
        self.fig.show()
        
    def plot_curves(self):

        color_palette = px.colors.qualitative.D3

        self.fig.add_trace(go.Scatter(
                x=self.x, y=self.y, 
                mode='lines', name='data', 
                line=dict(color=color_palette[0])
            ), 
            row=1, col=1
        )
        self.fig.add_trace(go.Scatter(
                x=self.x, y=self.trend, 
                mode='lines', name='trend', legendgroup=f'groupTrend', 
                line=dict(color=color_palette[1])
            ), 
            row=1, col=1
        )
        self.fig.add_trace(go.Scatter(
                x=self.x, y=self.y - self.trend, 
                mode='lines', name="true detrended", legendgroup=f'groupTrend', showlegend=False, 
                line=dict(color=color_palette[1])
            ), 
            row=2, col=1
        )
        
        for i in range(self.num_traces):
            if self.approxs[i]['type'] == 'scatter':
                self.fig.add_trace(go.Scatter(
                        x=self.x, y=self.approxs[i]['y'], 
                        mode='markers', name=self.approxs[i]['name'], legendgroup=f'group{i}', 
                        line=dict(color=color_palette[i+2])
                    ), 
                    row=1, col=1
                )
            else:
                self.fig.add_trace(go.Scatter(
                        x=self.x, y=self.approxs[i]['y'], 
                        mode='lines', name=self.approxs[i]['name'], legendgroup=f'group{i}', 
                        line=dict(color=color_palette[i+2])
                    ), 
                    row=1, col=1
                )
                self.fig.add_trace(go.Scatter(
                        x=self.x, y=self.y - self.approxs[i]['y'], 
                        mode='lines', name=self.approxs[i]['name'], legendgroup=f'group{i}', showlegend=False, 
                        line=dict(color=color_palette[i+2])
                    ), 
                    row=2, 
                    col=1
                )
        
        self.fig.update_layout(
            updatemenus=[
                dict(
                    direction="down", x=0.1,
                    xanchor="left", y=1.1,
                    yanchor="top"
                ),
            ],
            legend={'font': {'size': 18}}
        )

        self.fig.update_xaxes(tickfont={'size': 18}, row=1, col=1)
        self.fig.update_yaxes(tickfont={'size': 18}, row=1, col=1)
        
        self.fig.update_xaxes(tickfont={'size': 18}, row=2, col=1)
        self.fig.update_yaxes(tickfont={'size': 18}, row=2, col=1)

        self.fig.update_layout(legend=dict(itemsizing='constant'))
    