import igraph
import numpy as np
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
from token_tree import TokenTree

def make_figure():
    fig = go.Figure(
    layout=go.Layout(
        showlegend=False,
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(
            zeroline=False, 
            showticklabels=False,
            gridcolor='rgb(48,48,48)'
        ),
        yaxis=dict(
            zeroline=False, 
            showticklabels=False,
            gridcolor='rgb(48,48,48)'
        ),
        plot_bgcolor="black",
        paper_bgcolor="black"
    ))
    return fig

def show(fig):
    config = {'scrollZoom': True}
    fig.show(config=config)

def add_to_fig_tree(fig, tree: TokenTree, color, x = 0, sign = 1.0):
    Xn = []
    Yn = []
    Xe = []
    Ye = []
    sizes = []
    labels = []
    hover_text = []
    colors = []

    plot_tree(tree, Xn, Yn, Xe, Ye, sizes, labels, hover_text, colors,
              x, 0, 1024, 512,
              sign = sign)

    fig.add_trace(go.Scatter(
                        x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(96,96,96)', width=2),
                        hoverinfo='none'
                        ))
    fig.add_trace(go.Scatter(x=Xn,
                        y=Yn,
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=sizes,
                            color=colors,
                            colorscale="Aggrnyl",
                            opacity=1.0,
                            line=dict(width=0),
                        ),
                        text=hover_text,
                        hoverinfo='text',
                        ))
    fig.add_trace(go.Scatter(x=Xn,
                        y=Yn,
                        mode='text',
                        text=labels,
                        hoverinfo='text',
                        textfont=dict(
                            family="serif",
                            size=24,
                            color="white",
                        )
                        ))

def plot_tree(tree: TokenTree, 
    Xn, Yn, Xe, Ye, sizes,
    labels, hover_text, colors,
    x, y, width, height,
    sign = 1.0
    ):
    
    prob = float(np.exp(tree.logprob))
    Xn.append(x)
    Yn.append(y)
    sizes.append(prob*128)
    colors.append(tree.distance)
    
    labels.append(tree.label)
    hover_text.append(f"[{prob:.2f} | {tree.distance:.4f}]\n{tree.text}")

    child_x = x
    child_y = y - height
    for child in tree.children.values():
        prob = np.exp(child.logprob)
        child_width = width*prob
        child_height = height
        Xe += [x,child_x, None]
        Ye += [y,child_y, None]
        plot_tree(child, Xn, Yn, Xe, Ye, sizes,
                  labels, hover_text, colors,
                  child_x, child_y, child_width, child_height,
                  sign=sign)
        child_x += child_width*sign


