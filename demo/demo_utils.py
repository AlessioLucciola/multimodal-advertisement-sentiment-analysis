import altair as alt
import pandas as pd

color_scheme = {
    'neutral': '#7F7F7F',  # Grey
    'positive': '#1F77B4', # Blue
    'negative': '#FF7F0E'  # Orange
}

def create_chart(windows, title: str):
    df = pd.DataFrame(windows)
    if 'logits' in df.columns:
        df = df.drop(columns=['logits'])
    df['color'] = df['emotion_string'].map(color_scheme)
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('window_type:N', title='Window Type'),
        x=alt.X('start_time:Q', title='Time'),
        x2='end_time:Q',
        color=alt.Color('emotion_string:N', scale=alt.Scale(range=list(color_scheme.values())), legend=None),
        tooltip=['start_time', 'end_time', 'emotion_string']
    ).properties(
        width=600,
        height=200
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    legend = chart.mark_rect().encode(
        y=alt.Y('emotion_string:N', axis=alt.Axis(orient='left'), title='Emotion'),
        color=alt.Color('emotion_string:N', scale=alt.Scale(range=list(color_scheme.values())), legend=None),
    ).properties(
        title=title
    )

    return chart, legend
