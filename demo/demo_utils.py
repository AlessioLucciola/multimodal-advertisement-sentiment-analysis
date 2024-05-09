import altair as alt
import pandas as pd

def create_chart(windows):
    df = pd.DataFrame(windows)
    if 'logits' in df.columns:
        df = df.drop(columns=['logits'])
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('window_type:N', title='Window Type'),
        x=alt.X('start_time:Q', title='Time'),
        x2='end_time:Q',
        color=alt.Color('emotion_string:N', legend=None),
        tooltip=['start_time', 'end_time', 'emotion_string']
    ).properties(
        width=600,
        height=200
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    legend = chart.mark_rect().encode(
        y=alt.Y('emotion_string:N', axis=alt.Axis(orient='right')),
        color=alt.Color('emotion_string:N', scale=alt.Scale(scheme='category20'), legend=None),
    ).properties(
        title='Emotion'
    )

    return chart, legend