from keras.layers.recurrent_v2 import calculate_sequence_by_mask
from pkg_resources import require
import streamlit as st
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.utils import to_categorical
import plotly.express as px

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

required_columns = ['TIME', 'UID', 'CALL TYPE', 'NET CALL TIME', 'TYPE']

st.title("Call Centre Attendant Prediction")
st.write(f"Upload the CSV file with {','.join(required_columns)} columns for predictions:")

def time_string_to_secs(time_string):
    hours, minutes, seconds = time_string.strip().split(':')
    minutes = int(minutes) + 60 * int(hours)
    seconds = int(seconds) + 60 * minutes
    
    return seconds

def make_plot(df, x, y, title, color=None):

    fig = px.line(df, x, y, title=title)
    
    if color is not None:
        fig.color = color

    return fig

def calculate_humans_predicted(arr):
    humans = set()
    for row in arr:
        class_ = np.argmax(row)
        if class_ != 0:
            humans.add(class_)

    return len(humans)

def hourly_calculate(df):
    humans = set()

    for record in df:
        if record != "Bot":
            humans.add(record)

    return len(humans)


def predict_df(df):
    st.title("Report")
    df.rename(columns={'TIME': 'time', 'UID': 'call_id', 'CALL TYPE': 'call_type', 'NET CALL TIME': 'call_duration', 'TYPE': 'attended_by'}, inplace=True)
    df = pd.get_dummies(df, columns=['call_type'])
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df.sort_values(by='time')

    df['call_duration'] = df['call_duration'].apply(time_string_to_secs)

    df['day_hour'] = df['time'].dt.hour
    df['hour_minute'] = df['time'].dt.minute

    X = df.drop(['attended_by', 'call_id', 'time'], axis=1).astype(np.float64)
    Y = df['attended_by'].copy()
    target_encoder = LabelEncoder()
    target_encoder.fit(Y)
    encoded_Y = target_encoder.transform(Y)

    Y = to_categorical(encoded_Y, num_classes=49)

    model = keras.models.load_model("model")


    X = X.values.reshape(-1, 1, X.shape[1])
    y_hat = model.predict(x=X)

    model_accuracy = accuracy_score(np.argmax(Y, axis=1), np.argmax(y_hat, axis=2))

    st.write(f"Model Accuracy: {model_accuracy*100:0.2f}")

    humans_required_predicted = calculate_humans_predicted(y_hat)
    humans_required = calculate_humans_predicted(Y)

    st.header("Humans required")
    st.write(f"Humans Required (Predicted): {humans_required_predicted}")
    st.write(f"Humans Required (Original): {humans_required}")

    st.header("Humans Required (Hourly)")

    hours = [i for i in range(0, 24)]

    humans_required_hourly  = [hourly_calculate(df[df['attended_by'] != "Bot"][df['day_hour'] == current_hour]['attended_by']) for current_hour in hours]

    df_data = np.stack([hours, humans_required_hourly], axis=1)

    chart_df = pd.DataFrame(df_data,
                            columns=["Hour of Day", "Humans Required"])
 
    fig = make_plot(chart_df,  "Hour of Day", "Humans Required", "Humans Required per hour")

    st.plotly_chart(fig)

    humans_hourly_table_data = np.stack([[f'{hour} to {(hour + 1)%24}' for hour in hours] , humans_required_hourly], axis=1)
    humans_hourly_table_df = pd.DataFrame(humans_hourly_table_data,
                                          columns=['Hour', 'Humans Required'])
    humans_hourly_table_df.set_index('Hour', inplace=True)

    st.write(humans_hourly_table_df)

    avg_call_duration = df[df['attended_by'] != 'Bot']['call_duration'].mean()

    st.header("Call durations")
    st.write(f"Average Call Duration (Human): {avg_call_duration:0.2f} seconds")

    avg_call_duration_hourly = [df[df['attended_by'] != "Bot"][df['day_hour'] == current_hour]['call_duration'].mean() for current_hour in hours]
    df_data = np.stack([hours, avg_call_duration_hourly], axis=1)
    chart_df = pd.DataFrame(df_data,
                            columns=["Hour of Day", "Avg Call Duration"])

    fig = make_plot(chart_df, "Hour of Day", "Avg Call Duration", "Average Call Duration per hour")    

    st.plotly_chart(fig)

    st.header("Call Type Durations")
    avg_inbound_call_duration = df[df['call_type_Inbound'] == 1]['call_duration'].mean()
    avg_inbound_call_duration_hourly  = [df[df['call_type_Inbound'] == 1][df['day_hour'] == current_hour]['call_duration'].mean() for current_hour in hours]
    
    avg_outbound_call_duration = df[df['call_type_Outbound'] == 1]['call_duration'].mean()
    avg_outbound_call_duration_hourly  = [df[df['call_type_Outbound'] == 1][df['day_hour'] == current_hour]['call_duration'].mean() for current_hour in hours]

    avg_manual_call_duration = df[df['call_type_Manual'] == 1]['call_duration'].mean()
    avg_manual_call_duration_hourly  = [df[df['call_type_Manual'] == 1][df['day_hour'] == current_hour]['call_duration'].mean() for current_hour in hours]

    st.write(f"Average Inbound Call Duration: {avg_inbound_call_duration:0.2f} seconds")
    st.write(f"Average Outbound Call Duration: {avg_outbound_call_duration:0.2f} seconds")    
    st.write(f"Average Manual Call Duration: {avg_manual_call_duration:0.2f} seconds")

    summary_data = np.stack([hours, avg_inbound_call_duration_hourly, avg_outbound_call_duration_hourly, avg_manual_call_duration_hourly], axis=1)

    call_type_summary = pd.DataFrame(summary_data,
                                    columns=["Hour of Day", "Inbound", "Outbound", "Manual"])

    fig = make_plot(call_type_summary, "Hour of Day", ['Inbound', 'Outbound', 'Manual'], "Average Call Duration for each call type")

    st.plotly_chart(fig)

    bots_original = df[df['attended_by'] == "Bot"].shape[0]
    humans_original = df[df['attended_by'] != "Bot"].shape[0]

    st.header("Calls Attended by Humans vs. Bots")

    st.write()

    st.write(f"Total calls attended by bots: {bots_original}")
    st.write(f"Total calls attended by humans: {humans_original}")

    humans_hourly = [df[df['attended_by'] != "Bot"][df['day_hour'] == current_hour].shape[0] for current_hour in hours]
    print(humans_hourly)
    bots_hourly = [df[df['attended_by'] == "Bot"][df['day_hour'] == current_hour].shape[0] for current_hour in hours]

    comparision_data = np.stack([hours, humans_hourly, bots_hourly], axis=1)

    chart_df = pd.DataFrame(comparision_data,
                               columns=['Hour of Day', 'Human', 'Bot'],
                               index=hours)

    fig = make_plot(chart_df, 'Hour of Day', ['Human', 'Bot'], 'Number of calls attended by Humans vs. Bots')

    st.plotly_chart(fig)

def predict_file():
    if(input_file is None):
        st.error('Please upload a file to proceed further')
    else:
        df = pd.read_csv(input_file)

        if (set(df.columns) != set(required_columns)):
            st.error(f'Please input a csv file with follwing columns\n{",".join(required_columns)}')
        else:
            predict_df(df)


input_file = st.file_uploader("Upload CSV", type='csv', )
st.button("Predict", on_click=predict_file)

