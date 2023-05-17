import matplotlib.pyplot as plt
from flask import *
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'xls'}

lstm_model = keras.models.load_model('static/model')
threshold = 0.00241

scaler = MinMaxScaler(feature_range=(0, 1))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'qwerty'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files.get('file')
        global data_filename
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] =\
            os.path.join(app.config['UPLOAD_FOLDER'],
                         data_filename)
        return render_template('index2.html')
    return render_template('index.html')


def create_figure(data):
    aacg = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
    if aacg.shape[0] % 5:
        while not aacg.shape[0] % 5:
            aacg = aacg.iloc[:-1]

    f = plt.figure(figsize=(30, 5), dpi=50)

    # OPEN
    f.add_subplot(1, 4, 1, title='OPEN')
    plt.plot(aacg.index, aacg['open'], color='b', linestyle=':', linewidth=1)
    plt.plot(aacg.index, aacg['open'], color='r', linestyle=':', linewidth=1)

    # CLOSE
    f.add_subplot(1, 4, 2, title='CLOSE')
    plt.plot(aacg.index, aacg['close'], color='b', linestyle=':', linewidth=1)
    plt.plot(aacg.index, aacg['close'], color='r', linestyle=':', linewidth=1)

    # HIGH
    f.add_subplot(1, 4, 3, title='HIGH')
    plt.plot(aacg.index, aacg['high'], color='b', linestyle=':', linewidth=1)
    plt.plot(aacg.index, aacg['high'], color='r', linestyle=':', linewidth=1)

    # LOW
    f.add_subplot(1, 4, 4, title='LOW')
    plt.plot(aacg.index, aacg['low'], color='b', linestyle=':', linewidth=1)
    plt.plot(aacg.index, aacg['low'], color='r', linestyle=':', linewidth=1)
    plt.savefig('static/imgs/pic1.png')


@app.route('/show_data')
def show_data():
    data_file_path = session.get('uploaded_data_file_path', None)
    uploaded_df = pd.read_csv(data_file_path)
    create_figure(uploaded_df)
    return render_template('show_csv_data.html',
                           data_var=uploaded_df.head(10).to_html())


@app.route('/show_res')
def show_res():
    data_file_path = session.get('uploaded_data_file_path', None)

    uploaded_df_processing = pd.read_csv(data_file_path)[['date', 'open', 'high', 'low', 'close']]
    if uploaded_df_processing.shape[0] % 5 !=0:
        while uploaded_df_processing.shape[0] % 5 != 0:
            uploaded_df_processing = uploaded_df_processing.iloc[:-1]

    uploaded_df_processing['date'] = pd.to_datetime(uploaded_df_processing['date'])
    uploaded_df_processing['year'] = uploaded_df_processing['date'].dt.year
    uploaded_df_processing['month'] = uploaded_df_processing['date'].dt.month
    uploaded_df_processing['day'] = uploaded_df_processing['date'].dt.day

    uploaded_df_processing.index = uploaded_df_processing['date']

    uploaded_df_processing.drop(columns='date', inplace=True)

    model_data = scaler.fit_transform(uploaded_df_processing)
    model_data_df = pd.DataFrame(model_data,
                                 columns=uploaded_df_processing.columns)
    model_data = model_data.reshape(model_data.shape[0]//5, 5, model_data.shape[1])
    predicted_data = lstm_model.predict(model_data)
    predicted_data = predicted_data.reshape(predicted_data.shape[0]*predicted_data.shape[1], predicted_data.shape[2])
    model_data = model_data.reshape(model_data.shape[0]*model_data.shape[1], model_data.shape[2])
    model_data = model_data[:, :4]

    # mae-threshold
    scored2 = pd.DataFrame(index=uploaded_df_processing.index)
    scored2['Loss_mae'] = np.mean(np.abs(predicted_data-model_data), axis=1)
    scored2['Threshold'] = threshold
    scored2['Anomaly'] = scored2['Loss_mae'] > scored2['Threshold']
    scored2['Anomaly_detect'] = scored2[['Loss_mae', 'Anomaly']].apply(lambda x: None if not x[1] else x[0], axis=1)
    scored2.plot(logy=True, rot=45)
    plt.savefig('static/imgs/pic2.png')

    # real/pred
    f1 = plt.figure(figsize=(30, 5), dpi=50)

    # OPEN
    f1.add_subplot(2, 4, 1, title='OPEN')
    plt.plot(uploaded_df_processing.index, model_data_df['open'], color='b', linestyle=':', linewidth=1)
    plt.plot(uploaded_df_processing.index, predicted_data[:, 0], color='r', linestyle=':', linewidth=1)

    # CLOSE
    f1.add_subplot(2, 4, 2, title='CLOSE')
    plt.plot(uploaded_df_processing.index, model_data_df['close'], color='b', linestyle=':', linewidth=1)
    plt.plot(uploaded_df_processing.index, predicted_data[:, 1], color='r', linestyle=':', linewidth=1)

    # HIGH
    f1.add_subplot(2, 4, 3, title='HIGH')
    plt.plot(uploaded_df_processing.index, model_data_df['high'], color='b', linestyle=':', linewidth=1)
    plt.plot(uploaded_df_processing.index, predicted_data[:, 2], color='r', linestyle=':', linewidth=1)

    # LOW
    f1.add_subplot(2, 4, 4, title='LOW')
    plt.plot(uploaded_df_processing.index, model_data_df['low'], color='b', linestyle=':', linewidth=1)
    plt.plot(uploaded_df_processing.index, predicted_data[:, 3], color='r', linestyle=':', linewidth=1)

    f1.savefig('static/imgs/pic3.png')

    # detect anomaly
    anomaly_checked = pd.DataFrame(model_data, index=scored2.index, columns=['open', 'high', 'low', 'close'])
    detected_anom = anomaly_checked[anomaly_checked.index.isin(scored2[~scored2['Anomaly_detect'].isnull()].index)]

    f2 = plt.figure(6, figsize=(30, 10), dpi=50)
    plt.plot(anomaly_checked.mean(axis=1))
    plt.scatter(detected_anom['open'].index, detected_anom.mean(axis=1), marker='x', c='red')
    f2.savefig('static/imgs/pic4.png')

    return render_template('show_preprocessing.html',
                           data_var=pd.DataFrame(predicted_data,
                                                 columns=[['open', 'high', 'low', 'close']]).head(10).to_html())


if __name__ == '__main__':
    app.run(debug=True)
