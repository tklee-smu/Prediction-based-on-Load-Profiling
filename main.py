import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

#15분 단위 96 timestep 일일 예측 & 1시간 단위 4 timestep 일일 예측
def prediction(filename, inputname):
 
    tem_in = pd.read_csv(inputname, header=0, names=['TEM'])
    tem_in = np.array(tem_in)

    df = pd.read_csv(filename, encoding='cp949', sep='\\\t', engine='python', index_col=0)
    df_MP = df.loc['"1']
    df_TEM = df.loc['"859']

    df_MP = df_MP.iloc[:,10:12]
    df_TEM = df_TEM.iloc[:, 10:12 ]
    df_MP.columns = ['Datetime', 'MP']
    df_TEM.columns = ['Datetime', 'TEM']

    df_MP.index = pd.to_datetime(df_MP['Datetime'])
    df_TEM.index = pd.to_datetime(df_TEM['Datetime'])

    df_con = pd.concat([df_MP['MP'], df_TEM['TEM']], axis=1)

    df_day = pd.concat([df_MP['Datetime'], df_MP['MP'], df_TEM['TEM']], axis=1)
    df_day['Datetime'] = pd.to_datetime(df_day['Datetime'])
    df_day = df_day.groupby(df_day['Datetime'].dt.floor('d')).apply(func_groupby)

    df_hour = pd.concat([df_MP['Datetime'], df_MP['MP'], df_TEM['TEM']], axis=1)
    df_hour['Datetime'] = pd.to_datetime(df_hour['Datetime'])
    df_hour = df_hour.groupby(df_hour['Datetime'].dt.floor('h')).apply(func_groupby)

    df_con = df_con.reset_index(drop=True)
    df_con = df_con.loc[((df_con['MP'] != 0) & (df_con['TEM'] != 0))]
    df_con = df_con.dropna(axis=0)

    MP = IQR(df_con['MP'])
    TEM = IQR(df_con['TEM'])

    MP_in = MP[1:]
    TEM_in = TEM[1:]
    MP_lag_in = pd.DataFrame(MP[:].values)

    MP_in = MP_in.reset_index(drop=True)
    TEM_in = TEM_in.reset_index(drop=True)
    MP_lag_in = MP_lag_in.reset_index(drop=True)

    input = pd.concat([MP_in, TEM_in, MP_lag_in], axis=1, ignore_index=True)
    input.columns = ['y', 'x1', 'x2']
    input = input.dropna(0)

    x = input.drop('y', axis=1)
    y = input['y']

    x = np.array(x).reshape(-1, 2)
    y = np.array(y).reshape(-1, 1)

    # ------------------------------------------------------------------------
    # daily df
    # ------------------------------------------------------------------------

    df_day = df_day.reset_index(drop=True)
    df_day = df_day.loc[((df_day['MP'] != 0) & (df_day['TEM'] != 0))]
    df_day = df_day.dropna(axis=0)

    MP_d = IQR(df_day['MP'])
    TEM_d = IQR(df_day['TEM'])

    MP_in_d = MP_d[1:]
    TEM_in_d = TEM_d[1:]
    MP_lag_in_d = pd.DataFrame(MP_d[:].values)

    MP_in_d = MP_in_d.reset_index(drop=True)
    TEM_in_d = TEM_in_d.reset_index(drop=True)
    MP_lag_in_d = MP_lag_in_d.reset_index(drop=True)

    input_d = pd.concat([MP_in_d, TEM_in_d, MP_lag_in_d], axis=1, ignore_index=True)
    input_d.columns = ['y', 'x1', 'x2']
    input_d = input_d.dropna(0)
                    
    x_d = input_d.drop('y', axis=1)
    y_d = input_d['y']

    x_d = np.array(x_d).reshape(-1, 2)
    y_d= np.array(y_d).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(x_d, y_d)

    w_d = lr.coef_[0]
    b_d = lr.intercept_

    pred_d = []
    x2_d_update = np.array(y_d[-1:]).reshape(-1,1)

    for i in range(len(tem_in)):
        x2_d = np.array(x2_d_update)
        out_d = w_d[0]*tem_in[i]+ w_d[1]*x2_d + b_d
        out_d = out_d.tolist()
        pred_d.append(out_d)
        x2_d_update = out_d
        # print(x2)

    pred_d = np.array(pred_d).reshape(-1, 1)

    # ------------------------------------------------------------------------
    # hourly df
    # ------------------------------------------------------------------------
    
    df_hour = df_hour.reset_index(drop=True)
    df_hour = df_hour.loc[((df_hour['MP'] != 0) & (df_hour['TEM'] != 0))]
    df_hour = df_hour.dropna(axis=0)

    MP_h = IQR(df_hour['MP'])
    TEM_h = IQR(df_hour['TEM'])

    MP_in_h = MP_h[1:]
    TEM_in_h = TEM_h[1:]
    MP_lag_in_h = pd.DataFrame(MP_h[:].values)

    MP_in_h = MP_in_h.reset_index(drop=True)
    TEM_in_h = TEM_in_h.reset_index(drop=True)
    MP_lag_in_h = MP_lag_in_h.reset_index(drop=True)

    input_h = pd.concat([MP_in_h, TEM_in_h, MP_lag_in_h], axis=1, ignore_index=True)
    input_h.columns = ['y', 'x1', 'x2']
    input_h = input_h.dropna(0)
                    
    x_h = input_h.drop('y', axis=1)
    y_h = input_h['y']

    x_h = np.array(x_h).reshape(-1, 2)
    y_h= np.array(y_h).reshape(-1, 1)

    lr_h = LinearRegression()
    lr_h.fit(x_h, y_h)

    w_h = lr_h.coef_[0]
    b_h = lr_h.intercept_

    pred_h = []
    x2_h_update = np.array(y_h[-1:]).reshape(-1,1)

    for i in range(len(tem_in)):
        x2_h = np.array(x2_h_update)
        out_h = w_h[0]*tem_in[i]+ w_h[1]*x2_h + b_h
        out_h = out_h.tolist()
        pred_h.append(out_h)
        x2_h_update = out_h
        # print(x2)

    pred_h = np.array(pred_h).reshape(-1, 1)

    # ------------------------------------------------------------------------
    # Load profiling : daily 15min timestep
    # ------------------------------------------------------------------------
    # #clusters
    seed = 0
    NC = 2

    in_list = y
    e_in = in_list.reshape(-1, 1)
    scaler = MinMaxScaler()
    e_in = scaler.fit_transform(e_in)
    e_in_r = np.round(e_in, 4)
    
    ratio_15min = len(e_in_r)
    for i in range(1, 96):
        if (ratio_15min%96 != 0):
            ratio_15min = ratio_15min - 1
        else:
            continue
        
    e_in_s = pd.DataFrame(e_in_r)[:ratio_15min]
    X_e = np.array(e_in_s).reshape(-1, 96)
    km_e, cln_e = kmeans_in(X_e, NC, seed)
    TP_e = np.array([km_e.cluster_centers_[yi] for yi in range(NC)])
    TP_e = TP_e.reshape(NC, 96)
    result = np.zeros(shape=(len(pred_d), 96))

    #Division of days
    datetime_Date = datetime.today()
    weekday = datetime_Date.weekday()

    if (weekday == 6):
        km_in = km_e.cluster_centers_[1]
    elif (weekday == 5):
        km_in = km_e.cluster_centers_[0]
    elif (weekday == 4):
        km_in = km_e.cluster_centers_[0]
    else:
        km_in = km_e.cluster_centers_[1]
    for i in range(len(pred_d)):
        result[i] = np.array(km_in/np.sum(km_in) * pred_d[i]).reshape(-1, 96)

    result = pd.DataFrame(result[0], columns=['15min step prediction'])

    # ------------------------------------------------------------------------
    # Load profiling : daily 1hour timestep
    # ------------------------------------------------------------------------
    in_list_h = y_h
    e_in_h = in_list_h.reshape(-1, 1)
    scaler_h = MinMaxScaler()
    e_in_h = scaler_h.fit_transform(e_in_h)
    e_in_r_h = np.round(e_in_h, 4)

    ratio_1h = len(e_in_r_h)
    for i in range(1, 24):
        if (ratio_1h%24 != 0):
            ratio_1h = ratio_1h - 1
        else:
            continue

    e_in_s_h = pd.DataFrame(e_in_r_h)[:ratio_1h]
    X_e_h = np.array(e_in_s_h).reshape(-1, 24)
    km_e_h, cln_e_h = kmeans_in(X_e_h, NC, seed)
    TP_e_h = np.array([km_e_h.cluster_centers_[yi] for yi in range(NC)])
    TP_e_h = TP_e_h.reshape(NC, 24)
    result_h = np.zeros(shape=(len(pred_h), 24))

    #Division of days
    datetime_Date = datetime.today()
    weekday = datetime_Date.weekday()

    if (weekday == 6):
        km_in_h = km_e_h.cluster_centers_[1]
    elif (weekday == 5):
        km_in_h = km_e_h.cluster_centers_[0]
    elif (weekday == 4):
        km_in_h = km_e_h.cluster_centers_[0]
    else:
        km_in_h = km_e_h.cluster_centers_[1]

    for i in range(len(pred_h)):
        result_h[i] = np.array(km_in_h/np.sum(km_in_h) * pred_d[i]).reshape(-1, 24)

    result_h = pd.DataFrame(result_h[0], columns=['1hour step prediction'])

    return result, result_h

#groupby
def func_groupby(x):
    d = {}
    d['MP'] = x['MP'].sum()
    d['TEM'] = x['TEM'].mean()
    return pd.Series(d, index=['MP', 'TEM'])

#Imputation
def IQR(df_in):
    avg = np.average(df_in)
    sd = np.std(df_in)
    z = (df_in - avg)/sd
    Q3, Q1 = np.percentile(z, [75, 25])
    IQR = Q3-Q1
    outlier = z[(Q1-1.5*IQR > z) | (Q3+1.5*IQR < z)]
    outindex = outlier.index
    z_rev = z

    for i in outindex:
        if (z_rev[i] > np.max(z_rev[z_rev != z_rev[i]])):
            z_rev[i] = np.average(z_rev[z_rev != z_rev[i]])
            
        elif (z_rev[i] < np.min(z_rev[z_rev != z_rev[i]])):
            z_rev[i] = np.average(z_rev[z_rev != z_rev[i]])

        elif (z_rev[i] == 'Null'):
            z_rev[i] = np.average(z_rev)

    out = z_rev * sd + avg
    return out

# k-means method
def kmeans_in(X, NC, seed):
    print("Euclidean k-means processing")
    km = TimeSeriesKMeans(n_clusters=NC, 
                        max_iter = 1000, 
                        metric = 'euclidean', 
                        random_state=seed)
    cluster_num = km.fit_predict(X)
    return km, cluster_num


if __name__ == "__main__":
    
    filename = 'test.csv' #전력사용량 데이터(.csv) 원본
    inputname = 'input.csv' #일일 평균온도값
    output_m, output_h = prediction(filename, inputname)
    print(output_m) # 15분 단위 일일 예측
    print(output_h) # 1시간 단위 일일 예측
    output_m.to_csv('result_15m_step.csv')
    output_h.to_csv('result_1h_step.csv')