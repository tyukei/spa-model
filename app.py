import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

upload_file = None
enable_hr = True
approxi_graph=False
labels_list = []
temp_mean = 0

def graph(PATH):
    # Load the data
    data = pd.read_csv(PATH)
    data['時刻'] = pd.to_datetime(data['時刻'])
    data['体表温度_平滑化'] = data['体表温度'].rolling(window=100,min_periods=1).mean()
    df_filled = data.dropna(subset=['脈周期[ms]'])
    df_filled.reset_index(drop=True, inplace=True)
    data['脈周期[ms]'] = data['脈周期[ms]'].fillna(df_filled['脈周期[ms]'].rolling(window=3, min_periods=1, center=True).mean())
    data['脈周期[ms]_平滑化'] = data['脈周期[ms]'].rolling(window=100,min_periods=1).mean()   

    # Initialize the plot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True)

    # Plot 体表温度
    axes[0].plot(data['時刻'], data['体表温度'], label='体表温度', color='blue')
    axes[0].set_ylabel('体表温度')
    axes[0].legend(loc='upper right')

    # Plot 体動
    axes[1].plot(data['時刻'], data['体動'], label='体動', color='green')
    axes[1].set_ylabel('体動')
    axes[1].legend(loc='upper right')

    # Plot 脈周期[ms]
    axes[2].plot(data['時刻'], data['脈周期[ms]'], label='脈周期[ms]', color='red')
    axes[2].set_ylabel('脈周期[ms]')
    axes[2].set_xlabel('時刻')
    axes[2].legend(loc='upper right')

    # Shade regions based on ラベル values
    labels = data['ラベル'].unique()
    colors = ['grey', 'yellow', 'blue', 'green']

    for i, ax in enumerate(axes):
        for j, label in enumerate(labels):
            mask = data['ラベル'] == label
            ax.fill_between(data['時刻'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask, color=colors[j], alpha=0.2, label=f'ラベル {label}')

    # Remove duplicate legends
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')

    # Set title and show plot
    plt.suptitle('Graphs of 体表温度, 体動, and 脈周期[ms] against 時刻', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    st.pyplot(fig)

def process_data(path):
    global temp_mean
    # CSVファイルを読み込む
    df = pd.read_csv(path)
    with st.expander("アップロードしたデータ"):
        st.dataframe(df)
    # 窓のサイズ
    window_size = 50
    # 体表温度を移動平均で平滑化
    df['体表温度_平滑化'] = df['体表温度'].rolling(window=window_size).mean().shift(-window_size + 1)
    df_filled = df.dropna(subset=['脈周期[ms]'])
    df_filled.reset_index(drop=True, inplace=True)
    df['脈周期[ms]'] = df['脈周期[ms]'].fillna(df_filled['脈周期[ms]'].rolling(window=3, min_periods=1, center=True).mean())
    df['hr_平滑化'] = df['脈周期[ms]'].rolling(window=window_size,min_periods=1).mean()

    temp_mean = df['体表温度_平滑化'].quantile(0.7)
    # 平滑化したデータの1階微分と2階微分を計算
    df['temp\'_平滑化'] = df['体表温度_平滑化'].diff()
    df['temp\'\'_平滑化'] = df['temp\'_平滑化'].diff()
    df['hr\'_平滑化'] = df['hr_平滑化'].diff()
    return df

def assign_label_v2(diff1, diff2):
    b1 = 0.001
    ave2_3 = 0.01635
    ave2_3_2 = 0.0011006
    ave3_4 = - 0.02700
    ave3_4_2 = 0.0002941
    ave4_2 = -0.00056 
    ave4_2_2 = 0.0007256

    if diff1 > ave4_2 and (diff2 < -b1 or (-ave4_2_2 <= diff2 <= ave4_2_2)):
        return 2
    elif (-ave2_3 <= diff1 <= ave2_3) and (ave2_3_2 < diff2 < ave2_3_2):
        return 3
    elif diff1 < -ave3_4:
        if diff2 < ave3_4_2:
            return 3
        elif diff2 >= ave3_4_2:
            return 4
    return None

def constrained_label_assignment(current_label, proposed_label, current_temp):
    global labels_list
    if current_label == 0 and proposed_label == 2:
        return proposed_label
    elif current_label == 2 and proposed_label == 3:
        if current_temp > temp_mean:
            return proposed_label
        else:
             for i in range(len(labels_list)-1, -1, -1):
                  if labels_list[i] == 2:
                       labels_list[i] = 4
                  else:
                       break
    elif current_label == 3 and proposed_label == 4:
        return proposed_label
    elif current_label == 4 and proposed_label == 2:    
            return proposed_label
    elif current_label == 3 and proposed_label == 2:
        for i in range(len(labels_list)-1, -1, -1):
            if labels_list[i] == 3:
                labels_list[i] = 2
            else:
                break
    else:
        return current_label

def label_for_segment(segment):
    last_row = segment.iloc[-1]
    return assign_label_v2(last_row['temp\'_平滑化'], last_row['temp\'\'_平滑化'])

def predict_labels(df):
    global labels_list
    previous_label = None
    sample_size = 10
    for i in range(0, len(df), sample_size):
        segment = df.iloc[i:i+sample_size]
        label = label_for_segment(segment)
        current_temp = segment.iloc[-1]['体表温度_平滑化']
        label = constrained_label_assignment(previous_label, label, current_temp)
        if label is None:
            if previous_label is None:
                label = 0
            else:
                label = previous_label
        labels_list.extend([label]*len(segment))
        previous_label = label

    df['ラベル'] = labels_list
    return df

def check_starttime(df):
    samelabel_count = 0
    is_start = False
    for i in range(1, len(df['ラベル'])):
        if df['ラベル'].iloc[i-1] == df['ラベル'].iloc[i]:
            samelabel_count += 1
        else:
            if is_start == False and df['ラベル'].iloc[i-1] != 2:
                for j in range(0, i):
                    df.loc[j, 'ラベル'] = 0
            elif df['ラベル'].iloc[i-1] == 2:
                is_start = True
            samelabel_count = 0
    return df

def check_lasttime(df):
    for i in range(len(df['ラベル']),1,-1):
        if df['ラベル'].iloc[i-1] == 2 or df['ラベル'].iloc[i-1] == 3:
            df.loc[i-1, 'ラベル'] = 0
        elif df['ラベル'].iloc[i-1] == 4:
            break
    return df

def add_0_label(df):
    gap = 50
    minimum_3 = 60
    minimum_4 = 150
    last_label_len = 0
    for i in range(1, len(df['ラベル'])):
        last_label_len += 1
        if df['ラベル'].iloc[i-1] == 0 and df['ラベル'].iloc[i] == 2:
            df.loc[df.index[i-gap:i], 'ラベル'] = 0
        # ラベルが2から3に変わる時
        elif df['ラベル'].iloc[i-1] == 2 and df['ラベル'].iloc[i] == 3:
            df.loc[df.index[i-gap:i], 'ラベル'] = 0
            last_label_len = 0
        # ラベルが3から4に変わる時
        elif df['ラベル'].iloc[i-1] == 3 and df['ラベル'].iloc[i] == 4:
            if last_label_len < minimum_3:
                df.loc[df.index[i-last_label_len:i-last_label_len+minimum_3], 'ラベル'] = 3
            else:
                df.loc[df.index[i:i+gap], 'ラベル'] = 0
                last_label_len = 0
        # ラベルが4から2に変わる時
        elif df['ラベル'].iloc[i-1] == 4 and df['ラベル'].iloc[i] == 2:
            if enable_hr:
                hr_diff_ave = sum(df.iloc[i-gap:i]['hr\'_平滑化']) / gap
                if last_label_len < 120:
                    df.loc[df.index[i-last_label_len:i-last_label_len+120], 'ラベル'] = 4
                elif hr_diff_ave > -0.2:
                    df.loc[df.index[i-last_label_len:i], 'ラベル'] = 4
                    print(f"hr_diff>0: {i}")
                    j = i
                    while hr_diff_ave > -0.3:
                        j += gap
                        hr_diff_ave = sum(df.iloc[j-gap:j]['hr\'_平滑化']) / gap
                        df.loc[df.index[j-gap:j], 'ラベル'] = 4
                else:
                    print(f"hr_diff_ave: {hr_diff_ave}")
                    df.loc[df.index[i:i+gap], 'ラベル'] = 0
                    last_label_len = 0
            else:
                if last_label_len < minimum_4:
                    df.loc[df.index[i-last_label_len:i-last_label_len+minimum_4], 'ラベル'] = 4
                else:
                    df.loc[df.index[i:i+gap], 'ラベル'] = 0
                    last_label_len = 0
    return df

def output(df):
    df = df[['時刻', '体表温度', '体動', '脈周期[ms]', 'ラベル']]
    RESULT_PATH = "result.csv"
    df.to_csv(RESULT_PATH, index=False)
    with st.expander("処理後のデータ"):
        st.dataframe(df)
    graph(RESULT_PATH)
    with open(RESULT_PATH, "rb") as f:
        st.download_button(
			label="Download result.csv",
			data=f,
			file_name="result.csv",
			mime="text/csv")



def init_uis():
    global upload_file,enable_hr
    st.title("サウナラベル付けアプリ🧖")
    upload_file = st.file_uploader("csvをアップロードしてください", type="csv")
    enable_hr = st.sidebar.checkbox("脈周期",value=True)

    
def on_upload():
    if upload_file is not None:
        with st.spinner("アップロードしたファイルを処理しています"):
            df = process_data(upload_file)
            df1=predict_labels(df)
            df2=check_starttime(df1)
            df3=check_lasttime(df2)
            df4=add_0_label(df3)
            output(df4)
        
def on_check():
    if enable_hr:
        on_upload()
    else:
        on_upload()

def main():
    init_uis()
    on_check()
    

if __name__ == "__main__":
    main()