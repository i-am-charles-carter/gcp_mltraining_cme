import pandas as pd


def create_ohlcv(df):
    # Create a helper column 'group' that assigns a group to every 10 rows
    df['group'] = df.index // 10
    # Calculate OHLC values
    ohlc = df.groupby('group')['ES_price'].agg(['first', 'max', 'min', 'last'])
    # Rename the columns
    ohlc.columns = ['open', 'high', 'low', 'close']
    # Calculate volume
    volume = df.groupby('group')['ES_volume'].sum()
    # Get the first datetime from each group
    timestamps = df.groupby('group')['datetime'].last()
    # Combine OHLC, Volume and Timestamps into one DataFrame
    ohlcv = pd.concat([timestamps, ohlc, volume], axis=1)
    ohlcv.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    return ohlcv


def get_ohlcv():
    # Load and preprocess data
    # data = pd.read_feather(r"C:\Users\charl\Desktop\Tick\ES_TICK_HOL_FILTERED_08_18.feather")
    data = pd.read_feather(r"C:\Users\charl\Desktop\Tick\ES_TICK_HOL_FILTERED_08_18_1Day_2022_07_01_2022_07_15.feather")
    # Group by date and apply the create_ohlcv function to each group
    data = data.groupby(data['datetime'].dt.date).apply(create_ohlcv)
    # Reset the index
    data.reset_index(drop=True, inplace=True)
    # Calculate the number of features in the dataset
    data = data.set_index("datetime")
    data = data[['close']]
    return data
