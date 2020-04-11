import pandas as pd


def preprocess_datasets_datadista(raw_df):
    """ Repo: https://github.com/datadista/datasets.git
    """
    raw_df.index = raw_df['CCAA']

    raw_df = raw_df[[item for item in raw_df.columns if item not in ['cod_ine', 'CCAA']]]
    raw_df = raw_df.transpose()

    raw_df.index = pd.to_datetime(raw_df.index, format="%Y/%m/%d")

    return raw_df


def get_data_datadista(csv_path):
    """ Repo: https://github.com/datadista/datasets.git
    """
    df = pd.read_csv(csv_path, sep=",")
    return preprocess_datasets_datadista(df)


cases = get_data_datadista(r"https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_casos.csv")
critical = get_data_datadista(r"https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_uci.csv")
dead = get_data_datadista(r"https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_fallecidos.csv")
recovered = get_data_datadista(r"https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_altas.csv")
