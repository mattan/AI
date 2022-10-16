import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import pymysql
# כאן הספריות שעושת את החישוב עצמו
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def write_to_db(df):
    """
    This is just put the data frame in the database

    Arguments:
        df: data frame

    Returns:
         nothing
    """
    try:
        table_name = "data_6_months"
        sql_engine = create_engine('mysql+pymysql://root:@127.0.0.1/test', pool_recycle=3600)
        db_connection = sql_engine.connect()
        df.to_sql(table_name, db_connection, if_exists='fail')
        db_connection.close()
    except ValueError as vx:
        # print(vx)
        pass
    except Exception as ex:
        # print(ex)
        pass
    else:
        print("המידע נרשם בטבלא")
    finally:
        pass


def read_from_db():
    """
    This is just read the data frame from the database

    Returns:
        df: data frame
    """
    try:
        table_name = "data_6_months"
        sql_engine = create_engine('mysql+pymysql://root:@127.0.0.1/test', pool_recycle=3600)
        db_connection = sql_engine.connect()
        df = pd.read_sql(f"select * from {table_name}", db_connection)
        db_connection.close()
    except ValueError as vx:
        # print(vx)
        pass
    except Exception as ex:
        # print(ex)
        pass
    else:
        print("המידע נרשם בטבלא")
    finally:
        return None
    return df


def create_random_data():
    """
    מייצר קבוצה של פעולות רגילות של 10 בנקראים ועוד בנקאי שמועל בסכומים קטנים של עד 10 שקל מחשבונות אקראיים

    Returns:
        df: data frame
    """
    df = pd.DataFrame(dict(SCHOOM=np.append(np.random.randint(0, 1000, 100), np.random.randint(0, 10, 1000)),
                           GOREM_MEASHER=np.append(np.random.randint(0, 10, 100), [11] * 1000),
                           TIME=np.append(np.random.randint(0, 100, 100), [10000] * 1000)))
    df.plot(kind='scatter', x='SCHOOM', y='GOREM_MEASHER', s='TIME', title="Before kmeans")
    # plt.show()
    return df


def kmeans(df):
    """
    מנסה לאתר את המעילה (בצבע אדום) - במקרה הזה יש יותר מדי false possitive כך שהאלגוריתם לא יעיל

    Returns:
        df: data frame
    """
    color_map = ['red', 'green', 'yellow', 'black', 'blue']

    vector = KMeans(n_clusters=2).fit_predict(df)
    kmeans_colors = list(map(lambda x: color_map[x], vector))
    df.plot(kind='scatter', x='SCHOOM', y='GOREM_MEASHER', color=kmeans_colors, title="After kmeans")
    plt.show()


def pca(df):
    pca_test = PCA(n_components=2)
    pca_test.fit(df)
    print(pca_test.explained_variance_ratio_)


if __name__ == '__main__':
    df = create_random_data()
    write_to_db(df)
    df = df if df is not None else read_from_db()
    kmeans(df)
    pca(df)
