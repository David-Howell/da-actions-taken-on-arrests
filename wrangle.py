import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import harmonic_mean

# modeling methods

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import explained_variance_score

from IPython.display import display, Markdown, Latex


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  wrangle_da >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def wrangle_da():
    df = pd.read_csv('dataset.csv')

    df = df.dropna()

    

    df = df.drop(columns=['incident_number', 'data_loaded_at', 'data_as_of'])

    df = df.drop(columns='list_of_booked_charges')

    df.arrest_date = pd.to_datetime(df.arrest_date)

    for i in df.arresting_agency.unique():
        if len(df[df.arresting_agency == i]) < 1000:
            df= df.drop(index=df.index[df.arresting_agency == i])

    df = df.reset_index(drop=True)
    
    df_time = df.set_index('arrest_date')

    df_time['year'] = df_time.index.year

    df['year'] = df_time.index.year
    
    df['month'] = df_time.index.month
    
    df_time['weekday'] = df_time.index.day_of_week
    
    df['weekend'] = df_time.weekday.map({5:1, 6:1, 0:0, 1:0, 2:0, 3:0, 4:0}).reset_index().weekday
    
    df.da_action_taken = df.da_action_taken.astype('category')

    df.da_action_taken =\
    df.da_action_taken.cat.set_categories(
        ['Discharged', 'Further Investigation Requested', 
         'MTR/Referred to Other Agency', 'Charges Filed', 'Other Action'])
    
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SUMMARIZE V.1.1 >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def summarize(df, cat_cols= None, too_long= 50, show_all= False, q= 10):
    '''
    Takes in a DataFrame and provides a summary of whats going on

    Parameters:
    ------------
    a Dataframe :
    
    cat_cols= None  : if you have them, otherwise they will be taken from object column types
    
    show_all= False :  if set to true:
                       it will print out the longest of the 
                       long markdown dataframes...BEWARE
    
    too_long= 50    : how many rows are too many rows in column.value_counts()??
    
    q = 10          : how many quartiles should numerical data be divided into?

    no return: just prints the:
    --------------
    df.head()
    df.info()
    df.describe()
    
    Null Values:
        By Column:
        By Row:

    df[each_separate_column].value_counts()
        Note: The object variables are not binned, the numerical variables are
        v1.1 -- Now you can control how much is displayed in long Markdown
    '''
#     print('test')
    display(Markdown(
    f'''
    DataFrame .head():
    -----------------
{df.head().to_markdown()}
    
    DataFrame .info():
    -----------------\n'''))
    print(df.info())
    display(Markdown(f'''\n    
    DataFrame .describe():
    -----------------
{df.describe().T.to_markdown()}
    
    Null Value Assessments:
    -----------------
        
        Nulls By Column:
{nulls_by_col(df).to_markdown()}
    -----------------
        
        Nulls By Row:
{nulls_by_row(df).to_markdown()}
    
    DataFrame .value_counts():
    -----------------
    
    '''))
    column_value_counts(df,
                        cat_cols= cat_cols, 
                        too_long= too_long, 
                        show_all= show_all, 
                        q=q
                       )                    
                    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  COLUMN_VALUE_COUNTS >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def column_value_counts(df, cat_cols=None, too_long=50, show_all=False, q= 10):
    
    if cat_cols == None:
        num_cols = [col for col in df.columns if df[col].dtype != 'O']
        cat_cols = [col for col in df.columns if col not in num_cols]

    for col in df.columns:
        
        print('Column Name: ', col,'\n--------------')
        
        if col in cat_cols:
            print_this = df[col].value_counts(dropna=False)
            
            print('Categorical:\n - ', len(print_this), 'Categories')
            
            if (len(print_this) < too_long) | (show_all == True):
                
                display(Markdown(print_this.to_markdown()))
            
            else:
                print('\n',print_this)
            
            print('\n\n-------------')
        else: 
            print_this = df[col].value_counts(bins= q, sort=False, dropna=False)
            
            print('Numerical: Divided by Quartile\n - ', len(print_this), 'bins')
            
            if (len(print_this) < too_long) | (show_all == True):
                
                display(Markdown(print_this.to_markdown()))
            
            else:
                print(print_this)
                
            print('\n\n-------------')
    print('-----------------\n---End of Line---')

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  REMOVE_OUTLIERS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def remove_outliers(df, k=1.5, col_list=[]):
    ''' 
    Removes outliers from a list of columns in a dataframe 
    and return that dataframe
    
    PARAMETERS:
    ------------
    
    df    :   DataFrame that you want outliers removed from
    
    k     :   The scaler of IQR you want to use for tromming outliers
                 k = 1.5 gives a 8σ total range
    col_list : The columns to have outliers removed using
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df = df.drop(columns=['outlier'])
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  NULLS_BY_ROW  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def nulls_by_row(df):
    '''
    Takes in a DataFrame and tells us the number of rows with missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    prnt_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing,
                             'percent_missing': prnt_missing,
                            })\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_missing'])\
    .count().reset_index().rename(columns={'index': 'count'})
    return rows_missing

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  NULLS_BY_COL  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def nulls_by_col(df):
    '''
    
    '''
#     
    num_missing = df.isnull().sum(axis=0)
    prnt_missing = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing,
                             'percent_missing': prnt_missing,
                            })
    null_cols = cols_missing[cols_missing['num_rows_missing'] > 0]
    print(f'Number of Columns with nulls: {len(null_cols)}')
    return null_cols


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_DB_URL  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_db_url(schema):
    import env
    user = env.username
    password = env.password
    host = env.host
    conn = f'mysql+pymysql://{user}:{password}@{host}/{schema}'
    return conn

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GDB! (Get DataBase) >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def gdb(db_name, query):
    '''
    gdb(db_name, query):
    
        takes in    a (db_name) schema name from the codeup database ;dtype int
        and         a (query) to the MySQL server ;dtype int

        and         returns the query using pd.read_sql(query, url)
        having      created the url from my environment file
    '''
    from pandas import read_sql
    url = get_db_url(db_name)
    return read_sql(query, url)