#~~~~~~~~~~~~~<  Hey sister, go sister, where you get that flow sister? >~~~~~~~~~#

import numpy as np
import pandas as pd
from datetime import datetime
from IPython.display import display, Markdown, Latex
import acquire as aq

def show_funcs():
    '''
    Shows the functions available in this library
    '''
    display(Markdown('''
|Function: | Param:|Description: | Returns:|
|:---:|---:|---:|---:|
|col_to_datetime()| df= | Takes in the DataFrame from acquire.all_sales and returns it with sale_date as a datetime and weekday as an abrieviated day of the week| df|
|sale_date_to_datetime()| df= | Takes in the DataFrame returned by col_to_datetime and sets the sale_date column to the index | df|
|month_and_sales_total()| df= | Takes in the DataFrame returned by sale_date_to_datetime and makes a month column and sales total column| df|
|prepare_sales()| None | Uses the aquire.all_sales() function to get that DataFrame; runs all above functions to return a DataFrame with a weekday, month, and sales_total column, with the index set to the original sale_date column| df |
|prepare_opsd| None | Use the aquire.opsd to get the opsd data and makes it nice| df|
    '''))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  col_to_datetime  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def col_to_datetime(df):
    '''
    Convert date column to datetime format.
    
    Parameters:
    -----------
    
    df        :     The dataframe brought in by acquire.all_sales
    
    Returns:
    -----------
    
    df        :     The same dataframe, but with df.sale_date in DateTime, and df.weekday as the day of week
    
    Giddy-up!
    '''
    
    df.sale_date = df.sale_date.str.replace('00:00:00 GMT', '')


    split_date = df.sale_date.str.split(' ', n=1, expand=True)


    split_date[1] = split_date[1].str.strip()

    split_date[0] = split_date[0].str.replace(',','')

    df.sale_date = split_date[1]
    df['weekday'] = split_date[0]


    df.sale_date = pd.to_datetime(df.sale_date, format='%d %b %Y')
    
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  sale_date_to_index  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def sale_date_to_index(df):
    '''
    
    '''
    
    df = df.set_index(df.sale_date)
    
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  month_and_sales_total  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def month_and_sales_total(df):
    '''
    
    '''
    
    df['month']= df.index.month
    
    df['sales_total'] = df.sale_amount * df.item_price
    
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  prepare_sales  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def prepare_sales():
    '''
    
    '''
    
    df = aq.all_sales()
    
    df = col_to_datetime(df)
    
    df = sale_date_to_index(df)
    
    df = month_and_sales_total(df)
    
    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  prepare_opsd  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def prepare_opsd():
    '''
    
    '''
    
    df = aq.open_power_systems_data()    
    
    df.columns = df.columns.str.lower()    

    df.date = pd.to_datetime(df.date)

    df = df.set_index('date')

    df['month'] = df.index.month_name()

    df['year'] = df.index.year

    df = df.fillna(0)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<    >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<    >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<    >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#