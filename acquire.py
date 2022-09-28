import os
import math
import pandas as pd
import numpy as np
import requests

pznet_urls = {'base': 'https://python.zgulde.net/'
             ,'items': '/api/v1/items'
             ,'stores': '/api/v1/stores'
             ,'sales': '/api/v1/sales'}



def pznet_api(db):
    """
    Throw me the name of the data you want, and I'll throw you a DataFrame full of it
        - The wise old Data Scientist
        
    ---------
    
    Parameters:
    ----------
    
    db        :  Should be a string... WHERE db IN ['items', 'stores', 'sales']
    
    Returns:
    ----------
    
    a DataFrame with the python.zgulde.net data for each of the options for db
    
    
    Note:  because of caching, if you have a file in your folder == db, and it's a pickle
            that's what you'll get, instead of any new data. 
            If you have or prefer a csv... write your own code
    """
    
    # Set the filename for caching
    filename= db
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:
    
    #     set the url
        url = pznet_urls['base'] + pznet_urls[db]
#         set response as the payload dictionary
#             We'll use this to call the dictionary for our data
#             as well as max_page, page, and next_page to get further data
        response = requests.get(url).json()['payload']
    
#         we set df to a list so that it can be appended until we have all the data
        df = []
    
#         response_n is the iterable response dictionary of response
#             if response_n is the last page, we will break our loop
#             after appending the information from that page
        response_n = response
    
#         Start the Loop!!
#             in range of the max_page, so we can iterate through each one
        for i in range(response['max_page']):
        
            for i in response_n[db]:
            
                df.append(i)
            
            if response_n['next_page'] != None:

                next_page = response_n['next_page']
                # print(next_page)
#             the response_n is reset to the next page 
                # by using the url 'base' and adding the next_page
                response_n = requests.get(pznet_urls['base'] + next_page).json()['payload']

#         make our list of dictionaries into a dataframe
        df = pd.DataFrame(df)

#         pickle that dataframe for cache! 
        df.to_pickle(filename)

    return df



# items = []

# ir_n = ir
# for i in range(ir['max_page']):

#     # while ir_n['items'] != None:
        
#         for i in ir_n['items']:
            
#             items.append(i)#(ir_n['items'])

#         # if ir_n['page'] == i+1: break

#         if ir_n['next_page'] != None:

#             next_page = ir_n['next_page']
#             print(next_page)
#             ir_n = requests.get('https://python.zgulde.net' + next_page).json()['payload']
#             print(ir_n.keys())
# # items = pd.DataFrame(items)

# # for i in items:
# #     items = i

# items = pd.DataFrame(items)

# items


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<

# def pznet_api(db):
#     """
#     Throw me the name of the data you want, and I'll throw you a DataFrame full of it
#         - The wise old Data Scientist
        
#     ---------
    
#     Parameters:
#     ----------
    
#     db        :  Should be a string... WHERE db IN ['items', 'stores', 'sales']
    
#     Returns:
#     ----------
    
#     a DataFrame with the python.zgulde.net data for each of the options for db
    
    
#     Note:  because of caching, if you have a file in your folder == db, and it's a pickle
#             that's what you'll get, instead of any new data. 
#             If you have or prefer a csv... write your own code
#     """
    
#     # Set the filename for caching
#     filename= db
    
#     # if the file is available locally, read it
#     if os.path.isfile(filename):
#         df = pd.read_pickle(filename)
        
#     else:
    
#     #     set the url
#         url = pznet_urls['base'] + pznet_urls[db]
# #         set response as the payload dictionary
# #             We'll use this to call the dictionary for our data
# #             as well as max_page, page, and next_page to get further data
#         response = requests.get(url).json()['payload']
    
# #         we set df to a list so that it can be appended until we have all the data
#         df = []
    
# #         response_n is the iterable response dictionary of response
# #             if response_n is the last page, we will break our loop
# #             after appending the information from that page
#         response_n = response
    
# #         Start the Loop!!
# #             in range of the max_page, so we can iterate through each one
#         for i in range(response['max_page']):

# #             append the response string with the data
#             df.append(response_n[db])
    
# #             break the loop if we're on the last page
#             if response_n['page'] == i+1: break

# #             next_page will grab the rest of the url for the next page
#             next_page = response_n['next_page']
    
# #             the response_n is reset to the next page 
#                 # by using the url 'base' and adding the next_page
#             response_n = requests.get(pznet_urls['base'] + next_page).json()

# #         the list is in a list... let's break it out
#         for i in df:
#             df = i

# #         make our list of dictionaries into a dataframe
#         df = pd.DataFrame(df)

# #         pickle that dataframe for cache! 
#         df.to_pickle(filename)

#     return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~< all_sales >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def all_sales():
    """
    Throw me the name of the data you want, and I'll throw you a DataFrame full of it
        - The wise old Data Scientist
        
    ---------
    
    Parameters:
    ----------
    
    None!! Just run it and have fun
    
    Returns:
    ----------
    
    a DataFrame with the python.zgulde.net data for all of: 
    
    ['items', 'stores', 'sales'] joined together in holy... joinery??
    
    LEFT JOIN sales, stores ON sales.store = stores.store_id
    LEFT JOIN (above), items ON sales.item = items.item_id
    
    Didn't use SQL but if I had, then that's the idea I was working from.
    
    
    Note:  because of caching, if you have a file in your folder == all_sales, and it's a pickle
            that's what you'll get, instead of any new data. 
            If you have or prefer a csv... write your own code
    """
    
    # Set the filename for caching
    filename= 'all_sales'
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:
        
        sales = pznet_api('sales')
        
        items = pznet_api('items')
        
        stores = pznet_api('stores')
        
        df = sales.join(stores.set_index('store_id'), on='store')\
.join(items.set_index('item_id'), on='item')
        
        df.to_pickle(filename)
        
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def open_power_systems_data():
    """
    Throw me the name of the data you want, and I'll throw you a DataFrame full of it
        - The wise old Data Scientist
        
    ---------
    
    Parameters:
    ----------
    
    None!! Just run it and have fun
    
    Returns:
    ----------
    
    a DataFrame with the Open Power Systems Data for Germany
    
    Note:  because of caching, if you have a file in your folder == open_power_systems_data,
            and it's a pickle that's what you'll get, instead of any new data. 
            
            If you have or prefer a csv... write your own code
            
    # END OF LINE 
    -----------
    """
    # Set the filename for caching
    filename= 'open_power_systems_data'
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:
        csv = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'

        df = pd.read_csv(csv, parse_dates=True, infer_datetime_format=True)

#         pickle that dataframe for cache! 
        df.to_pickle(filename)
    
    return df