import numpy as np
import pandas as pd
from datetime import datetime
from wrangle import wrangle_da

def explore_df():
    '''
    Uses wrangle_da() to pull in the original data cleaned up
    
    then cuts down to the top 7 crimes, drops "Other" bs and further investigations and let\'s us explore some more
    '''
    
    df = wrangle_da()
    
    df = df[(df.crime_type == 'Assault') | (df.crime_type == 'Narcotics') | (df.crime_type == 'Burglary')\
       | (df.crime_type == 'DUI') | (df.crime_type == 'Assault and Battery') | (df.crime_type == 'Theft') | (df.crime_type == 'Robbery')]


    df = df.drop(index=df[df.da_action_taken == 'Other Action'].index)



    mtr_folks = df[df.da_action_taken == 'MTR/Referred to Other Agency']

    df = df.drop(index=mtr_folks.index)

    df = df.drop(index=df[df.da_action_taken == 'Further Investigation Requested'].index)


    df = df.reset_index(drop=True)

    df.da_action_taken = df.da_action_taken.cat.set_categories(['Charges Filed', 'Discharged'], ordered=True)

    df = df.reset_index(drop=True)

    df['low_high_court_num'] = pd.cut(df.court_number, bins= (0, 4_000_000, 40_000_000), labels= ['low', 'high'])

    return df


def split_data(df, strat_by, rand_st=123):
    '''
    Takes in: a pd.DataFrame()
          and a column to stratify by  ;dtype(str)
          and a random state           ;if no random state is specifed defaults to [123]
          
      return: train, validate, test    ;subset dataframes
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=.2, 
                               random_state=rand_st, stratify=df[strat_by])
    train, validate = train_test_split(train, test_size=.25, 
                 random_state=rand_st, stratify=train[strat_by])
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')


    return train, validate, test

