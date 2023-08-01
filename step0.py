import pandas as pd



def loadData():
    '''
    function loads data into train, test
    dropna : ignore data row with missing data
    '''
    train = pd.read_csv("Train.csv").dropna(0)
    test = pd.read_csv("Test.csv").fillna('')

    return train, test

    
