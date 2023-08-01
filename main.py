import step0 as s0
import Step1CleanData as s1
import step2Vectorization as s2
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    
    train, test = s0.loadData()
    #print(train.describe())
    #print(train)


    train['safe_text'] = train['safe_text'].apply(s1.pipelineClean)
    test['safe_text'] = test['safe_text'].apply(s1.pipelineClean)

    
    vectors = s2.vecTfid(train['safe_text'])
    tfid_matrix = vectors.todense()
    np.set_printoptions(threshold=np.inf)
    print(tfid_matrix)







    

    


if __name__ == '__main__':
    main()