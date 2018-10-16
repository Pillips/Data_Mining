# -*- coding: utf-8 -*-
# frequence caculator for every value in some variable name
def freq_cal(var_str,train_data,target_name):
    import pandas as pd
    import numpy as np
    var_data=pd.DataFrame(train_data[var_str].drop_duplicates())
    var_data['ave_'+target_name]=0.1
    var_data=var_data.reset_index(drop=True)
    for i in range(0,len(var_data)):
        var_data['ave_'+target_name][i]=np.mean(train_data[target_name].loc[train_data[var_str]==var_data[var_str][i]])
    freq_data=pd.DataFrame(train_data[var_str].value_counts())
    freq_data=freq_data.rename(columns={var_str:'frequence'})
    freq_data[var_str]=freq_data.index
    freq_data=freq_data.reset_index(drop=True)
    com_data=pd.merge(var_data,freq_data,on=var_str,how='left')
    com_data.sort_values(var_str,inplace=True)
    return com_data   

