import numpy as np

def booleanizeNumericalColumnUsingQuartiles(df, col_name):
    """
    Transform numerical column with given name into three boolean columns.
    The three columns indicate whether the value falls into one of three ranges defined by the three quartiles of the data.
    The columns correspond to x<q1, q1<x<q2 and q3<x.     
    """
    
    #compute quartiles for values
    x = df[col_name].values
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    
    #generate new columns corresponding to quartile intervals
    b1 = x <= q1
    df[col_name+"_small"] = b1
    
    b2 = (q1 < x) & (x < q3)
    df[col_name+"_med"] = b2
    
    b3 = x >= q3
    df[col_name+"_large"] = b3
    
    #delete old column
    df.drop(columns=col_name, inplace=True)
    
def booleanizeBinaryColumn(df, col_name):
    """
    Transform binary column with values 0, 1 into boolean column.
    """

    b = df[col_name].values == 1
    df[col_name] = b