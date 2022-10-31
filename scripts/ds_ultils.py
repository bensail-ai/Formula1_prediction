from locale import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display


def df_explore(df):
    """
    getting some basic information about a dataframe
    Showing the shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    Data types of columns
    Check for duplicated columns
    Args:
        df (dataframe): dataframe containing the data for analysis
        
    """
    
    print()
    print(f"Dataframe Rows: {df.shape[0]} \t Dataframe Columns: {df.shape[1]}")
    print("\n")
    print('Summary of the dataframe : \n',' \n ')
    print(df.info())
    print("-------------\n")
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print("------------\n")
    display(df.head())
    # print(f'Are there duplicated columns: {df.T.duplicated()}')
    # print("------------\n")
    # print(f'Summary of the data type:','\n', df.dtypes)
    print("------------------------------\n")

def nan_checker(df):
    """
    checks for nan values in the dataset

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        nan_summary (String): returns a summary of nan values 
    """
    test_df = df.copy()
    # find nans
    nan_series = test_df.isna().sum()
    # any nans
    any_nans = nan_series.sum() > 0
    #how many
    n_nans = nan_series.sum()
    #which variables they come from - list of columns
    nan_cols = list(nan_series[nan_series>0].index)
    # any variables more than 50% data missing?
    big_nan_cols = list(nan_series[nan_series/len(test_df) > 0.5].index)
    #summary
    summary = f'This dataframe has {n_nans} NaN values'

    if len(nan_cols)>0:
        summary += f"\n The NaN values come from: {nan_cols}. \n"
    if len(big_nan_cols)>0:
        summary += f'\t These {big_nan_cols} columns have > 50% NaNs'

    return print(summary)


def variables_explore(df):
    """
    For numerical variables in a dataframe it prints the describe method
    for Categorical variables in a dataframe prints the value counts and percentage

    Args:
        df (dataframe): dataframe containing the data for analysis
    """

    num_col_list = list(df.select_dtypes("number").columns)
    cat_col_list = list(df.select_dtypes("object").columns)+list(df.select_dtypes("category").columns)
    print()
    print(f"Dataframe numeric columns: \n",',\n'.join(num_col_list))
    print()
    print("Categorical columns: \n", ',\n '.join(cat_col_list))
    print()
    for col in num_col_list:
        print()
        print(f"{col.upper()} numeric column distribution: \n", df[col].describe())
        print("----------------\n")
    for col in cat_col_list:
        print()
        print(f"{col.upper()} cateogrical column total counts: \n", df[col].value_counts())
        print()
        print(f"{col.upper()} cateogrical column percentage counts: \n", df[col].value_counts(normalize=True))
        print("----------------\n")


def plot_numerical_variables_hist(df,num_columns=5,columns=None,**kwargs):
    if columns == None:
        num_col_list = list(df.select_dtypes("number").columns)
    else:
        num_col_list = columns
    if len(num_col_list) > 50:
        figsize=(14,50)
    elif len(num_col_list) > 20:
        figsize=(14,20)
    else:
        figsize=(14,12)
    fig, axes = plt.subplots(math.ceil(len(df[num_col_list].columns)/num_columns), num_columns, figsize=figsize,**kwargs)
    for col, axs in zip(df[num_col_list].columns, axes.flatten()): # axes are a numpy array need to flatten to be able to iterate over it easily
        sns.histplot(df[col],ax=axs,bins=100)
        axs.set_title(col)
    plt.tight_layout()
    plt.show()


def normaility_check(df,variables):
    """
    Runs Shipro-Wilk normality check
    Creates a series of plots including
    Histograms, box plots and q-q plots to check for normality

    ------
    Parameters

    df (dataframe): dataframe containing the data for analysis

    variables - list of columns to check

    ------
    Plots Histogram
    Plots Boxplot
    Plots Q-Q plot   
    """

    rows=1
    cols=3
    for variable in variables:
        print("number of samples :", len(df[variable]))
        print("Shapiro-Wilk test:", stats.shapiro(df[variable]))
        print("----------")
        if float(stats.shapiro(df[variable])[1]) < 0.05:
            print("can reject null hypothesis and data may not be normally distributed")
            print()
            if len(df[variable]) > 100:
                print('Sample size is greater than 100 \n', 
                'therefore as per Central Limit Theorem reasonable to assume underlying population will be normal')
            else:
                print('sample size not big enough therefore if not normal \n',
                'do not continue with t test')
        else:
            print("can not reject null hypothesis of normality")
            print()

        print()
        print('Plots')
        x=df[variable].astype('float')
        plt.subplots(rows,cols, figsize=(10,5))
        plt.title(variable)            
        plt.subplot(rows, cols, 1)
        plt.hist(x, bins=50)
        plt.title(f'{variable} Histogram')        
        plt.subplot(rows, cols, 2)
        plt.boxplot(x)
        plt.title(f'{variable} Boxplot')
        plt.subplot(rows, cols, 3)        
        stats.probplot(x, dist="norm", plot = plt.subplot(rows, cols, 3))        
        plt.title(f'{variable} Q-Q plot')            
        plt.show()
        


def correlation_test(df,variables,target,figsize=(15,15)):
    """
    Runs a pearsonsr statistical test of correlation with a target variable 
    and plots scatter plots of variables

    ------
    Parameters

    df (dataframe): dataframe containing the data for analysis

    variables (list) : list of columns names to check for correlation

    target (string) : target name to check feature variables against

    ------
    Returns
    Prints pearsonsr correlation test
    Plots the seaborn regression plot between variables and target

    """

    print("Pearsonsr Statistical Correlation Test:")
    print("-----------------")

    for col in variables:
        print(f'{col} pearsonsr correlation test against {target} : \n', stats.pearsonr(df[target],df[col]))
        print()
    print("------------------")
    rows=math.ceil(len(variables)/3)
    cols=3
    plt.subplots(rows,cols,figsize=figsize)
    count = 1
    for col in variables:
        x=df[col].astype("float")
        sns.regplot(ax=plt.subplot(rows, cols, count),data = df, y=target, x=x)
        plt.title(col)
        
        count +=1
    plt.tight_layout()    
    plt.show()

def independence_test(df,variables,annot=True):
    """
    Checks for independence between numerical variables in a dataframe prior 
    to linear model regression    
    Prints the VIF independence test results and plots the correlation table

    ------
    Parameters

    df (dataframe): dataframe containing the data for analysis

    variables (list) : list of columns names to check for independence

    ------
    Returns

    Plots SNS heatmap plot between variables 

    Prints Variation Inflation Factor test between variables



    """
    corr =df[variables].corr()
    plt.figure(figsize=(10,10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    
    plt.title('SNS heatmap')
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=1, vmin=-1,annot=annot)
    plt.title("Heatmap of correlation coefficients")
    plt.show()
    
    print("-------------------------------")
    print("VIF test results")
    x=df[variables]
    x_const=sm.add_constant(x)
    print("Variation Inflation Factor : \n", pd.Series([variance_inflation_factor(x_const.values, i) 
               for i in range(x_const.shape[1])], 
              index=x_const.columns)[1:])
               
    print("-----------------------------------")

def linear_regression(df,variables,target):
    """
    Builds and evaluates a linear regression model

    ------
    Parameters

    df (dataframe): dataframe containing the data for analysis

    variables (list) : list of columns names to include in model

    target (string) : target name in data to predict with model

    ------
    Returns
    data frame of R2 squared, number of variables and model descritpion

    prints: 
    Linear regression model summary ouput
    Displays a dataframe of key results
    Breusch-Pagan test result
    Shaprio-Wilk test result

    Displays
    Plot of residuals against fitted values
    q-q plot of residuals 
    
    """

    X= df[variables]
    y= df[target]
    X_const=sm.add_constant(X)
    # 1. Instantiate model
    lin_model = sm.OLS(y,X_const)
    # 2. Fit model
    lin_model_res = lin_model.fit()
    print("----------------------")
    print(" Linear regression model summary \n:", lin_model_res.summary())
    df_results = lin_model_res.params.reset_index()
    df_results= df_results.merge(lin_model_res.bse.reset_index(), left_on='index', right_on='index')
    df_results = df_results.merge(lin_model_res.pvalues.reset_index(), left_on='index', right_on='index')
    df_results.columns=['variables','params','error','pvalue']
    print("Summary of results :")
    display(df_results)
    print("--------------------")
    df_results_summary=pd.DataFrame({'Model': 1, 'R squared':lin_model_res.rsquared,'Number of variables': len(variables) ,'Desc':[variables]}, index=[0])

  
    plt.figure()
    sns.regplot(lin_model_res.fittedvalues, lin_model_res.resid, lowess = True, line_kws = {'color': 'red'})
    plt.title('Residuals vs Predicted Values', fontsize=16)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

   
    bp_test = pd.DataFrame(sms.het_breuschpagan(lin_model_res.resid, lin_model_res.model.exog), 
                       columns=['value'],
                       index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])
    display(bp_test)
    if bp_test['value']['p-value'] < 0.05:
        print('Breusch-Pagan Test suggest we can reject null hypothesis and suggest the residuals are heteroscedasticity')
    else:
        print('Breusch-Pagan Test suggest we can not reject null hypothesis and that the residuals are homoscedastic')
    
    plt.figure(figsize=(3,3))
    sm.ProbPlot(lin_model_res.resid).qqplot(line='s')
    plt.title('Q-Q plot of residuals')
    plt.show()
    
    print("Shaprio-Wilk test of Residuals is: \n ", stats.shapiro(lin_model_res.resid))

    if float(stats.shapiro(lin_model_res.resid)[1]) < 0.05:
        print("can reject null hypothesis and residuals may not be normally distributed")
    else:
        print("can not reject null hypothesis and the residuals are likely normally distributed")
    print("---------------------------")
    
    
    
    return df_results_summary


def homoscedasticity_test(residuals,predictions,**kwargs):

    plt.subplots(1, 2, figsize=(10,5),**kwargs)

    
    plt.subplot(1,2,1)
    sm.ProbPlot(residuals).qqplot(line='s')
    plt.title('Q-Q plot of residuals')

    plt.subplot(1,2,2)
    plt.plot(predictions,residuals,marker='o',linestyle = 'None')
    plt.title('Plot of residuals against predictions')
    plt.ylabel('Residuals')
    plt.xlabel('Predictions')
    plt.show()
    
    print("Shaprio-Wilk test of Residuals is: \n ", stats.shapiro(residuals))

    if float(stats.shapiro(residuals)[1]) < 0.05:
        print("can reject null hypothesis and residuals may not be normally distributed")
    else:
        print("can not reject null hypothesis and the residuals are likely normally distributed")
    print("---------------------------")
    


def logistic_regression(df,variables,target):
    """
    Runs a logistic regression model and evaluates the results 

    ------
    Parameters

    df (dataframe): dataframe containing the data for analysis

    variables (list) : list of columns names to include in model

    target (string) : target name in data to predict with model


    ------
    Returns
    data frame of accuracy, threshold and number of model variables

    prints: 
    Logistic regression model summary ouput
    Displays a dataframe of key results including odds ratio, p value and coeficient
    

    Displays:
    The coeficients per variable
    Odds ratio per variable
    Plt of accuracy vs threshold
  
    """

    X= df[variables]
    y= df[target]
    X_const=sm.add_constant(X)
    # 1. Instantiate model
    log_model = sm.Logit(y,X_const)
    # 2. Fit model
    log_model_res = log_model.fit()
    print("----------------------")
    print(" Logisitic regression model summary \n:", log_model_res.summary())
    coefs_df = pd.DataFrame(data={"coefs": log_model_res.params, "odds_ratios": np.exp(log_model_res.params), "p_value": log_model_res.pvalues})
    print("Summary of results :")
    display(coefs_df)
    print("--------------------")
    print("How do the coefficients vary across parameters?")
    plt.figure(figsize=(10, 6))
    coefs_df["coefs"].sort_values().plot(
    kind="barh", 
    color=np.where(coefs_df["coefs"].sort_values() <= 0, "red", "blue"))
    
    plt.vlines(0, ymin=-1, ymax=10, color="black", linestyle="--")
    plt.title("Coefficients of the model")
    plt.show()    
    print("--------------------")
    print("How do the odds ratios vary across parameters?")
    plt.figure(figsize=(10, 6))
    coefs_df["odds_ratios"].sort_values().plot(
    kind="barh", 
    color=np.where(coefs_df["odds_ratios"].sort_values() <= 1, "red", "blue"))

    plt.vlines(1, ymin=-1, ymax=10, color="black", linestyle="--")
    plt.title("Odds ratios per variable")
    plt.show()

    thresholds = np.arange(0,1,0.01)
    predictions = log_model_res.predict(X_const)

    accuracies =[(np.mean((predictions>p) == df[target])) for p in thresholds ]
    print("--------------------")
    print("How does the acuracy vary with threshold?")
    plt.figure(figsize=(6 ,6))
    plt.plot(thresholds,accuracies)
    plt.vlines(thresholds[np.argmax(accuracies)],ymin=0, ymax=1, color='r')
    plt.annotate('Best threshold', xy=(thresholds[np.argmax(accuracies)],np.max(accuracies)),  xycoords='data' )
    plt.title("Accuracy of the model with variation in threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.show()
       
    
    df_log_results = pd.DataFrame(data={'Model number': 1,'accuracy': np.max(accuracies), 'threshold': thresholds[np.argmax(accuracies)], 'Number of variables': len(variables)}, index=[0])
    
    return df_log_results






def plot_dist_by_dim(data, column, dim):
    """
    Plots the given column against the registration station in the data.
    The function assumes data is a dataframe, column is string (existing column in data),
    and data has a registered column too.
    """
    total_count = data.groupby([column, dim])[column].count()
    pct_contact_type = total_count/data.groupby(column)[column].count()
    pct_contact_type = pct_contact_type.unstack()
    print(pct_contact_type.sort_values([1]))
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = (3.2, 2)
    # set the font name for a font family
#     plt.rcParams.update({'font.sans-serif':'Helvetica'})
    sns.set(style="whitegrid")
    pct_contact_type.sort_values([1]).plot(kind="bar", stacked=True)
    sns.despine(left=True)
    plt.title(f"{column} group distribution", size=10)
    plt.xlabel('')
    #plt.xticks(size=8, rotation=90)
    #plt.yticks(size=8, color='#4F4E4E')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()


def plot_categorical_cols_by_dim(df,dim,columns=5,**kwargs):
    figsize=(14,12)
    cat_list = list(df.select_dtypes("object").columns)+list(df.select_dtypes("category").columns)
    fig,axes=plt.subplots(math.ceil(len(df[cat_list].columns)/columns), columns, figsize=figsize)
    for col,axes in zip(df[cat_list].columns,axes.flatten()):
        total_count = df.groupby([col, dim])[col].count()
        pct_contact_type = total_count/df.groupby(col)[col].count()
        pct_contact_type = pct_contact_type.unstack()
        #print(pct_contact_type)
        #print(pct_contact_type.sort_values())    
        pct_contact_type.plot(kind="bar", stacked=True, ax=axes)
        sns.despine(left=True)
        axes.set_title(f"{col} group distribution", size=10)
    plt.tight_layout()
    plt.show()


def check_column_encoding(data,col1,col2):
    """"
    Checks the outputs of two series objects when one has been encoded into numerical values 
    and the other is strings
    ------
    Parameters

    data (dataframe) : dataframe containing the data for analysis

    col1 : column1 to check against column 2

    col2 : column2 to check against column 1
    
    """

    try:
        pd.testing.assert_series_equal(data[col1].value_counts(),
                                    data[col2].value_counts(),
                                                    check_names= False, check_index=False) # using pandas function to run assert series are equal. it check names and object type as well as numbers. to do this we need to ignore index and names to check just the numbers
        print('Series are equal')
        # in this case dtypes are different for columns, however for value counts dtypes are the same as they are both intergers 
    except:
        print('Series are not equal')

def convert_time_miliseconds(x):
    if x == 'nan':
        return None
    else:    
        if len(x.split(':')) > 2:
            secs = float(x.split(':')[0])*60*60*1000 + float(x.split(':')[1])*60*1000+ float(x.split(':')[2])*1000
        elif len(x.split(':')) > 1:
            secs = float(x.split(':')[0])*60*1000 + float(x.split(':')[1])*1000
        elif len(x.split(':')) == 1:
            secs= float(x.split(':')[0])*1000
        return round(secs,0)

