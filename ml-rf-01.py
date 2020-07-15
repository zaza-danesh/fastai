%load_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

PATH = 'data/'

!ls {PATH}
!head {PATH}/file.csv

df_raw = pd.read_csv(f'{PATH}file.csv', low_memory=False) #read the data
df_raw = pd.read_csv(f'{PATH}file.csv', low_memory=False, parse_dates=["date"]) #read the data and parse the dates
df_raw.column #check the column of data
df_raw.shape #check the size of data
display(df_raw) #display the data

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

display_all(df_raw.tail().T)

display_all(df_raw.describe(include='all').T) #Statistical description

df_raw.SalePrice #The dependent variable (target variable), e.g. SalePrice
df_raw.SalePrice = np.log(df_raw.SalePrice) #The dependent variable (target variable), e.g. SalePrice

df_raw.drop('Redundant column', axis=1) #Drop a column from the dataset

#Initial processing

##To parse the dates
fld = df_raw.date #the name date is assigned through parse_dates=["date"] argument from pandas
type(fld) # pandas.core.series.Series
fld.dt.is_leap_year #panda attribute (builtin functions?) to check if a given date is in leap year

add_datepart(df_raw, 'date') #Parse the date column to get more information from the dates
df_raw.column #Check new columns added instead of date column

##To train the categorical columns
train_cats(df_raw) #Categories will be displayed still as categories but treated numerically internally 

#UsageBand is an example from spcific dataset, check BlueBulldozers
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True) #Give specific order to the categories, otherwise will be chosen by default

##To replace the text category with numbers, not necessary however
df_raw.UsageBand = df_raw.UsageBand.cat.codes


df_raw.isnull().sum().sort_index() #check how many null input in different columns, sorted
display_all(df_raw.isnull().sum().sort_index()/len(df_raw)) #give the result in ratio of null inputs



#Save the current processed dataset
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/-raw') #titanic-raw, bulldozers-raw, whatever-raw

#Preprocessing

df_raw = pd.read_feather('tmp/whatever-raw')

##Replace categories with values, handle missing values with median and split the dependent variable into a separate variable
df, y, nas = proc_df(df_raw, 'SalePrice') #SalePrice is the target variable

#Prediction step
m = RandomForestRegressor(n_jobs=-1)
m.fit(df,y)
m.score(df,y) #will get an overfitted predicition, need to split to train and validation set



def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn) #it's a deterministic step, so no mixed up for independent and dependent variable

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
	res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
	m.score(X_train, y_train), m.score(X_valid, y_valid)]
	if hasattr(m,'oob_score'): res.append(m.oob_score_)
	print(res)

m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)

# A good ptactice for large data sets is to practice or train your work on a subset of data,
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)

#Single Tree with three level split, max_depth=3
m = RandomForestRegressor(n_estimator=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)

#Visualize the tree
draw_tree(m.n_estimator_[0], df_trn, precision=3)

#Bigger tree, deeper tree
m = RandomForestRegressor(n_estimator=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)

#Bootstrap Aggregating, Bagging

m = RandomForestRegressor(n_jobs=-1) #default of n_estimator, number of trees is 100
m.fit(X_train. y_train)
print_score(m)

preds = np.stack([t.predict(X_valid) for t in m.estimators_]
preds[:,0], np.mean(preds[:,0], y_valid[0]))

#Visualize how increase in number of trees improve r2 score
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])


#Try different number of trees to see how it improves the scores, in the course, it's assumer n_estimator default is 10
m = RandomForestRegressor(n_estimator=20, n_jobs=-1) #default of n_estimator, number of trees is 100
m.fit(X_train. y_train)
print_score(m)

m = RandomForestRegressor(n_estimator=40, n_jobs=-1) #default of n_estimator, number of trees is 100
m.fit(X_train. y_train)
print_score(m)

m = RandomForestRegressor(n_estimator=80, n_jobs=-1) #default of n_estimator, number of trees is 100
m.fit(X_train. y_train)
print_score(m)

#Out of bag score, OOB score
##validation set uses separate period of time (if data is timed), however oob samples from the training set
## and therefore the same period of time, and if it shows better score, this could be one reason
m = RandomForestRegressor(n_estimator=40, n_jobs=-1, oob_score=True) 
m.fit(X_train. y_train)
print_score(m)

#Subsampling
## So that each tree takes a different subset of all data, given large number of trees, eventually it will see all the data

set_rf_samples(20000) #Function written by fastai
m = RandomForestRegressor(n_estimator=80, n_jobs=-1, oob_score=True) # Having larger number of trees increase the score, since RF sees more of the data
m.fit(X_train. y_train)
print_score(m)

#To reset the sample size
reset_rf_samples()
