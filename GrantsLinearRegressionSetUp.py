#Install Required Libraries
#For google colab only
#!pip install torch torchvision torchaudio
#!pip install seaborn tqdm


#----------------------------------------------------------
#------------Importing the libraries-----------------------
#----------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statistics


#-----------------------------------------------------------
#---------------Setup CUDA/CPU Device-----------------------
#-----------------------------------------------------------

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
#-----------------------------------------------------------



#-----------------------------------------------------------
#------------------Load Data and Preprocess-----------------
#-----------------------------------------------------------

# filelocation = sys.argv[0].replace("GrantsLinearRegressionSetUp.py", "")
# file_name = filelocation + 'Data\\grants.csv'
file_name = 'https://raw.githubusercontent.com/NickStitely/Capstone/refs/heads/LDBranch/Data/grants.csv'
df = pd.read_csv(file_name)
df.head()

#----------------------------------------------------------



#----------------------------------------------------------------------------------------------
#-------------------------Data Cleaning & Encoding---------------------------------------------
#----------------------------------------------------------------------------------------------
# Extract columns
X = df['category_of_funding_activity'].values
y = df['estimated_total_program_funding'].values



# Cap values at 1 billion (replace values > 1 billion with 1 billion)
y_capped = np.where(y > 1_000_000_000, 1_000_000_000, y)

# Handle NaNs by replacing with median (median calculated on y_capped ignoring NaNs)
median_val = np.nanmedian(y_capped)
y = np.where(np.isnan(y_capped), median_val, y_capped)

# Bin y into quantiles for stratification
y_binned = pd.qcut(y, q=5, duplicates='drop').codes  # converts bins to numeric codes

# Remove classes with less than 1 in X and y_binned
counts = pd.Series(y_binned).value_counts()
valid_classes = counts[counts > 1].index
mask = pd.Series(y_binned).isin(valid_classes)
X = X[mask.values]
y = y[mask.values]
y_binned = y_binned[mask.values]


# Encode X (categorical string) to numeric labels
le = LabelEncoder()
X_encoded = le.fit_transform(X)

#split with stratify on binned y
x_train, x_test, y_train, y_test = train_test_split(
    X_encoded, y, stratify=y_binned, test_size=0.2, random_state=42
)

print(f'train data shape: {x_train.shape}')
print(f'test data shape: {x_test.shape}')
#-----------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------
#---------------------------------PyTorch Dataset and DataLoader----------------------------------
#-------------------------------------------------------------------------------------------------
# Create Tensor datasets - convert to torch tensors
train_data = TensorDataset(
    torch.from_numpy(x_train).long(),  # categorical feature as long tensor
    torch.from_numpy(y_train).float()  # target as float tensor
)
valid_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))


# dataloaders
batch_size = 50

#shuffles data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

#-------------------------------------------------------------------------------------------------------------

#sample inputs and outputs
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample output: \n', sample_y)

# Convert input back to categories
categories = le.inverse_transform(sample_x.cpu().numpy() if isinstance(sample_x, torch.Tensor) else sample_x)

# Convert output to numpy
funding_amounts = sample_y.detach().cpu().numpy() if isinstance(sample_y, torch.Tensor) else sample_y

for cat, fund in zip(categories, funding_amounts):
    print(f"Category: {cat} -> Estimated Funding: ${fund:,.2f}")


from sklearn.linear_model import LinearRegression

#----------------------------------------------------------------------------------------------------------------
#---------------------------------------PRINT HELPERS------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# Define an object to print the Predictions
class PredictionHolder:
  def __init__(self, num, category, predicted):
    self.num = num
    self.category = category
    self.predicted = predicted

class DistinctPredictions:
  def __init__(self, categoryLookups, x, y, predType):
    self.categoryLookups = categoryLookups
    self.x = x
    self.y = y
    self.predType = predType

  def printPredictions(self):
    objects = [0]*26

    for i in range(len(self.x)):
      curX = self.x[i]
      curY = self.y[i]

      newObj = PredictionHolder(curX, self.categoryLookups[i], curY)

      if(objects[curX] == 0):
        objects[curX] = newObj

    print(f'Printing the {self.predType} Predictions')
    for i in range(len(objects)):
      if(objects[i] != 0):
        print(f'X Value: {objects[i].num}, \t\t Category: {objects[i].category}, \t\tPredicted: {objects[i].predicted}')




#------------------------------------------------------------------------------------------------------------------
#-----------------------------------------Linear Regression--------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Setup a Linear Regression from SKLearn
model = LinearRegression()

# Fit the training data. Had to reshape as this expects 2-D
model.fit(train_data.tensors[0].reshape(-1, 1), train_data.tensors[1])

# Print some details
print(f'Intercept: {model.intercept_}')
print(f'Score: {model.score(x_test.reshape(-1, 1), y_test)}')

# Predictions
y_pred = model.predict(x_test.reshape(-1, 1))

# Reverse transform the categories back to text
cats = le.inverse_transform(x_test)

# Test the transform
print(f'Category Test: {cats[1]}')



# Plot Things

def hundreds_of_millions(x, pos):
    return f'{x / 1:.1f}'

max_y = max(y_train)

# cut off the billions and above - not a big deal for the graph but the few awards of 178 billion are insane to try and plot. Even up to 1 billion is not great
mask_train_y = (y_train <= 1000000000)
mask_test_y = (y_test <= 1000000000)

plt.figure(figsize=(16, 10))
plt.scatter(le.inverse_transform(x_train[mask_train_y]), y_train[mask_train_y], color='green', label='train', alpha=0.7)
plt.title("Scatter Plot: Training")
plt.xlabel("Categories")
plt.ylabel("(Award Amounts (M))")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(hundreds_of_millions))
plt.xticks(rotation=90)
plt.show()



# Show the predictions
x_axis_linear = le.inverse_transform(x_test)
y_axis_linear = y_pred;
df_linear = pd.DataFrame({ 'x' : x_axis_linear, 'y' : y_axis_linear })
df_linear_sorted = df_linear.sort_values('y')

plt.figure(figsize=(16, 6))
plt.scatter(df_linear_sorted['x'], df_linear_sorted['y'], color='red', label='pred', alpha=0.7)
plt.title("Scatter Plot: LINEAR Regression Predictions")
plt.xlabel("Categories")
plt.ylabel("(Award Amounts (M))")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(hundreds_of_millions))
plt.xticks(rotation=90)
plt.show()


# End Model Creation Here



#------------------------------------------------------------------------------------------------------------------
#----------------------------------------Model Validation----------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Error Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Validation Metrics:")
print(f"MAE  (Mean Absolute Error):      {mae:,.2f}")
print(f"MSE  (Mean Squared Error):       {mse:,.2f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:,.2f}")
print(f"R²   (R-squared Score):           {r2:.4f}")
#------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------
#----------------------------------------Add Visualization for Validation------------------------------------------
#------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Estimated Funding")
plt.ylabel("Predicted Funding")
plt.title("Actual vs Predicted Funding")
plt.grid(True)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.show()
#------------------------------------------------------------------------------------------------------------------














#-----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------Attempt a Lasso Regression (No reason we still only use one feature------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import numpy as np

scaler = StandardScaler().fit(train_data.tensors[0].reshape(-1, 1))

# Scale Train and Test X
x_train_scaled = scaler.transform(train_data.tensors[0].reshape(-1, 1))
x_test_scaled  = scaler.transform(x_test.reshape(-1, 1))


cv = RepeatedKFold(n_splits = 3, n_repeats = 15, random_state = 42)
alphas = np.logspace(-5, 0, 50)


lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=20000).fit(x_train_scaled, train_data.tensors[1])
print("Best alpha:", lasso_cv.alpha_)

best_pred_lasso = lasso_cv.predict(x_test_scaled)
print("Tuned Test MSE:", mean_squared_error(y_test, best_pred_lasso))
print("Non-zero coefficients:", np.sum(lasso_cv.coef_ != 0))


# Plot the Lasso Predictions
x_axis_lasso = le.inverse_transform(x_test)
y_axis_lasso = best_pred_lasso;

df_lasso = pd.DataFrame({ 'x' : x_axis_lasso, 'y' : y_axis_lasso })
df_lasso_sorted = df_lasso.sort_values('y')

plt.figure(figsize=(16, 6))
plt.scatter(df_lasso_sorted['x'], df_lasso_sorted['y'], color='green', label='pred', alpha=0.7)
plt.title("Scatter Plot: LASSO Regression Predictions")
plt.xlabel("Categories")
plt.ylabel("(Award Amounts (M))")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(hundreds_of_millions))
plt.xticks(rotation=90)
plt.show()




DistinctPredictions(cats, x_test, y_pred, 'Linear Regression').printPredictions()
print('======================================================================================================================================')
DistinctPredictions(x_axis_lasso, x_test, best_pred_lasso, 'Lasso Regression Repeated').printPredictions()
