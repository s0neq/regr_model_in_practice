1) post your program and output
2) post a frequency table for your (recoded) categorical explanatory variable or report the mean for your centered explanatory variable. 
3) Write a few sentences describing the results of your linear regression analysis.


my explanatory variable is sex worker's bmi (body mass index) and response variable is Log of Hourly Wage


my code:

```
# -*- coding: utf-8 -*-

import numpy as numpyp
import pandas as pandas
import statsmodels.api
import statsmodels.formula.api as smf
import seaborn 
import matplotlib.pyplot as plt

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/adult_services.csv')
data.dropna(inplace=True)
data = data.drop(columns=["Unnamed: 0"])

# convert variables to numeric format using convert_objects function
data['bmi'] = pandas.to_numeric(data['bmi'], errors='coerce')
data['lnw'] = pandas.to_numeric(data['lnw'], errors='coerce')

# If you have a quantitative explanatory variable, center it so that the mean = 0 
# (or really close to 0) by subtracting the mean, 
# and then calculate the mean to check your centering. 

# bmi needs centering
temp = data[["bmi"]].apply(lambda x: x-x.mean())
print("this is the mean of the centered bmi column ", temp.mean())
data["bmi_centered"] = temp["bmi"]


############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################
scat1 = seaborn.regplot(x="bmi_centered", y="lnw", scatter=True, data=data)
plt.xlabel("worker's bmi") # explanatory variable
plt.ylabel("Log of Hourly Wage") # response variable
plt.title ("Scatterplot for the Association Between worker's bmi and Log of Hourly Wage")
print(scat1)

```

this prints this output

<img width="490" alt="image" src="https://user-images.githubusercontent.com/22098104/155390846-35856d9f-6421-43e2-97a6-c7c7cfe84d00.png">

report the mean for your centered explanatory variable: as you can see at the top bmi mean is 0

another piece of code:

```
print("OLS regression model for the association between worker's bmi and Log of Hourly Wage")
reg1 = smf.ols('lnw ~ bmi_centered', data=data).fit()
print(reg1.summary())
```

yields this table

<img width="779" alt="image" src="https://user-images.githubusercontent.com/22098104/155391129-9abd30e8-b78d-4eca-9642-cdee149dbfce.png">



From our analysis, we can conclude that sex worker's BMI is negatively associated with their hourly wage. 

p value is very small (Prob (F-statistic): 1.56e-16), so we can reject H0. 

coef for bmi is -0.0189.



