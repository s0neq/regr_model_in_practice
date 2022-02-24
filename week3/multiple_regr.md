
1st part of the code

```
# -*- coding: utf-8 -*-

import numpy as numpyp
import pandas as pd
import statsmodels.api as sm
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


# age needs centering
temp = data[["age"]].apply(lambda x: x-x.mean())
print("this is the mean of the centered age column ", temp.mean())
data["age_centered"] = temp["age"]


# age_cl needs centering
temp = data[["age_cl"]].apply(lambda x: x-x.mean())
print("this is the mean of the centered age_cl column ", temp.mean())
data["age_cl_centered"] = temp["age_cl"]

```

models' outputs

```
# regression analysis
reg1 = smf.ols('lnw ~ bmi_centered', data=data).fit()
print(reg1.summary())
```
<img width="770" alt="image" src="https://user-images.githubusercontent.com/22098104/155539842-1d413ecc-0293-4c4c-b21a-5e2d4a3710b5.png">


```
# regression analysis with adding variables: age
reg1 = smf.ols('lnw ~ bmi_centered + age_centered', data=data).fit()
print(reg1.summary())
```
<img width="773" alt="image" src="https://user-images.githubusercontent.com/22098104/155539909-ec7c93d6-5e44-400b-bff9-3bed64d50f74.png">

```
# regression analysis with adding variables: race
reg1 = smf.ols('lnw ~ bmi_centered + age_centered + white', data=data).fit()
print(reg1.summary())
```
<img width="770" alt="image" src="https://user-images.githubusercontent.com/22098104/155540058-2e1e3f2f-c4d7-43fe-9e48-ca9958955641.png">


```
# regression analysis with adding variables: age_cl
reg1 = smf.ols('lnw ~ bmi_centered + age_centered + white + age_cl_centered', data=data).fit()
print(reg1.summary())
```
<img width="774" alt="image" src="https://user-images.githubusercontent.com/22098104/155540137-9b94d604-932f-4aa7-afec-125c9c6c684f.png">


some thoughts: 

with each new variable R-squared shows that the percentage explained by the model increases.

After adjusting for potential confounding factors (age of the worker, race of the worker: white/non-white, age of the client), it appears as bmi  (Beta=-0.0184, p=.0001), age  (Beta=-0.0066, p=.0001) and  age of the client (Beta=-0.0035, p=.0001)  are statistically significantly associated with the Log of Hourly Wage. in every case it's negative correlation.


evaluating model fit:

```
#Q-Q plot for normality
fig4=sm.qqplot(reg1.resid, line='r')
```
<img width="426" alt="image" src="https://user-images.githubusercontent.com/22098104/155542932-24f7151f-86ed-4be1-9c91-65af8e59df0c.png">

The qqplot for our regression model shows that the residuals generally follow a straight line (with very slight skew in the middle), but deviate at the lower and higher quantiles. This indicates that our residuals did not follow perfect normal distribution.

```
# simple plot of residuals
stdres=pandas.DataFrame(reg1.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
```
we can see that the residuals are centered around zero and mostly lie within 2 standard deviaions from the mean.

<img width="403" alt="image" src="https://user-images.githubusercontent.com/22098104/155543169-209e9e5f-5225-4e69-8e6d-bcbe427896bc.png">


```
# leverage plot
fig3=sm.graphics.influence_plot(reg1, size=8)
print(fig3)
```

<img width="448" alt="image" src="https://user-images.githubusercontent.com/22098104/155543497-bbc15061-a3b9-49ae-90ed-3f01a94cdd67.png">

leverage plot shows that we have some outliers that we need to deal with.
