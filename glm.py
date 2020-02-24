import statsmodels.api as sm
from scipy.linalg import toeplitz
import numpy as np
import scipy.stats as stats

# fit model 
# X_train and X_test already have bias

mod = sm.OLS(y_train, X_train).fit()
resid = mod.resid
res = sm.OLS(resid[1:], resid[:-1]).fit()
rho = res.params
# get predictions
pred = mod.predict(X_test)

# calculate sigma
order = toeplitz(np.arange(X_train.shape[0]))
sigma = rho**order

# calculate likelihood
def calculate_llh(pred, data, sigmas):
	length = len(data)
	nll_total = 0.
	for i in range(length):
		nll_total += llh(pred[i], data[i], sigmas)
	return nll_total

def llh(pred, data, sigmas):
	length = len(data)
	ll = 0.
	for index in range(length):
		y_hat = pred[index]
		residual = float(data[index]) - y_hat
		ll += stats.norm.logpdf(residual, 0, sigmas[index])
	return ll