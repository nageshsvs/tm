#Author : Nagesh Somayajula
# Date : 02 Sep 2019
# Time series Analysis for all top most used Algorithms
# LIC : Personal use by EDP team

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class time_series:
 # define eash function for use autoregression

    def time_series_ar(ardata):
        try:
            model = AR(ardata)
            model_fit = model.fit()
            # make prediction
            ar_prediction = model_fit.predict(len(ardata), len(ardata))
            return (ar_prediction)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_ma(madata,p,q):
       # Moving Average (MA)
       # fit model
        try:
           model = ARMA(madata, order=(p, q))
           model_fit = model.fit(disp=False)
           # make prediction
           ma_prediction = model_fit.predict(len(madata), len(madata))
           return (ma_prediction)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_ARAM(aramdata,p,q):
        try:
           model = ARMA(aramdata, order=(p, q))
           model_fit = model.fit(disp=False)
           # make prediction
           ma_prediction = model_fit.predict(len(aramdata), len(aramdata))
           return (ma_prediction)
        except:
           return ("Please verify data : It should be one dimensional and list type.")
# for Arima
    def arima_lag_identification(arimadata):
        try:
            autocorrelation_plot(arimadata)
            pyplot.show()
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_ARIMA(arimadata,p,d,q):
        try:
           model = ARIMA(arimadata, order=(p, d, q))
           model_fit = model.fit(disp=0)
           print(model_fit.summary())
           # plot residual errors
           residuals = DataFrame(model_fit.resid)
           residuals.plot()
           pyplot.show()
           residuals.plot(kind='kde')
           pyplot.show()
           print(residuals.describe())
           arima_pred = model_fit.predict(len(arimadata), len(arimadata), typ='levels')
           print(arima_pred)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_ARIMA_forcast(arimadata,p,d,q):
        try:
            X = arimadata.values
            size = int(len(X) * 0.66)
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(p, d, q))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
                print('predicted=%f, expected=%f' % (yhat, obs))
            error = mean_squared_error(test, predictions)
            print('Test MSE: %.3f' % error)
            # plot
            pyplot.plot(test,label="Actual")
            pyplot.plot(predictions, color='red',label="Prediction")
            pyplot.show()
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_SARIMA(sarimadata,p,d,q):
        try:
            model = SARIMAX(sarimadata, order=(p, d, q), seasonal_order=(1, 1, 1, 1))
            model_fit = model.fit(disp=False)
            # make prediction
            sarima_pred = model_fit.predict(len(sarimadata), len(sarimadata))
            print(sarima_pred)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_SARIMAX(sarimaxdata,exodata,exopreddata,p,d,q):
        try:
        # SARIMAX example
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from random import random
            # contrived dataset
            data1 = [sarimaxdata]
            data2 = [exodata]
            # fit model
            model = SARIMAX(data1, exog=data2, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            # make prediction
            exog2 = [exopreddata]
            print(exog2)
            yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
            print(yhat)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_VAR(vardata):
        try:
            vardata = list()
            model = VAR(vardata)
            model_fit = model.fit()
            # make prediction
            yhat = model_fit.forecast(model_fit.y, steps=1)
            print(yhat)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_VARMA(varmadata):

        try:
            varmadata = list()
              # fit model
            model = VARMAX(varmadata, order=(1, 1))
            model_fit = model.fit(disp=False)
            # make prediction
            yhat = model_fit.forecast()
            print(yhat)
        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_SES(sesdata):
        try:
            model = SimpleExpSmoothing(sesdata)
            model_fit = model.fit()
            # make prediction
            ses_pred = model_fit.predict(len(sesdata), len(sesdata))
            print(ses_pred)

        except:
            return ("Please verify data : It should be one dimensional and list type.")

    def time_series_HWES(hwsedata):

        try:

            model = ExponentialSmoothing(hwsedata)
            model_fit = model.fit()
            # make prediction
            hwse_pred = model_fit.predict(len(hwsedata), len(hwsedata))
            print(hwse_pred)
        except:
            print ("Please verify data : It should be one dimensional and list type.")


