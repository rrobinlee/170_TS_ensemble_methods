## Random Forest and Gradient Boosting Algorithms for Time Series

### Benchmark Regression Model

![image](https://github.com/user-attachments/assets/28798cb9-7b16-4156-beae-64e7e1d5e152)

* **Lag:** We decided to use Lag 12, because the data has seasonality. We can see this from the left hand side of the decomposition plot and from the `tsfeatures()` package, which has the value of 12.
* **Trend:** This feature is important as well, because `Unemployment`—our dependent variable—possesses trend as exemplified in the decomposition above. Furthermore, we believe that the trend is significant, since there is a change in trend when we use the `tsfeatures()` function. We were able to do this by choosing the first three years and comparing it with the next three.
* **Trend Squared:** By using the `tsfeatures()` function, we also believe that trend squared is significant by using the same method as above. We select chunks of our time series, before selecting tsfeatures for them. The trend squared appears to change as well.
* **Frequency:** Our data is monthly, so we believe that this is an important feature of Unemployment and we should probably use it for modeling.
