from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

data = load_boston()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lreg = LazyRegressor()
models, predictions = lreg.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print(models.to_markdown())

""" RESULTS:

| Model                         |   Adjusted R-Squared |   R-Squared |     RMSE |   Time Taken |
|:------------------------------|---------------------:|------------:|---------:|-------------:|
| GradientBoostingRegressor     |            0.887201  |  0.896912   |  3.0591  |   0.169985   |
| RandomForestRegressor         |            0.860975  |  0.872944   |  3.39615 |   0.333036   |
| ExtraTreesRegressor           |            0.859213  |  0.871334   |  3.4176  |   0.198031   |
| XGBRegressor                  |            0.844723  |  0.858091   |  3.58916 |   0.121968   |
| HistGradientBoostingRegressor |            0.844371  |  0.857769   |  3.59323 |   0.516886   |
| LGBMRegressor                 |            0.842659  |  0.856205   |  3.61294 |   0.0749984  |
| AdaBoostRegressor             |            0.829432  |  0.844117   |  3.76174 |   0.0970004  |
| BaggingRegressor              |            0.824871  |  0.839949   |  3.8117  |   0.0370347  |
| DecisionTreeRegressor         |            0.742419  |  0.764595   |  4.62271 |   0.00999999 |
| KNeighborsRegressor           |            0.732458  |  0.755492   |  4.71125 |   0.0120001  |
| PoissonRegressor              |            0.697782  |  0.723801   |  5.00726 |   0.0119994  |
| TransformedTargetRegressor    |            0.604037  |  0.638126   |  5.73149 |   0.00803423 |
| LinearRegression              |            0.604037  |  0.638126   |  5.73149 |   0.0270116  |
| Lars                          |            0.604037  |  0.638126   |  5.73149 |   0.0150006  |
| Ridge                         |            0.603191  |  0.637353   |  5.73761 |   0.00699472 |
| LassoCV                       |            0.601955  |  0.636224   |  5.74654 |   0.0779998  |
| LassoLarsIC                   |            0.600949  |  0.635305   |  5.75379 |   0.059135   |
| LassoLarsCV                   |            0.600657  |  0.635038   |  5.7559  |   0.0259981  |
| LarsCV                        |            0.599898  |  0.634344   |  5.76137 |   0.033999   |
| BayesianRidge                 |            0.599328  |  0.633823   |  5.76547 |   0.0259643  |
| ExtraTreeRegressor            |            0.599091  |  0.633606   |  5.76718 |   0.009969   |
| ElasticNetCV                  |            0.597511  |  0.632162   |  5.77853 |   0.077035   |
| RidgeCV                       |            0.596805  |  0.631517   |  5.78359 |   0.00696826 |
| SGDRegressor                  |            0.593919  |  0.628879   |  5.80426 |   0.00999975 |
| SVR                           |            0.578408  |  0.614704   |  5.91407 |   0.0210001  |
| NuSVR                         |            0.565353  |  0.602773   |  6.00494 |   0.0220008  |
| OrthogonalMatchingPursuitCV   |            0.554607  |  0.592952   |  6.07872 |   0.0558507  |
| Lasso                         |            0.539031  |  0.578717   |  6.1841  |   0.00999808 |
| GammaRegressor                |            0.529769  |  0.570253   |  6.24591 |   0.00899553 |
| ElasticNet                    |            0.524276  |  0.565232   |  6.28229 |   0.0463424  |
| MLPRegressor                  |            0.514107  |  0.555939   |  6.34908 |   0.5555     |
| HuberRegressor                |            0.511223  |  0.553303   |  6.36789 |   0.0199966  |
| TweedieRegressor              |            0.510697  |  0.552823   |  6.37132 |   0.00800133 |
| PassiveAggressiveRegressor    |            0.489293  |  0.533261   |  6.50918 |   0.0129991  |
| LinearSVR                     |            0.486295  |  0.530521   |  6.52826 |   0.0249848  |
| OrthogonalMatchingPursuit     |            0.446442  |  0.494099   |  6.77676 |   0.0100009  |
| GaussianProcessRegressor      |            0.227951  |  0.294419   |  8.00318 |   0.0734396  |
| RANSACRegressor               |            0.18691   |  0.256912   |  8.21314 |   0.101094   |
| LassoLars                     |           -0.0963124 | -0.00192791 |  9.5369  |   0.00900221 |
| DummyRegressor                |           -0.0963124 | -0.00192791 |  9.5369  |   0.00699973 |
| QuantileRegressor             |           -0.103811  | -0.00878064 |  9.56946 |   0.763288   |
| KernelRidge                   |           -6.00406   | -5.40106    | 24.1054  |   0.0150025  |
"""
