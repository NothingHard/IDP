# IDP notes

## A. multi-layer perceptron (MLP) on MNIST
### Experiment setting
|epoch|batch_size|batch_per_epoch|optimizer|learning_rate|early-stopping|
|-----|----------|---------------|---------|-------------|--------------|
| 50  | 32       | 200           | Adam    | 0.001       | patience = 4 |



### Performance comparisons
|  Method    | Avg. Accuracy (%) |
|----------  |-------------------|
|Individual  |   96.16316        |
|R-TESLA+ATP |   95.47474        |
|R-TESLA     |   95.15053        |
|TESLA       |   94.66842        |
|TESLA+ATP   |   94.03316        |
|Original IDP|   88.36737        |

Note that dot product percentages under test range from 10 to 100 by interval of 5.

Observations:
- **Individual** ensembles 19 models to achieve average accuracy of 96.16%
- **R-TESLA+ATP** empowers a single model to achieve 95.47% accuracy
- In this experiment, **Original IDP** cannot reach accuracy of 90%

## 
