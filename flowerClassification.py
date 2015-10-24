#%matplotlib inline
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

iris_data = pd.read_csv('iris.data.txt')

assert len(iris_data['Species'].unique()) == 3

assert iris_data.loc[iris_data['Species'] == 'Iris-versicolor', 'Sepal Length'].min() >= 2.5

assert len(iris_data.loc[(iris_data['Sepal Length'].isnull()) |
                            (iris_data['Sepal Width'].isnull()) |
                            (iris_data['Petal Length'].isnull()) |
                            (iris_data['Petal Width'].isnull())
                        ]) == 0
all_inputs = iris_data[[
                        'Sepal Length',
                        'Sepal Width',
                        'Petal Length',
                        'Petal Width'
                        ]].values

all_classes = iris_data['Species'].values

random_forest_classifier = RandomForestClassifier()

parameter_grid = {'n_estimators': [5, 10, 25, 50],
                  'criterion': ['gini', 'entropy'],
                  'max_features': [1, 2, 3, 4],
                  'warm_start': [True, False]}

cross_validation = StratifiedKFold(all_classes, n_folds=10)

grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(all_inputs, all_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_search.best_estimator_

random_forest_classifier = grid_search.best_estimator_

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75)
 
random_forest_classifier.fit(training_inputs, training_classes)
 
for input_features, prediction, actual in zip(testing_inputs[:10],
                                                random_forest_classifier.predict(testing_inputs[:10]),
                                                testing_classes[:10]):
                                                    print('{}\t-->\t{}\t(Actual: {})'.format(input_features, prediction, actual))