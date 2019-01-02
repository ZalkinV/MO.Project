import pandas as pd
import numpy as np
from sklearn import model_selection

import matplotlib.pyplot as plt

import xgboost as xgb

def show_graphs(features_train, labels, features_test, hypothesis, columns_names=None, n_max=200, rand_seed=0):
	def select_rows(columns):
		data_size = 1 - n_max / features_train.shape[0] if n_max < features_train.shape[0] else 0 
		data_splitted = model_selection.train_test_split(columns, labels, test_size=data_size, random_state=rand_seed)
		X_train, y_train = data_splitted[0], data_splitted[2]
		return X_train, y_train

	feature_importances = select_feature_importances(features_train, labels, hypothesis=hypothesis)
	picked_features = features_train[feature_importances["Feature"][:4]]
	X, y = select_rows(picked_features)

	plot_dependencies(X, y)
	if hasattr(hypothesis, "feature_importances_"):
		xgb.plot_importance(hypothesis)

	plt.show()
	pass

def select_feature_importances(features, labels, sort=True, hypothesis=None):
	if hypothesis is not None and hasattr(hypothesis, "feature_importances_"):
		importances = hypothesis.feature_importances_
	else:
		importances = build_xgb_regr(features, labels).feature_importances_
			
	feature_importance = pd.DataFrame()
	feature_importance.insert(loc=0, column="Importance", value=importances)
	feature_importance.insert(loc=0, column="Feature", value=features.columns)

	if sort == True:
		feature_importance.sort_values(by="Importance", axis=0, ascending=False, inplace=True)
		
	return feature_importance

def plot_dependencies(X, y):
	fig, axes = plt.subplots(2, 2)
	fig.suptitle("Dependence label from features")
	plt.subplots_adjust(wspace=0.3, hspace=0.4)

	for i in range(4):
		current_feature = X.iloc[:, i]
		plt.subplot(2, 2, i + 1)
		
		plt.xlabel(current_feature.name)
		plt.ylabel("target")
		plt.plot(current_feature, y, 'bx', markersize=4)

	return fig

def build_xgb_regr(features, labels):
	return xgb.XGBRegressor().fit(features, labels)
