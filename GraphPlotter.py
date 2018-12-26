import pandas as pd
import numpy as np
from sklearn import model_selection

import matplotlib.pyplot as plt

import xgboost as xgb

def show_graphs(features, labels, hypothesis=None, columns_names=None, n_max=200, rand_seed=0):
	def select_features():
		X_train = None

		if columns_names is None:
			importances = xgb.XGBRegressor().fit(features, labels).feature_importances_
			importances = importances.reshape(1, importances.size)
			
			feature_importance = pd.DataFrame(data=importances, columns=features.columns)
			feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
			
			X_train = features[feature_importance.columns[:4]]
		else:
			X_train = features[columns_names[:4]]
		
		return X_train

	def select_rows(columns):
		data_size = 1 - n_max / features.shape[0]
		data_splitted = model_selection.train_test_split(columns, labels, test_size=data_size, random_state=rand_seed)
		X_train, y_train = data_splitted[0], data_splitted[2]
		return X_train, y_train

	picked_features = select_features()
	X, y = select_rows(picked_features)

	fig_dep, axes = plt.subplots(2, 2)
	fig_dep.suptitle("Dependence label from features")
	plt.subplots_adjust(wspace=0.3, hspace=0.4)

	for i in range(4):
		current_feature = X.iloc[:, i]
		plt.subplot(2, 2, i + 1)
		
		plt.xlabel(current_feature.name)
		plt.ylabel("target")
		plt.plot(current_feature, y, 'bx', markersize=4)

	plt.show()
	pass