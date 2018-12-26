import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection

import xgboost as xgb

import matplotlib.pyplot as plt

def main():
	label_name = "target"

	quantile_value = 0.85

	path_train = "../train_clear_" + str(quantile_value) + ".csv"
	path_test = "../test_clear_" + str(quantile_value) + ".csv"
	path_submission = "../submission.csv"

	data_raw_train = pd.read_csv(path_train, index_col="ID")
	data_raw_test = pd.read_csv(path_test, index_col="ID")
	
	labels_train = data_raw_train[label_name]
	features_train = preprocess_data(data_raw_train.drop(label_name, axis=1), quantile_value)
	del data_raw_train

	features_test = preprocess_data(data_raw_test, quantile_value)
	del data_raw_test

	similar_features_names = get_similar_columns(features_train, features_test)
	features_train = features_train[similar_features_names]
	features_test = features_test[similar_features_names]

	hypothesis = build_hypothesis(features_train, labels_train)
	labels_test = hypothesis.predict(features_test)

	show_graphs(features_train, labels_train)
	#pd.DataFrame(labels_test, index=features_test.index, columns=[label_name]).to_csv(path_submission)
	#write_submission_info(hypothesis, quantile_value)
	pass

def show_graphs(features, labels, columns_names=None, n_max=200, rand_seed=0):
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
	
	for i in range(4):
		current_feature = X.iloc[:, i]
		plt.subplot(2, 2, i + 1)
		
		plt.xlabel(current_feature.name)
		plt.ylabel("target")
		plt.plot(current_feature, y, 'bx', markersize=4)

	plt.show()
	pass

def preprocess_data(data_raw, quantile, for_file=False):
	if for_file:
		data_raw.columns = ["f" + str(i) for i in range(1, data_raw.shape[1] + 1)]

	quantiled_values = data_raw.quantile(q=quantile)
	data_processing = data_raw.loc[:, quantiled_values > 0]

	if not for_file:
		data_processing[:] = preprocessing.minmax_scale(data_processing)

	return data_processing

def build_hypothesis(features_train, labels_train):
	hypothesis = xgb.XGBRegressor(objective="reg:linear")
	hypothesis.fit(features_train, labels_train)
	return hypothesis

def create_processed_datafile(file_name, quantile, label=None, suffix="clear"):
	data_raw = pd.read_csv(file_name, sep=',', index_col="ID")

	if label is not None:
		data_ready = preprocess_data(data_raw.drop(label, axis=1), quantile=quantile, for_file=True)
		data_ready.insert(loc=0, column=label, value=data_raw[label])
	else:
		data_ready = preprocess_data(data_raw, quantile=quantile, for_file=True)

	new_file_name = "_".join([file_name.rsplit(sep='.', maxsplit=1)[0], suffix, str(quantile)])
	data_ready.to_csv(new_file_name + ".csv")
	pass

def write_submission_info(hypothesis, quantile, extra_info=None):
	info = [
		f"**Hypothesis:** {hypothesis}",
		f"**Quantile:** {quantile}",
		]
	if extra_info is not None:
		info.append(f"{extra_info}")

	with open("submission_info.txt", 'w') as fo:
		fo.writelines("\n\n".join(info))
	pass

def get_similar_columns(df_first, df_second):
	unsorted_names = list(set(df_first.columns).intersection(df_second.columns))
	sorted_names = unsorted_names.copy()
	sorted_names.sort(key=lambda f: int(f[1:]))
	return sorted_names

main()
