import pandas as pd
from sklearn import preprocessing

import xgboost as xgb

def main():
	label_name = "target"

	data_raw_train = pd.read_csv("../train_clear.csv", index_col="ID")
	data_raw_test = pd.read_csv("../test_clear.csv", index_col="ID")
	
	labels_train = data_raw_train[label_name]
	features_train = preprocess_data(data_raw_train.drop(label_name, axis=1))
	del data_raw_train

	features_test = preprocess_data(data_raw_test)
	del data_raw_test

	print(features_train)
	print(features_test)
	pass

def preprocess_data(data_raw):
	data_processing = data_raw.copy(deep=True)

	data_processing.columns = ["f" + str(i) for i in range(1, data_processing.shape[1] + 1)]

	quantile_values = data_processing.quantile(q=0.85)
	data_processing = data_processing.loc[:, quantile_values > 0]
	
	data_processing[:] = preprocessing.minmax_scale(data_processing)

	return data_processing

def build_hypothesis(features_train, labels_train):
	hypothesis = xgb.XGBRegressor(objective="reg:linear")
	hypothesis.fit(features_train, labels_train)
	return hypothesis

def create_processed_datafile(file_name, label=None, suffix="_clear"):
	data_raw = pd.read_csv(file_name, sep=',', index_col="ID")

	if label is not None:
		data_ready = preprocess_data(data_raw.drop(label, axis=1))
		data_ready.insert(loc=0, column=label, value=data_raw[label])
	else:
		data_ready = preprocess_data(data_raw)

	new_file_name = file_name.rsplit(sep='.', maxsplit=1)[0] + suffix + ".csv"
	data_ready.to_csv(new_file_name)
	pass

main()
