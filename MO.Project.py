import pandas as pd
from sklearn import preprocessing

import xgboost as xgb

label_name = "target"

def main():
	data_raw_train = pd.read_csv("../train.csv", index_col="ID", nrows=200)
	
	labels_train = data_raw_train[label_name]
	features_train = preprocess_data(data_raw_train.drop(label_name, axis=1))
	del data_raw_train

	print(features_train)
	pass

def preprocess_data(data_raw):
	data_processing = data_raw.copy(deep=True)

	quantile_values = data_processing.quantile(q=0.70)
	data_processing = data_processing.loc[:, quantile_values > 0]
	
	data_processing[:] = preprocessing.minmax_scale(data_processing)
	
	data_processing.columns = ["f" + str(i) for i in range(1, data_processing.shape[1] + 1)]

	return data_processing

def build_hypothesis(features_train, labels_train):
	hypothesis = xgb.XGBRegressor(objective="reg:linear")
	hypothesis.fit(features_train, labels_train)
	return hypothesis

def create_processed_datafile(file_name, suffix="_clear"):
	data_raw = pd.read_csv(file_name, sep=',', index_col="ID")
	data_ready = preprocess_data(data_raw.drop(label_name, axis=1))

	data_ready.insert(loc=0, column=label_name, value=data_raw[label_name])

	new_file_name = file_name.rsplit(sep='.', maxsplit=1)[0] + suffix + ".csv"
	data_ready.to_csv(new_file_name)
	pass

main()
