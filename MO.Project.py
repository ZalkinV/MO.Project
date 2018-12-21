import pandas as pd

def main():
	data_raw_train = pd.read_csv("../train.csv", sep=',', index_col="ID", nrows=200)
	
	label_name = "target"
	labels_ready = data_raw_train[label_name]
	features_ready = preprocess_data(data_raw_train.drop(label_name, axis=1))
	del data_raw_train

	print(features_ready)
	pass

def preprocess_data(data_raw):
	data_processing = data_raw.copy(deep=True)

	quantile_values = data_processing.quantile(q=0.70)
	data_processing = data_processing.loc[:, quantile_values > 0]
	
	return data_processing

main()
