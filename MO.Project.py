import pandas as pd

def main():
	data_raw = pd.read_csv("../train.csv", sep=',', index_col="ID", nrows=200)
	
	label_name = "target"
	features_ready = preprocess_data(data_raw.drop(label_name, axis=1))

	print(features_ready)
	pass

def preprocess_data(data_raw):
	data_processing = data_raw.copy(deep=True)

	quantiles = data_processing.quantile(q=0.70)
	data_processing = data_processing.loc[:, quantiles > 0]
	
	return data_processing

main()
