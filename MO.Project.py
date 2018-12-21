import pandas as pd

def main():
	data_raw = pd.read_csv("../train.csv", sep=',', index_col="ID", nrows=500)
	print(data_raw)
	pass

main()
