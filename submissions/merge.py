import pandas as pd

csv_1 = pd.read_csv('fridge_freezer.csv')
csv_2 = pd.read_csv('washing_machine.csv')
csv_3 = pd.read_csv('TV.csv')
csv_4 = pd.read_csv('kettle.csv')

merged_1 = csv_1.merge(csv_2, on='time_step')
merged_2 = merged_1.merge(csv_3, on='time_step')
merged = merged_2.merge(csv_4, on='time_step')
merged.to_csv('merged.csv', sep=',', header=True, index=False)
