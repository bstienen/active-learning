import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


home = ''
data = home+'./'
diff = 4


sample_times = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1000000]
sample_time_names = ['n_parameters', '1us', '10us', '100us', '1ms', '10ms', '100ms', '1s', '10s', '100s', '1000s', 'inf']
parameter_counts = [19, 14, 10, 8, 4, 1]

results = pd.DataFrame(columns=sample_time_names)

for i, num_params in enumerate(parameter_counts):

	step = int(2500/((num_params/19)**2))
	al = pd.read_csv("{}/active_{}params.csv".format(data, num_params))
	rs = pd.read_csv("{}/random_{}params.csv".format(data, num_params))

	# Calculate which active learning steps to include
	rs_best_accs = rs.groupby(['iteration']).max()['acc']
	for iteration in rs['iteration'].unique():
		selected = al[al.iteration == iteration]
		al = al[al.iteration != iteration]
		al = al.append( selected[selected.acc <= rs_best_accs[iteration]] )

	# Calculate number of samples for both cases
	al_samples = al.groupby(['iteration']).max().mean()['size']
	rs_samples = rs.groupby(['iteration']).max().mean()['size']

	# Calculate total time to reach best random accuracy
	al_time = al.groupby(['iteration']).sum().mean()['dt']
	rs_time = rs.groupby(['iteration']).mean().mean()['dt']

	# Calculate time needed to perform active learning without the sampling (i.e. the AL overhead)
	al_time_raw = al_time - rs_time

	# Calculate time improvement for set of sample times
	entry = {'n_parameters': int(num_params)}
	for j,stime in enumerate(sample_times):
		random_time = stime * rs_samples
		active_time = stime * al_samples + al_time_raw
		entry[sample_time_names[j+1]] = random_time / active_time
	results = results.append(entry, ignore_index=True)


plt.plot(results[sample_time_names[1:]].transpose())
#plt.yscale('log')
plt.show()


results = results.round(4)
results.n_parameters = results.n_parameters.astype(int)
table_columns = results.keys()

print('\\begin{tabular}{ll'+('c'*(len(table_columns)-2))+'}')
print('\hline')
print('\hline')
print('\\textbf{'+'} & \\textbf{'.join(table_columns) + '}\\\\')
print('\hline')
for i in range(len(results)):
	print( ' & '.join(map(str, results[table_columns].iloc[i]))+'\\\\' )
print('\hline')
print('\hline')
print("\\end{tabular}")

print(results)

