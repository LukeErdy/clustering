import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.preprocessing import MinMaxScaler
import pandas
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def silhoutte_coefficient(n_clusters, data):
	model = KMeans(n_clusters = n_clusters)
	labels = model.fit_predict(data)
	score = silhouette_score(data, labels)
	return score

def optimize_n_clusters(r, data):
	if len(data) < r.stop:
		r = range(2, len(data))
	best_choice = r.start
	best_silhouette = float('-inf')
	for n_clusters in r:
		score = silhoutte_coefficient(n_clusters, data)
		print('%d clusters: %f' % (n_clusters, score))
		if score > best_silhouette:
			best_silhouette = score
			best_choice = n_clusters
	return best_choice, best_silhouette

def main():
	data = pandas.read_csv('Pokemon.csv')
	df = pandas.get_dummies(data)
	for column in df.columns:
		df[column].fillna(df[column].mean(), inplace = True)
	steps = [
			('scale', MinMaxScaler()),
			('cluster', KMeans())
			]
	pipe = Pipeline(steps)

	type_columns = [col for col in df.columns if 'Type 1' in col]
	dfs = []
	cluster_counts = []
	for type in type_columns:
		new_df = df[df[type] == True]
		new_df = pandas.DataFrame(new_df).copy()
		print(type[7:] + '\n-----------')
		best_choice, best_silhouette = optimize_n_clusters(range(2, 15), new_df)
		cluster_counts.append(best_choice)
		print('best number of clusters: %d\nbest score: %f\n' % (best_choice, best_silhouette))
		pipe.set_params(cluster__n_clusters = best_choice)
		pipe.fit(new_df)
		new_df.loc[:, 'Cluster'] = pipe['cluster'].labels_
		for cluster_id in new_df['Cluster'].unique():
			cluster_df = new_df.loc[new_df['Cluster'] == cluster_id].drop(columns = 'Cluster')
			merged_df = pandas.merge(data, cluster_df[['#']], on='#', how='inner')
			dfs.append(pandas.DataFrame(merged_df).copy())
	offset = 0
	for i in range(len(cluster_counts)):
		print('\n' + type_columns[i][7:] + '\n-----------')
		for j in range(cluster_counts[i]):
			print('Cluster %d' % (j))
			print(dfs[j + offset].drop(columns = ['Type 1', 'Type 2', 'Generation', 'Legendary']).to_string(index=False))
			print('Mean HP: %f\nMean Attack: %f\nMean Defense: %f\nMean Sp. Atk: %f\nMean Sp. Def: %f\nMean Speed: %f\n'
				% (dfs[j + offset]['HP'].mean(),
					dfs[j + offset]['Attack'].mean(),
					dfs[j + offset]['Defense'].mean(),
					dfs[j + offset]['Sp. Atk'].mean(),
					dfs[j + offset]['Sp. Def'].mean(),
					dfs[j + offset]['Speed'].mean()))
		offset += cluster_counts[i]

if __name__ == '__main__':
	main()