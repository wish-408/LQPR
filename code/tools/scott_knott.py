import scikit_posthocs as sp
import pandas as pd


def scott_knott_clustering(group_names, stats_results):
    data = {'Group': [], 'Value': []}
    for i, group in enumerate(group_names):
        for value in stats_results[i]:
            data['Group'].append(group)
            data['Value'].append(value)
    df = pd.DataFrame(data)

    sk_result = sp.posthoc_skott(df, val_col='Value', group_col='Group')

    clusters = {}
    for i in range(len(group_names)):
        for j in range(len(group_names)):
            if sk_result.iloc[i, j] == 0:
                if group_names[i] not in clusters:
                    clusters[group_names[i]] = []
                if group_names[j] not in clusters[group_names[i]]:
                    clusters[group_names[i]].append(group_names[j])

    sorted_groups = sorted(group_names, key=lambda x: df[df['Group'] == x]['Value'].mean(), reverse=True)

    return clusters, sorted_groups

# 示例调用
group_names = ['A', 'B', 'C']
stats_results = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
clusters, sorted_groups = scott_knott_clustering(group_names, stats_results)
print("聚类结果:", clusters)
print("排序结果:", sorted_groups)
