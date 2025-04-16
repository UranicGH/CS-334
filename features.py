# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sb

# def corr(data):
#     correlation = data.corr('pearson')
#     print(sb.heatmap(correlation))
#     plt.savefig("heatmap.png")
#     return correlation

# def feature_selection(train, test):
#     correlation = corr(train)

#     remove = set()

#     for i in range(len(correlation.columns)):
#         for j in range(i):
#             if (correlation.iloc[i,j]) > 0.8:
#                 if (correlation.iloc[i, len(correlation.columns)-1] > correlation.iloc[j, len(correlation.columns)-1]):
#                     remove.add(correlation.columns[j])
#                 else:
#                     remove.add(correlation.columns[i])

#     # target = correlation['Sentiment'].drop('Sentiment')

#     # features = target[target.abs() < 0.1].index
#     # remove.update(features)

#     train = train.drop(columns=remove)
#     test = test.drop(columns=remove)

#     return train, test

# def main():
#     train = pd.read_csv('xTrain.csv')
#     test = pd.read_csv('xTest.csv')

#     print("Original Training Shape:", train.shape)
#     train, test = feature_selection(train, test)

#     print("Transformed Training Shape:", train.shape)
#     train.to_csv("select_train.csv", index=False)
#     test.to_csv("select_test.csv", index=False)

# if __name__ == "__main__":
#     main()