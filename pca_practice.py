import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from datetime import date
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


movie_data = pd.read_csv("./movie_metadata.csv")

# sns.heatmap(
#     movie_data.corr(),
#     linewidths=0.25,
#     vmax=1.0,
#     square=True,
#     cmap=sns.color_palette("BuGn"),
#     linecolor="black",
#     annot=True,
# )
# plt.show()

# calculate data length that contains many missing values
# nullnum = []
# for i in range(13, 29)[::-1]:
#     count = len(movie_data.dropna(thresh=i, axis=0))
#     nullnum.append(count)
# nullnum = pd.DataFrame(
#     {"num_of_missing_data": list(range(29 - 13)), "data_length": nullnum}
# )
# sns.catplot(
#     x="num_of_missing_data", y="data_length", data=nullnum, kind="bar", color="g"
# )
# plt.show()

# delete data that contains many missing values
df = movie_data.dropna(thresh=22, axis=0)

# reduce dimension for drop nan values except gross and budget
df.isnull().sum(axis=0).plot.bar(color="g")
plt.show()
# sns.catplot(df.isnull().sum(axis=0))
# plt.show()
print(df.isnull().sum())
print(df.isnull().any(axis=1).sum())
print(df.drop(["gross", "budget"], axis=1).isnull().sum())
print(df.drop(["gross", "budget"], axis=1).isnull().any(axis=1).sum())
print(
    (
        df.isnull().any(axis=1) & df(["gross", "budget"], axis=1).isnull().any(axis=1)
    ).sum()
)
df = movie_data.drop(["gross", "budget"], axis=1).dropna(how="any", axis=0)
df = pd.concat([df, movie_data.loc[df.index, ["gross", "budget"]]], axis=1)


# reindex because deleted deleted parameter
df.reset_index(drop=True, inplace=True)
# sns.boxplot(x="color", y="title_year", data=df, palette="PRGn")


cut = pd.cut(df.imdb_score, bins=list(np.arange(1, 11)))
cut2 = pd.cut(df.title_year, bins=list(5 * (np.arange(380, 405))))
cut3 = pd.cut(df.imdb_score, bins=list([0, 4, 6, 7, 8, 10]))
# cut3 = pd.qcut(df.imdb_score, 5)
df["imdb_score_bin"] = cut
df["year_range"] = cut2
df["pc_imdb"] = cut3
le = LabelEncoder()
df["pc_imdb"] = le.fit_transform(df["pc_imdb"])
fig, ax = plt.subplots(figsize=(10, 10))
# sns.barplot(df["year_range"], df["budget"], ci=None)
sns.barplot(df["imdb_score_bin"], df["gross"], ci=None)
# sns.barplot(df["pc_imdb"], df["gross"], ci=None)
# sns.boxplot(data=df, x="imdb_score_bin", y="gross")

mean_chart = pd.DataFrame(df.groupby(by=["year_range"])["budget"].mean())
df = pd.merge(df, mean_chart, left_on="year_range", right_index=True)
# delete nan values
df["budget_x"].fillna(df["budget_y"], inplace=True)
# print(df["budget_x"].count())


# apply the decision tree regression to fill the gross
var_mod = ["imdb_score_bin", "year_range"]
for i in var_mod:
    df[i] = le.fit_transform(df[i])
clf = DecisionTreeRegressor()
clf.fit(
    df[df["gross"].notnull()][["imdb_score_bin", "year_range"]],
    df["gross"].dropna(axis=0),
)
pred = clf.predict(df[df["gross"].isnull()][["imdb_score_bin", "year_range"]])
j = 0
for i in df[df["gross"].isnull()][["imdb_score_bin", "year_range"]].index:
    df["gross"][i] = pred[j]
    j = j + 1
# df_genre = (
#     df["genres"].str.split("|", expand=True).stack().str.get_dummies().sum(level=0)
# )
# fig, ax = plt.subplots(figsize=(10, 10))
# plt.xticks(rotation=45)
# k = pd.DataFrame(df_genre.sum(), columns=["sum"])
# sns.barplot(y="sum", x=k.index, data=k, orient="v")


# calculate the age of the movie
today = date.today()
df["age"] = today.year - df.title_year
# k = df.groupby(by="director_name", sort=False).director_facebook_likes.mean()
# l = df.groupby(by="director_name", sort=False).imdb_score.sum()
# m = df.groupby(by="director_name", sort=False).age.max()
# pd.DataFrame(df["director_name"].value_counts())
# dir_ran = pd.concat([k, l, m], axis=1)
# dir_ran_sorted = dir_ran.sort_values(by="imdb_score", ascending=False)

# col_5 = list(df["director_name"].value_counts().index[:5])

# pp = df.loc[
#     (df.director_name == col_5[0])
#     | (df.director_name == col_5[1])
#     | (df.director_name == col_5[2])
#     | (df.director_name == col_5[3])
#     | (df.director_name == col_5[4])
# ]

# sns.boxplot(x="director_name", y="imdb_score", data=pp)


str_list = []  # empty list to contain columns with strings (words)
for colname, colvalue in df.iteritems():
    if type(colvalue[1]) == str:
        str_list.append(colname)
num_list = df.columns.difference(str_list)
X = df[num_list]


X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=20)
Y = pca.fit_transform(X_std)
cum_sum = pca.explained_variance_ratio_.cumsum()
pca.explained_variance_ratio_[:10].sum()
cum_sum = cum_sum * 100
fig, ax = plt.subplots(figsize=(8, 8))
# plt.bar(
#     range(20),
#     cum_sum,
#     label="Cumulative _Sum_of_Explained _Varaince",
#     color="b",
#     alpha=0.5,
# )


pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_std)
Y = df["pc_imdb"]

plt.clf()
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

# plt.show()
