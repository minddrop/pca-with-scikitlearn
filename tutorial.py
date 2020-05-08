import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from datetime import date


data = pd.read_csv("./movie_metadata.csv")
# print(data.shape)
# data.count()

df = data.drop(["gross", "budget"], axis=1).dropna(axis=0)


# fig, ax = plt.subplots(figsize=(15,15))
# sns.heatmap(data=data.corr(), annot=True)

# data.loc[data.color == ' Black and White'].title_year.mean()

df = pd.concat([df, data.loc[df.index, ["gross", "budget"]]], axis=1)
# df.head(5)
df.reset_index(drop=True, inplace=True)
# df.columns
# sns.boxplot(x="color", y="title_year", data=df, palette="PRGn")


cut = pd.cut(df.imdb_score, bins=list(np.arange(1, 11)))
cut2 = pd.cut(df.title_year, bins=list(5 * (np.arange(380, 405))))
cut3 = pd.cut(df.imdb_score, bins=list([0, 4, 6, 7, 8, 10]))

df["imdb_score_bin"] = cut
df["year_range"] = cut2
df["pc_imdb"] = cut3

le = LabelEncoder()
df["pc_imdb"] = le.fit_transform(df["pc_imdb"])

# fig, ax = plt.subplots(figsize=(10, 10))
# plt.xticks(rotation=45)
# sns.barplot(df["year_range"], df["budget"], ci=None)
# sns.barplot(df["year_range"], df["budget"], ci=None)


# fig, ax = plt.subplots(figsize=(10, 10))
# plt.xticks(rotation=45)
# sns.barplot(df["year_range"], df["budget"], ci=None)
# sns.barplot(df["year_range"], df["gross"], ci=None)


# sns.barplot(df["imdb_score_bin"], df["gross"], ci=None)


# sns.boxplot(data=df, x="imdb_score_bin", y="gross")


mean_chart = pd.DataFrame(df.groupby(by=["year_range"])["budget"].mean())


df = pd.merge(df, mean_chart, left_on="year_range", right_index=True)

# df.columns

df["budget_x"].fillna(df["budget_y"], inplace=True)
# df["budget_x"].count()


df2 = df


var_mod = ["imdb_score_bin", "year_range"]
le = LabelEncoder()
for i in var_mod:
    df2[i] = le.fit_transform(df2[i])


clf = DecisionTreeRegressor()

# df.budget.fillna(0,inplace =True)

clf.fit(
    df[df["gross"].notnull()][["imdb_score_bin", "year_range"]],
    df["gross"].dropna(axis=0),
)

pred = clf.predict(df[df["gross"].isnull()][["imdb_score_bin", "year_range"]])

df[df["gross"].isnull()][["imdb_score_bin", "year_range"]].index

j = 0
for i in df[df["gross"].isnull()][["imdb_score_bin", "year_range"]].index:
    df["gross"][i] = pred[j]
    j = j + 1


df_genre = (
    df["genres"].str.split("|", expand=True).stack().str.get_dummies().sum(level=0)
)

# fig, ax = plt.subplots(figsize=(10, 10))
# plt.xticks(rotation=45)
# k = pd.DataFrame(df_genre.sum(), columns=["sum"])
# sns.barplot(y="sum", x=k.index, data=k, orient="v")


today = date.today()
df["age"] = today.year - df.title_year


k = df.groupby(by="director_name", sort=False).director_facebook_likes.mean()
l = df.groupby(by="director_name", sort=False).imdb_score.sum()
m = df.groupby(by="director_name", sort=False).age.max()
pd.DataFrame(df["director_name"].value_counts())
dir_ran = pd.concat([k, l, m], axis=1)


col_5 = list(df["director_name"].value_counts().index[:5])
col_5


pp = df.loc[
    (df.director_name == col_5[0])
    | (df.director_name == col_5[1])
    | (df.director_name == col_5[2])
    | (df.director_name == col_5[3])
    | (df.director_name == col_5[4])
]

sns.boxplot(x="director_name", y="imdb_score", data=pp)


str_list = []  # empty list to contain columns with strings (words)
for colname, colvalue in df.iteritems():
    if type(colvalue[1]) == str:
        str_list.append(colname)
num_list = df.columns.difference(str_list)
print(num_list)
X = df[num_list]
X.shape


X_std = StandardScaler().fit_transform(X)


sklearn_pca = PCA(n_components=20)
Y_sklearn = sklearn_pca.fit_transform(X_std)

cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

sklearn_pca.explained_variance_ratio_[:10].sum()

cum_sum = cum_sum * 100

fig, ax = plt.subplots(figsize=(8, 8))
plt.bar(
    range(20),
    cum_sum,
    label="Cumulative _Sum_of_Explained _Varaince",
    color="b",
    alpha=0.5,
)
plt.show()
