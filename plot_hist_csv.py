
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

csv_dir = "data/george_data_13bands_add_alltraining.csv"

df = pd.read_csv(csv_dir)

print(df.head())

columns_tobe_hist = ["ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
                   "s2_r", "s2_nir", "hv", "vv","seg"]

#band_labels = ["Id", "ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
#               "s2_r", "s2_nir", "hv", "vv", "s2_20m_1", "s2_20m_2", "s2_20m_3", "s2_20m_4", "s2_20m_5", "s2_20m_6",
#               "seg"]

#plot all the histgrams
df.hist(bins = 10, column=columns_tobe_hist)
#plt.show()

#for embed this to model training;
#https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029
training = df[columns_tobe_hist]

std_scale = preprocessing.StandardScaler().fit(training)
training_norm_std = std_scale.transform(training)
training_norm_std_df = pd.DataFrame(training_norm_std, index=training.index, columns=training.columns)

print("====")
print(training_norm_std_df.head())
#fig, axes = plt.subplots(4, 4, figsize=(10,2.5), dpi=100, sharex=False, sharey=False)
#colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive']
training_norm_std_df.hist(bins=10, column=columns_tobe_hist)
#plt.show()

minmax_255_scale = preprocessing.MinMaxScaler(feature_range=(0,255)).fit(training)
training_norm_255 = minmax_255_scale.transform(training)
training_norm_255_df = pd.DataFrame(training_norm_255, index=training.index, columns=training.columns)
training_norm_255_df.hist(bins=10, column=columns_tobe_hist)
#plt.show()

robust_scale = preprocessing.RobustScaler(quantile_range=(25,75)).fit(training)
training_norm_robust = robust_scale.transform(training)
training_norm_robust_df = pd.DataFrame(training_norm_robust, index=training.index, columns=training.columns)
training_norm_robust_df.hist(bins=10, column=columns_tobe_hist)
#plt.show()

# check HV and VV outliner
df.update(training_norm_255_df)

fig, axes = plt.subplots(1, 7, figsize=(10, 2.5), dpi=100, sharex=True, sharey=True)
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:cyan', 'tab:orange']

#colors = ['red', 'blue', 'green', 'pink', 'olive', 'purple', 'cyan', 'orange']

for i, (ax, class_name) in enumerate(zip(axes.flatten(), df.class_name.unique())):
    x = df.loc[df.class_name == class_name, 'hv']
    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(class_name), color=colors[i])
    ax.set_title(class_name)

plt.suptitle('Distribution for each land use class', y=1.05, size=16)
ax.set_xlim(-2,10); ax.set_ylim(0, 2)
plt.tight_layout()
plt.show()





