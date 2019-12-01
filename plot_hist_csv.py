
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pdb

def transform_data(indata_df, type, print_data_head= False, plot_hist = True):
    if type == "std":
        scale = preprocessing.StandardScaler().fit(indata_df)
    elif type == "255":
        scale = preprocessing.MinMaxScaler(feature_range=(0,255)).fit(training)
    elif type == "robust":
        scale = preprocessing.RobustScaler(quantile_range=(25,75)).fit(training)

    data_trans = scale.transform(indata_df)
    data_trans_df = pd.DataFrame(data_trans, index=indata_df.index, columns=indata_df.columns)

    if print_data_head:
        print(data_trans_df.head())

    if plot_hist:
        data_trans_df.hist(bins = 10)

    return data_trans_df


csv_dir = "data/george_data_13bands_add_alltraining.csv"

df = pd.read_csv(csv_dir)

print(df.head())


columns_tobe_hist = ["ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
                   "s2_r", "s2_nir", "hv", "vv","seg"]

#band_labels = ["Id", "ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
#               "s2_r", "s2_nir", "hv", "vv", "s2_20m_1", "s2_20m_2", "s2_20m_3", "s2_20m_4", "s2_20m_5", "s2_20m_6",
#               "seg"]
class_name_dic = {
    1: "Forest",
    2: "Open Forest",
    3: "Cocoa",
    4: "AF Cocoa",
    5: "Agricultural land",
    6: "Oil Palm",
    7: "Built-up"}

#plot all the histgrams
df.hist(bins = 10, column=columns_tobe_hist)
#plt.show()

#for embed this to model training;
#https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029
training = df[columns_tobe_hist]

#training_norm_std_df = transform_data(indata_df=training,type="std")
training_norm_255_df = transform_data(indata_df=training,type="255")

print(training_norm_255_df.describe())
pdb.set_trace()
#training_norm_robust_df = transform_data(indata_df=training,type="robust")

df.update(training_norm_255_df)
# check HV and VV outliner

for column_name in columns_tobe_hist:
    fig, axes = plt.subplots(1, 7, figsize=(10, 2), dpi=100, sharex=True, sharey=True)
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:cyan', 'tab:orange']

    for i, (ax, class_name) in enumerate(zip(axes.flatten(), df.class_name.unique())):
        x = df.loc[df.class_name == class_name, column_name]
        ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(class_name), color=colors[i])
        ax.set_title(class_name_dic[class_name])

    plt.suptitle(column_name, y=0.1, size=16)

    if column_name == 'hv' or column_name == 'vv':
        ax.set_xlim(-2, 10);ax.set_ylim(0, 0.8)
    elif column_name =="s2_rep":
        ax.set_xlim(220, 250);
        ax.set_ylim(0, 0.8)

    else:
        ax.autoscale_view()
    #
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()
    plt.close()







