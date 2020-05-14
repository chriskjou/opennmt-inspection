# import nibabel as nib

# save values
# img = nib.load("../Neurosynth_Parcellation_2.nii")  
# data = np.rint(np.array(img.dataobj))

# resize in matlab
# initial = zeros(79,95,68);
# initial(:) = imresize3(data, size(initial), 'nearest');

# initial saved as neurosynth_labels

import numpy as np
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

# get layers
print("getting data...")
layers = []
for i in range(1, 13):  
	layer = pickle.load(open("../rsa_neurosynth/bertmodel2brain_cv_-subj1-avg_layer" + str(i) + ".p", "rb"))  
	layers.append(layer)  

layers = pd.DataFrame(np.array(layers))
print(layers.head())
layer_label = pd.DataFrame({'layer': list(range(1, 13))})

data = pd.concat([layers, layer_label], axis=1)  

print("melting...")
df_melt = data.melt('layer', var_name='region',  value_name='vals') 

print(len(df_melt))
df_melt = df_melt.dropna()
print(len(df_melt))

print("plotting...")
sns.set(style="darkgrid")
plt.figure(figsize=(24, 9))
g = sns.pointplot(x="layer", y="vals", hue="region", data=df_melt, plot_kws=dict(alpha=0.3)) 
figure = g.get_figure()  
box = g.get_position()
g.set_position([box.x0, box.y0, box.width * .85, box.height])
g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=4)
figure.savefig("../test_rsa.png", bbox_inches='tight')