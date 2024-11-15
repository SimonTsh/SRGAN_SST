import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'statistics/di-lab_4_train_results.csv' #srf_4_train_results.csv' # 
df = pd.read_csv(data_dir)

df = df.set_index(df.columns[0]) # set first column as the index

fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=(10, 16)) # plot all other columns as variables

df.iloc[:,0:2].plot(ax=ax1)
ax1.set_xlabel('')
ax1.legend(df.columns[0:2])

# df.iloc[:,2:4].plot(ax=ax2)
# ax2.set_xlabel('')
# ax2.legend(df.columns[2:4])

df.iloc[:,4].plot(ax=ax3,label=df.columns[4])
ax3.set_xlabel('')
ax3.legend()

df.iloc[:,5].plot(ax=ax4,label=df.columns[5])
ax4.legend()

plt.xlabel(df.index.name)
plt.tight_layout()
plt.savefig(data_dir.split('.')[0]+'.png', dpi=300, bbox_inches='tight')