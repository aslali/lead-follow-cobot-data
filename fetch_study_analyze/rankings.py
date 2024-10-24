# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# patterns = ['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4']
# rankings = ['Most Difficult', 'Rank 2', 'Rank 3', 'Least Difficult']
# data = np.array([
#     [30, 8, 9, 11],
#     [19, 15, 17, 7],
#     [9, 31, 12, 6],
#     [0, 4, 20, 34]
# ])
#
# # Create a stacked bar chart
# fig, ax = plt.subplots(figsize=(10, 6))
#
# for i, pattern in enumerate(patterns):
#     ax.bar(
#         rankings,
#         data[i],
#         label=f'Pattern {i+1}',
#         bottom=np.sum(data[:i], axis=0)
#     )
#
# ax.set_xlabel('Rankings')
# ax.set_ylabel('Count')
# ax.set_title('Stacked Bar Chart for Pattern Rankings')
# ax.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
#
# # Display the chart
# plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
data = {
    'Pattern': ['Pattern A', 'Pattern B', 'Pattern C', 'Pattern D'],
    'Least Difficult': [9, 7, 4, 28],
    'Rank 2': [8, 15, 7, 18],
    'Rank 3': [6, 11, 29, 2],
    'Most Difficult': [25, 15, 8, 0]
}

df = pd.DataFrame(data)

# Set the style
sns.set(style="whitegrid")

# Create the stacked bar chart
ax = df.set_index('Pattern').T.plot(kind='bar', stacked=True, figsize=(10, 6))

# Customize the plot
plt.xlabel('Rankings', fontsize=13)
plt.ylabel('#Participants',fontsize=13)
plt.yticks(range(0, 51, 5))
plt.ylim([0, 55])
ax.set_xticklabels(labels=['Rank 1\nLeast Difficult', 'Rank 2', 'Rank 3', 'Rank 4\nMost Difficult'])
plt.legend(loc='upper left', ncol=4)
plt.xticks(rotation=45, fontsize=11)
plt.tight_layout()
plt.savefig('results_part_rankings.eps', format='eps')
plt.show()
