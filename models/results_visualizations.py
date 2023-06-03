import matplotlib.pyplot as plt
import numpy as np

# # dataset splits
# dataset_split = [0.2, 0.3, 0.35]

# # accuracies for each algorithm 
# accuracy_LR = [72.05, 72.48, 72.6]
# accuracy_DTC = [72, 72.2, 72.07]
# accuracy_RFC = [72.02, 72.27, 72.26]
# accuracy_NB = [38.85, 38.25, 38.2]
# accuracy_KNN = [61.15, 67, 72.3]

# # define bar width
# bar_width = 0.15

# # create bar positions
# r1 = np.arange(len(dataset_split))
# r2 = [x + bar_width for x in r1]
# r3 = [x + bar_width for x in r2]
# r4 = [x + bar_width for x in r3]
# r5 = [x + bar_width for x in r4]

# # create bar plot
# plt.bar(r1, accuracy_LR, color='b', width=bar_width, edgecolor='grey', label='LR')
# plt.bar(r2, accuracy_DTC, color='r', width=bar_width, edgecolor='grey', label='DTC')
# plt.bar(r3, accuracy_RFC, color='g', width=bar_width, edgecolor='grey', label='RFC')
# plt.bar(r4, accuracy_NB, color='y', width=bar_width, edgecolor='grey', label='NB')
# plt.bar(r5, accuracy_KNN, color='purple', width=bar_width, edgecolor='grey', label='KNN')

# # add labels and title
# plt.ylabel('Accuracy')
# plt.xlabel('Dataset Split')
# plt.title('Comparison of Different Models')
# plt.xticks([r + bar_width for r in range(len(dataset_split))], ['0.2', '0.3', '0.35'])

# # create legend
# plt.legend()

# # show plot
# plt.show()

###############################################################################################################
# LR for the driver gender feature

# # dataset splits
# dataset_split = [0.2, 0.3, 0.35]

# # accuracies for each algorithm 
# accuracy_LR_initial_dataset = [72.05, 72.48, 72.6]
# accuracy_LR_nc_durham = [62.08, 61.6, 61.74]
# accuracy_LR_nc_winston = [66.18, 66.3, 66.26]

# # define bar width
# bar_width = 0.15

# # create bar positions
# r1 = np.arange(len(dataset_split))
# r2 = [x + bar_width for x in r1]
# r3 = [x + bar_width for x in r2]

# # create bar plot
# plt.bar(r1, accuracy_LR_initial_dataset, color='b', width=bar_width, edgecolor='grey', label='LR Initial')
# plt.bar(r2, accuracy_LR_nc_durham, color='r', width=bar_width, edgecolor='grey', label='LR Durham')
# plt.bar(r3, accuracy_LR_nc_winston, color='g', width=bar_width, edgecolor='grey', label='LR Winston')

# # add labels and title
# plt.ylabel('Accuracy')
# plt.xlabel('Dataset Split')
# plt.title('Comparison of The LR For driver gender Across Different Datasets')
# plt.xticks([r + bar_width for r in range(len(dataset_split))], ['0.2', '0.3', '0.35'])

# # create legend
# plt.legend()

# # show plot
# plt.show()

###############################################################################################################
# DTC for the age group feature

# dataset splits
dataset_split = [0.2, 0.3, 0.35]

# accuracies for each algorithm 
accuracy_LR_initial_dataset = [38, 37.57, 37.56]
accuracy_LR_nc_durham = [35.48, 35.77, 35.59]
accuracy_LR_nc_winston = [36.1, 36.82, 36.94]

# define bar width
bar_width = 0.15

# create bar positions
r1 = np.arange(len(dataset_split))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# create bar plot
plt.bar(r1, accuracy_LR_initial_dataset, color='b', width=bar_width, edgecolor='grey', label='LR Initial')
plt.bar(r2, accuracy_LR_nc_durham, color='r', width=bar_width, edgecolor='grey', label='LR Durham')
plt.bar(r3, accuracy_LR_nc_winston, color='g', width=bar_width, edgecolor='grey', label='LR Winston')

# add labels and title
plt.ylabel('Accuracy')
plt.xlabel('Dataset Split')
plt.title('Comparison of the DTC for age group Across Different Datasets')
plt.xticks([r + bar_width for r in range(len(dataset_split))], ['0.2', '0.3', '0.35'])

# create legend
plt.legend()

# show plot
plt.show()