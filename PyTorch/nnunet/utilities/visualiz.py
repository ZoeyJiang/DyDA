# from mlxtend.plotting import plot_decision_regions
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.svm import SVC
#
# def plot_dynamic_boundary(X, y, classifier, out, epoch):
#     fig, ax = plt.subplots()
#     X, y = datasets.make_blobs(n_samples=600, n_features=3,
#                            centers=[[2, 2, -2], [-2, -2, 2]],
#                            cluster_std=[2, 2], random_state=2)
#     fig, ax = plt.subplots()
#     plot_decision_regions(X,y, clf=classifier, ax=ax)
#
#
#     ax.set_title(str(epoch)+' iter')
#     plt.savefig(out+'/'+epoch+'.jpg')