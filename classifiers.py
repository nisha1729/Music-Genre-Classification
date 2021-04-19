# reference: https://scikit-learn.org/0.15/auto_examples/mixture/plot_gmm_classifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html


from data_processor import *
import numpy as np
import sklearn.neighbors
from sklearn.metrics import accuracy_score
import sklearn.neighbors
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

feat_name = ['zero_crossing_rate', 'spectral_centroid', 'sprectral_rolloff', 'chroma_stft', 'tempo', 'beat']
for i in range(1, 21):
    feat_name.append(f' mfcc{i}')


def kNN():
    model = sklearn.neighbors.KNeighborsClassifier()
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)
    print('Accuracy : ', accuracy_score(y_test , y_pred)*100)

    # Select k best features
    best_feat_model = SelectKBest(score_func= f_classif, k=5).fit(X_train, y_train)
    best_feat = best_feat_model.get_support(True)

    print('Best 5 features for kNN model are: ', [feat_name[i] for i in best_feat])


def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def GaussMM():
    classifiers = dict((covar_type, GaussianMixture(n_components=4,
                        covariance_type=covar_type, verbose_interval=10))
                       for covar_type in ['spherical', 'diag', 'tied', 'full'])
    n_estimators = len(classifiers)
    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)

    for index, (name, classifier) in enumerate(classifiers.items()):

        # Initialise for supervision of labels since GMM is unsupervised
        classifier.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(4)])

        # Estimate model paramters
        classifier.fit(X_train)

        # Predict labels
        y_pred = classifier.predict(X_test)

        # Calculate accuracy
        print('(', name, ')', 'Accuracy : ', accuracy_score(y_test, y_pred) * 100)

        # Select k best features
        best_feat_model = SelectKBest(score_func=f_classif, k=5).fit(X_train, y_train)
        best_feat = best_feat_model.get_support(True)

        print('Best 5 features for kNN model are: ', [feat_name[i] for i in best_feat])

        # Visualisation
        if visualise:
            h = plt.subplot(2, 2, index + 1)
            make_ellipses(classifier, h)

            # Plot training data
            for n, color in enumerate('rgb'):
                data = X_train[y_train == n]
                plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                            label=y_train[n])

                # Plot test data with crosses
            for n, color in enumerate('rgb'):
                data = X_test[y_test == n]
                plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

            plt.xticks(())
            plt.yticks(())
            plt.title(name)

    if visualise:
        plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
        plt.show()


if __name__ == '__main__':
    visualise = False
    X_train, X_test, y_train, y_test = data_process()
    kNN()       # kNN + best feature selection
    GaussMM()   # Gaussian Mixture Model + best feat selection
