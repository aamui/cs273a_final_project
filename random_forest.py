import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import argparse
import os
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser(description="Training random forest")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--pca_components", type=int, default=100, help="Number of PCA components")
    parser.add_argument('--n_estimators', type=int, default = 100, help = 'Number of trees in forest')
    parser.add_argument('--criterion', type = str, default = 'gini', help = 'Split criterion')
    parser.add_argument('--max_depth', type = int, default = 15, help = 'Tree max depth')
    parser.add_argument('--min_samples_leaf', type=int, default = 5, help = 'Min samples per leaf')
    args = parser.parse_args()

    # loading data
    transform = transforms.ToTensor()
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform = transform)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # reshape training
    X_train = []
    y_train = []
    for img, label in train_loader:
        X_train.append(img.view(img.size(0), -1).numpy())
        y_train.append(label.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # reshape testing
    X_test = []
    y_test = []
    for img, label in test_loader:
        X_test.append(img.view(img.size(0), -1).numpy())
        y_test.append(label.numpy())
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # pca projection
    print('Doing PCA')
    pca = PCA(n_components = args.pca_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print('Fitting random forest')
    clf = RandomForestClassifier(n_estimators = args.n_estimators, criterion = args.criterion, max_depth = args.max_depth)
    clf.fit(X_train_pca, y_train)

    rf_train_acc = clf.score(X_train_pca, y_train)
    rf_train_err = 1. - rf_train_acc
    print(f'Random forest train error on {args.pca_components} PCA components: {rf_train_err:.3f}')

    rf_test_acc = clf.score(X_test_pca, y_test)
    rf_test_err = 1. - rf_test_acc
    print(f'Random forest test error on {args.pca_components} PCA components: {rf_test_err:.3f}')
    print('Args:')
    print(vars(args))

if __name__ == '__main__':
    main()
