import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import time

def main():
    parser = argparse.ArgumentParser(description="Training random forest")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in forest')
    parser.add_argument('--criterion', type=str, default='gini', help='Split criterion')
    parser.add_argument('--max_depth', type=int, default=20, help='Tree max depth')
    parser.add_argument('--min_samples_leaf', type=int, default=5, help='Min samples per leaf')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--use_subset', action='store_true', help='Use subset for faster testing')
    parser.add_argument('--subset_size', type=int, default=5000, help='Size of subset if enabled')
    args = parser.parse_args()
    
    # loading data
    transform = transforms.ToTensor()
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
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
    
    if args.use_subset:
        indices = np.random.choice(len(X_train), args.subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Using subset of {args.subset_size} samples for CV")
    
    # reshape testing
    X_test = []
    y_test = []
    for img, label in test_loader:
        X_test.append(img.view(img.size(0), -1).numpy())
        y_test.append(label.numpy())
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    train_accuracy_list = []
    test_accuracy_list = []
    cv_accuracy_list = []
    cv_std_list = []
    depths_list = []
    
    # k fold CV
    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    print(f'Fitting random forests with {args.cv_folds}-Fold Cross-Validation')
    for depth in range(5, args.max_depth + 1, 5):
        print(f'\nMax depth: {depth}')
        start_time = time.time()

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # CV
        cv_scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f'CV Accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})')
        
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f'Train Accuracy: {train_acc:.3f} (Error: {1-train_acc:.3f})')
        print(f'Test Accuracy: {test_acc:.3f} (Error: {1-test_acc:.3f})')
        print(f'Time: {time.time() - start_time:.2f}s')
        
        depths_list.append(depth)
        train_accuracy_list.append(train_acc)
        test_accuracy_list.append(test_acc)
        cv_accuracy_list.append(cv_mean)
        cv_std_list.append(cv_std)
    
    os.makedirs('results', exist_ok=True)
    file_save_path = os.path.join('results', 'rf_cv_results.npz')
    np.savez(file_save_path,
             depths=np.array(depths_list),
             train_acc=np.array(train_accuracy_list),
             test_acc=np.array(test_accuracy_list),
             cv_acc=np.array(cv_accuracy_list),
             cv_std=np.array(cv_std_list))
    
    print(f'\nResults saved to {file_save_path}')

if __name__ == '__main__':
    main()