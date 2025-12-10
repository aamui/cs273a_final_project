import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(description="Training random forest")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--use_pca', action='store_true')
    parser.add_argument('--subset_size', type=int, default=10000)
    parser.add_argument("--pca_components", type=int, default=1000, help="Number of PCA components")

    args = parser.parse_args()
    

    transform = transforms.ToTensor()
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Reshape training
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
        print(f"Using subset of {args.subset_size} samples")

    
    # Reshape testing
    X_test = []
    y_test = []
    for img, label in test_loader:
        X_test.append(img.view(img.size(0), -1).numpy())
        y_test.append(label.numpy())
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    if args.use_pca:
        print('Doing PCA')
        pca = PCA(n_components = args.pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    # Grid Search for optimal hyperparameters
    print("Running Grid Search to find optimal hyperparameters...")
    
    param_grid = {
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=3,  # 3-fold CV for speed
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print(f"Grid search completed in {time.time() - start_time:.2f}s")
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    train_acc = best_model.score(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    
    print(f"\nBest Model Performance:")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Gap: {train_acc - test_acc:.3f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    np.savez('results/grid_search_results.npz',
             best_params=grid_search.best_params_,
             best_cv_score=grid_search.best_score_,
             train_acc=train_acc,
             test_acc=test_acc)
    
    # Show all results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv('results/grid_search_full_results.csv', index=False)
    print("\nFull results saved to results/grid_search_full_results.csv")

if __name__ == '__main__':
    main()