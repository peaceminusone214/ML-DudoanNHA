import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

def plot_cross_val_confusion_matrix(confusion_matrix, display_labels='', title='Confusion Matrix', cv=5):
    """
    Vẽ confusion matrix từ kết quả cross-validation
    
    Parameters:
    -----------
    confusion_matrix : array-like
        Ma trận nhầm lẫn
    display_labels : str or list
        Nhãn hiển thị cho các lớp
    title : str
        Tiêu đề của biểu đồ
    cv : int
        Số fold trong cross-validation
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title}\n{cv}-fold Cross-validation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def plot_learning_curve(estimator, X, y, scoring='accuracy', training_set_size=5, cv=5, 
                       x_min=0, x_max=None, y_min=0, y_max=1.02, 
                       title='Learning Curve', leg_loc=4):
    """
    Vẽ learning curve cho mô hình
    
    Parameters:
    -----------
    estimator : object
        Mô hình học máy
    X : array-like
        Features
    y : array-like
        Target
    scoring : str
        Metric đánh giá
    training_set_size : int
        Số điểm để vẽ trên đường cong
    cv : int
        Số fold trong cross-validation
    x_min, x_max, y_min, y_max : float
        Giới hạn trục x, y
    title : str
        Tiêu đề của biểu đồ
    leg_loc : int
        Vị trí của legend
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, training_set_size),
        scoring=scoring)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc=leg_loc)
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if x_max:
        plt.xlim(x_min, x_max)

def pred_proba_plot(y_true, y_pred_proba, bins=10):
    """
    Vẽ histogram phân phối xác suất dự đoán
    
    Parameters:
    -----------
    y_true : array-like
        Nhãn thực tế
    y_pred_proba : array-like
        Xác suất dự đoán
    bins : int
        Số bins trong histogram
    """
    plt.figure(figsize=(20, 18))
    plt.hist(y_pred_proba[y_true == 1], bins=bins, alpha=0.5, label='Positive class')
    plt.hist(y_pred_proba[y_true == 0], bins=bins, alpha=0.5, label='Negative class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)