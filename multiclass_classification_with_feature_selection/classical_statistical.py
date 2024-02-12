import pandas as pd
import matplotlib.pyplot as plt


def reassemble(col_names_l, data):
    """
    helps to deal with the fit_transform that loses feature names
    :param col_names_l: a list of the column names
    :param data: the numpy array returned
    :return: the reassembled data frame for easy filtering
    """
    return pd.DataFrame(data, columns=col_names_l)


# create a universal function
def test_feature_selection(data_dict):
    """
    :param data_dict: a dictionary of data: {name: (X_train, y_train, X_test, y_test)}
    :return: no return, print out confusion matrix
    """

    from sklearn.metrics import confusion_matrix
    from sklearn.svm import LinearSVC
    from sklearn.metrics import ConfusionMatrixDisplay
    est = LinearSVC(random_state=35)
    for key, value in data_dict.items():
        X_train, y_train, X_test, y_test = value
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print(key)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=est.classes_)
        disp.plot()
        plt.title(key.upper())
        plt.xticks(rotation=45, ha='right')
        plt.show()


def main():
    data_train = pd.read_csv('data/training.csv')
    X_train = data_train.drop(columns=['class'])
    y_train = data_train['class']

    data_test = pd.read_csv('data/testing.csv')
    X_test = data_test.drop(columns=['class'])
    y_test = data_test['class']

    # below mostly follows the procedure from sklearn
    # https://scikit-learn.org/stable/modules/feature_selection.html

    # Remove low variance features
    from sklearn.feature_selection import VarianceThreshold
    threshold = .8 * (1 - .8)
    # threshold = 0
    sel = VarianceThreshold(threshold)
    X_train_1 = sel.fit_transform(X_train)
    X_train_1 = reassemble(col_names_l=list(sel.get_feature_names_out()), data=X_train_1)

    # transform X_test too
    X_test_1 = X_test[list(sel.get_feature_names_out())]

    # Univariate Feature Selection
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    sel = SelectKBest(f_classif, k=50)
    X_train_2 = sel.fit_transform(X_train_1, y_train)
    X_train_2 = reassemble(col_names_l=list(sel.get_feature_names_out()), data=X_train_2)
    X_test_2 = X_test_1[list(sel.get_feature_names_out())]

    data_dict = {'no feature selection': (X_train, y_train, X_test, y_test),
                 'only remove low variance features': (X_train_1, y_train, X_test_1, y_test),
                 'feature selection with ANOVA': (X_train_2, y_train, X_test_2, y_test)}

    test_feature_selection(data_dict)


if __name__ == '__main__':
    main()
