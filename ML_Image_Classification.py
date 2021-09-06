""" Applies Machine Learning Classification models on microscopy image data"""

from sklearn.preprocessing import StandardScaler  # For standardizing data
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree   # For decision trees
from sklearn.metrics import plot_confusion_matrix, multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

class ML_Classification:
    def __init__(self, data):
        self.data = data

    def split_data(self):
        X_trn, X_tst, y_trn, y_tst = train_test_split(
            self.data.filter(regex='\d'),
            self.data.y,
            test_size=0.30,
            random_state=42)
        return X_trn, X_tst, y_trn, y_tst

    def plotPCA(self, x_data, y_data):
        # Create an instance of the PCA class
        pca = PCA()
        # # Transforms the training data ('tf' = 'transformed')
        trn_tf = pca.fit_transform(StandardScaler().fit_transform(x_data))
        # Plot the variance explained by each component
        colours = ['red', 'green', 'blue', "yellow", "brown", "black"]
        NLabels = len(set(y_data))
        plt.figure(4)
        plt.plot(pca.explained_variance_ratio_)
        plt.xlim([-5, 30])
        # Plots the projected data set on the first two principal components and colors by class
        plt.figure(5)
        sns.scatterplot(
            x=trn_tf[:, 0],
            y=trn_tf[:, 1],
            style=y_data,
            hue=y_data,
            palette=colours[:NLabels])
        print('pca score:', pca.score(StandardScaler().fit_transform(x_data)))

    def fitPCA(self, x_train, x_test, prop=0.99):
        # Create  an instance of the StandardScaler object
        scaler = StandardScaler()
        # Scale and transform the training and testing data
        scaler.fit(x_train)
        train_t = scaler.transform(x_train)
        test_t = scaler.transform(x_test)
        # Create and fit a PCA object based on proportion of data captured from training data set
        fittedPCA = PCA(prop)
        fittedPCA.fit(train_t)
        train_t = fittedPCA.transform(train_t)
        test_t = fittedPCA.transform(test_t)
        return train_t, test_t

    # Fit a Random Forrest Clasifier Model
    def fitRfc(self, x_data, y_data):
        # Perform Grid-Search for RFC model
        rfc = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid={
                'max_depth': [4, 8, 16, 20, 30],
                'n_estimators': [10, 50, 100, 200, 300],
            },
            cv=5, verbose=0, n_jobs=-1)
        # Fit model to Grid Parameters
        fitted_grid_result = rfc.fit(x_data, y_data)
        # Option to save best model
        saveModel = input('Save best model? Y/N')
        if saveModel == 'Y':
            joblib.dump(rfc.best_estimator_, 'RFC_model.pkl')
        return fitted_grid_result


    # Fit a Gradient Boosting Clasifier Model
    def fitGbc(self, x_data, y_data):
        # Perform Grid-Search for RFC model
        gsc = GridSearchCV(
            estimator=GradientBoostingClassifier(),
            param_grid={
                'max_depth': [1, 3, 5, 7, 9],
                'n_estimators': [5, 50, 250, 500],
            },
            cv=5, verbose=0, n_jobs=-1)
        # Fit model to Grid Parameters
        fitted_grid_result = gsc.fit(x_data, y_data)
        # Option to save best model
        saveModel = input('Save best model? Y/N')
        if saveModel == 'Y':
            joblib.dump(gsc.best_estimator_, 'GBC_model.pkl')
        return fitted_grid_result

    # Fit a Support Vector Classifier Model
    def fitSvc(self, x_data, y_data):
        svc = SVC()
        parameters = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10, 50]
        }

        cv = GridSearchCV(svc, parameters, cv=5)
        # Fit model to Grid Parameters
        cv.fit(x_data, y_data.ravel())
        # Option to save best model
        saveModel = input('Save best model? Y/N')
        if saveModel == 'Y':
            joblib.dump(cv.best_estimator_, 'SVC_model.pkl')
        return cv

    # Print out results for each parameter combination
    def print_results(self, results):
        print('BEST PARAMS: {}\n'.format(results.best_params_))

        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


    def print_confusion_matrix(self, confusion_matrix, axes, class_label, class_names, fontsize=14):
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title("Confusion Matrix for the class - " + class_label)


    # Plot confusion matrices for each label and obtain f1 scores
    def plot_multiple_confusion_matrices(self, mdl,
                                         y_tst, test_t):
        # Extract unique label names using a Set
        labels = list(set(self.data['y']))
        # Run model on test features data
        predicted_values = mdl.predict(test_t)
        # Create Multilabel Confusion Matrix
        con_matrix = multilabel_confusion_matrix(
            y_tst, predicted_values, labels=labels)
        # Setup an empty Dict to store f1 model scores for each label
        f_1_scores = {}
        for i, j in enumerate(labels):
            rec = con_matrix[i][1][1] / (con_matrix[i][1][0] + con_matrix[i][1][1])
            pres = con_matrix[i][1][1] / (con_matrix[i][0][1] + con_matrix[i][1][1])
            f_1 = 2 * (pres * rec) / (pres + rec)
            f_1_scores[j] = f_1

        fig, ax = plt.subplots(len(labels), 2, figsize=(12, 7))
        for axes, cfs_matrix, label in zip(ax.flatten(), con_matrix, labels):
            self.print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
        fig.tight_layout()
        plt.show()
        return f_1_scores

if __name__ == "__main__":
    main()