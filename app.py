import streamlit as st
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Τίτλος της εφαρμογής
st.title("Web-Based Data Mining Application")

# Πλευρική μπάρα για πλοήγηση
st.sidebar.title("Πλοήγηση")
page = st.sidebar.selectbox(
    "Επιλέξτε σελίδα",
    [
        "Φόρτωση Δεδομένων",
        "Visualization",
        "Feature Selection",
        "Classification",
        "Info",
    ],
)

# Αρχικοποίηση μεταβλητών
df = None

# Σελίδα Φόρτωσης Δεδομένων
if page == "Φόρτωση Δεδομένων":
    st.header("Φόρτωση Δεδομένων")

    # Προσθήκη κουμπιού για χρήση του ενσωματωμένου Iris dataset
    use_iris = st.checkbox("Χρήση του ενσωματωμένου Iris dataset")

    if use_iris:
        from sklearn.datasets import load_iris

        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        df = X.copy()
        df["species"] = y
        st.write("### Δεδομένα Iris dataset:")
        st.write(df.head())
        st.write(f"Διαστάσεις του DataFrame: {df.shape}")
        st.session_state["df"] = df
    else:
        # Εργαλείο για φόρτωση αρχείων
        uploaded_file = st.file_uploader(
            "Επιλέξτε ένα αρχείο", type=["csv", "xlsx", "tsv"]
        )

        # Φόρτωση δεδομένων και εμφάνιση σε πίνακα
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".tsv"):
                    df = pd.read_csv(uploaded_file, delimiter="\t")

                # Εμφάνιση των διαστάσεων του DataFrame
                st.write(f"Διαστάσεις του αρχικού DataFrame: {df.shape}")

                # Προβολή των πρώτων 5 γραμμών
                st.write("### Προεπισκόπηση Δεδομένων:")
                st.write(df.head())

                # Αποθήκευση του DataFrame στο session state
                st.session_state["df"] = df

            except Exception as e:
                st.error(f"Σφάλμα κατά τη φόρτωση του αρχείου: {e}")
        else:
            st.info(
                "Παρακαλώ επιλέξτε ένα αρχείο για φόρτωση ή χρησιμοποιήστε το ενσωματωμένο Iris dataset."
            )

# Σελίδα Visualization
elif page == "Visualization":
    st.header("Οπτικοποίηση Δεδομένων")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # Προεπεξεργασία δεδομένων
        df_processed = df.copy()

        # Διαχωρισμός χαρακτηριστικών και ετικέτας
        X = df_processed.iloc[:, :-1]  # Όλες οι στήλες εκτός της τελευταίας
        y = df_processed.iloc[:, -1]  # Τελευταία στήλη (ετικέτα)

        # Επιλογή Κανονικοποίησης
        st.sidebar.subheader("Επιλογές Προεπεξεργασίας")
        scaling_option = st.sidebar.selectbox(
            "Επιλέξτε Μέθοδο Κανονικοποίησης",
            ["None", "Standardization", "Normalization"],
        )

        # Εντοπισμός κατηγορικών στηλών
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns

        # Κωδικοποίηση κατηγορικών στηλών
        if len(categorical_columns) > 0:
            X = pd.get_dummies(X, columns=categorical_columns)

        # Μετατροπή όλων των στηλών σε αριθμητικές
        X = X.apply(pd.to_numeric, errors="coerce")

        # Συμπλήρωση των NaN τιμών
        if X.isnull().values.any():
            st.warning("Συμπλήρωση των NaN τιμών με τον μέσο όρο της κάθε στήλης.")
            X = X.fillna(X.mean())

        # Κανονικοποίηση ή Τυποποίηση των δεδομένων
        if scaling_option == "Standardization":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = pd.DataFrame(X)
        elif scaling_option == "Normalization":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            X = pd.DataFrame(X)

        # Κωδικοποίηση της ετικέτας αν είναι κατηγορική
        if y.dtype == "object" or y.dtype.name == "category":
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.fillna(y.mode()[0])

        # Έλεγχος αν το X είναι κενό
        if X.shape[0] == 0:
            st.error(
                "Μετά την προεπεξεργασία, τα χαρακτηριστικά είναι κενά. Ελέγξτε τα δεδομένα σας."
            )
            st.stop()

        # Αποθήκευση των X και y στο session state
        st.session_state["X"] = X
        st.session_state["y"] = y

        # Επιλογή για 2D ή 3D προβολή
        plot_type = st.selectbox("Επιλέξτε τύπο γραφήματος", ["2D", "3D"])

        # Επιλογή αλγορίθμου για μείωση διάστασης
        algo = st.selectbox(
            "Επιλέξτε αλγόριθμο μείωσης διάστασης", ["PCA", "UMAP"]
        )

        if algo == "PCA":
            reducer = PCA(n_components=3)
        else:
            reducer = umap.UMAP(n_components=3)

        # Εφαρμογή μείωσης διάστασης
        X_reduced = reducer.fit_transform(X)

        # 2D Plot
        if plot_type == "2D":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="viridis", ax=ax
            )
            ax.set_title(f"2D {algo} Visualization")
            st.pyplot(fig)
            plt.clf()

        # 3D Plot
        elif plot_type == "3D":
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                X_reduced[:, 2],
                c=y,
                cmap="viridis",
            )
            ax.set_title(f"3D {algo} Visualization")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.legend(*scatter.legend_elements(), title="Classes")
            st.pyplot(fig)
            plt.clf()

        # EDA - Exploratory Data Analysis
        st.subheader("Exploratory Data Analysis")

        # Pairplot
        st.write("### Pairplot")
        df_pairplot = pd.DataFrame(X.copy())
        df_pairplot["label"] = y  # Προσθήκη της ετικέτας στο DataFrame
        pairplot_fig = sns.pairplot(df_pairplot, hue="label").fig
        st.pyplot(pairplot_fig)
        plt.clf()

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(X, pd.DataFrame):
            sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax)
        else:
            sns.heatmap(pd.DataFrame(X).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.clf()
    else:
        st.error(
            "Παρακαλώ φορτώστε ένα dataset πρώτα στη σελίδα 'Φόρτωση Δεδομένων'."
        )

# Σελίδα Feature Selection
# Σελίδα Feature Selection
elif page == "Feature Selection":
    st.header("Επιλογή Χαρακτηριστικών")

    if "X" in st.session_state and "y" in st.session_state:
        X = st.session_state["X"]
        y = st.session_state["y"]

        # Υπολογισμός του αριθμού των χαρακτηριστικών
        num_features = X.shape[1]

        # Επιλογή αριθμού χαρακτηριστικών - μέγιστη τιμή είναι τα διαθέσιμα χαρακτηριστικά
        k = st.slider(f"Επιλέξτε τον αριθμό των χαρακτηριστικών (μέγιστο {num_features})", 1, num_features, min(5, num_features))

        # Επιλογή μεθόδου επιλογής χαρακτηριστικών
        method = st.selectbox(
            "Επιλέξτε μέθοδο επιλογής χαρακτηριστικών", ["Chi-Squared", "ANOVA F-test"]
        )

        # Έλεγχος για αρνητικές τιμές όταν επιλέγεται η μέθοδος Chi-Squared
        if method == "Chi-Squared" and np.any(X < 0):
            st.error("Η μέθοδος Chi-Squared δεν μπορεί να εφαρμοστεί σε δεδομένα με αρνητικές τιμές. Παρακαλώ επιλέξτε 'Normalization' στην προεπεξεργασία ή χρησιμοποιήστε τη μέθοδο 'ANOVA F-test'.")
            st.stop()

        # Επιλογή χαρακτηριστικών
        if method == "Chi-Squared":
            selector = SelectKBest(score_func=chi2, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)

        # Εφαρμογή του selector
        X_new = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)

        # Εμφάνιση των επιλεγμένων χαρακτηριστικών
        if isinstance(X, pd.DataFrame):
            selected_feature_names = X.columns[selected_features]
        else:
            selected_feature_names = [f"Feature {i}" for i in selected_features]

        st.write("### Επιλεγμένα Χαρακτηριστικά:")
        st.write(selected_feature_names.tolist())

        # Αποθήκευση του νέου dataset με τα επιλεγμένα χαρακτηριστικά
        st.session_state["X_selected"] = X_new
    else:
        st.error(
            "Παρακαλώ εκτελέστε πρώτα την οπτικοποίηση δεδομένων στη σελίδα 'Visualization'."
        )


# Σελίδα Classification
elif page == "Classification":
    st.header("Κατηγοριοποίηση Δεδομένων")

    if "X_selected" in st.session_state and "y" in st.session_state:
        X_selected = st.session_state["X_selected"]
        y = st.session_state["y"]

        # Υπολογισμός του αριθμού των δειγμάτων
        n_samples = X_selected.shape[0]

        # Εύρεση του ελάχιστου αριθμού δειγμάτων ανά κλάση
        from collections import Counter
        class_counts = Counter(y)
        # Μετατροπή των κλειδιών σε int
        class_counts = {int(k): v for k, v in class_counts.items()}
        min_class_count = min(class_counts.values())

        # Εμφάνιση της κατανομής των κλάσεων
        st.write("Κατανομή των κλάσεων:", class_counts)

        # Διαχωρισμός δεδομένων σε training και testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42, stratify=y
        )

        # Υπολογισμός του αριθμού των δειγμάτων στο training set
        n_samples_train = X_train.shape[0]

        # Επιλογή αλγορίθμου
        classifier_name = st.selectbox(
            "Επιλέξτε Αλγόριθμο Κατηγοριοποίησης",
            ["KNN", "Decision Tree", "Random Forest", "SVM"],
        )

        # Ρυθμίσεις παραμέτρων ανά αλγόριθμο
        if classifier_name == "KNN":
            st.subheader("K-Nearest Neighbors")

            # Ορισμός του μέγιστου n_neighbors
            max_k = min(15, n_samples_train)

            n_neighbors = st.slider("Επιλέξτε το K για τον KNN", 1, max_k, min(5, max_k))
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            # Υπερπαραμετροποίηση με GridSearchCV
            param_grid = {"n_neighbors": np.arange(1, max_k + 1)}
        elif classifier_name == "Decision Tree":
            st.subheader("Decision Tree")
            max_depth = st.slider("Μέγιστο βάθος δέντρου", 1, 20, 5)
            classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            param_grid = {"max_depth": np.arange(1, 21)}
        elif classifier_name == "Random Forest":
            st.subheader("Random Forest")
            n_estimators = st.slider("Αριθμός δέντρων", 10, 100, 50, step=10)
            classifier = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42
            )
            param_grid = {"n_estimators": np.arange(10, 110, 10)}
        elif classifier_name == "SVM":
            st.subheader("Support Vector Machine")
            C = st.slider("Παράμετρος C", 0.01, 10.0, 1.0)
            classifier = SVC(C=C, probability=True, random_state=42)
            param_grid = {"C": np.linspace(0.01, 10.0, 10)}

        # Επιλογή αριθμού folds για cross-validation
        max_cv_folds = min(10, n_samples, min_class_count)
        cv_folds = st.slider(
            "Αριθμός Folds για Cross-Validation",
            min_value=2,
            max_value=max_cv_folds,
            value=min(5, max_cv_folds),
        )

        # Έλεγχος αν ο αριθμός των folds είναι έγκυρος
        if cv_folds > min_class_count:
            st.error(
                f"Ο αριθμός των folds ({cv_folds}) δεν μπορεί να είναι μεγαλύτερος από τον αριθμό των δειγμάτων στην μικρότερη κλάση ({min_class_count})."
            )
            st.stop()

        # Υπερπαραμετροποίηση με GridSearchCV
        perform_grid_search = st.checkbox("Εκτέλεση Υπερπαραμετροποίησης (Grid Search)")

        if perform_grid_search:
            st.write("Εκτέλεση Grid Search για βέλτιστες παραμέτρους...")
            grid_search = GridSearchCV(
                classifier, param_grid, cv=cv_folds, scoring="accuracy"
            )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            st.write(f"**Βέλτιστες Παράμετροι:** {best_params}")
            st.write(f"**Καλύτερο Accuracy:** {grid_search.best_score_:.2f}")
            classifier = grid_search.best_estimator_

        # Εκπαίδευση του αλγορίθμου
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Μετρικές αξιολόγησης
        st.write(f"**Αποτελέσματα {classifier_name}:**")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"F1-Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
        if len(np.unique(y)) == 2:
            y_proba = classifier.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            st.write(f"ROC-AUC: {roc_auc:.2f}")
        else:
            st.write("ROC-AUC δεν είναι διαθέσιμο για πολυκατηγορική κατηγοριοποίηση.")

        # Εμφάνιση Συγχυτικού Πίνακα
        st.write("### Συγχυτικός Πίνακας")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Προβλεπόμενες Ετικέτες")
        ax.set_ylabel("Πραγματικές Ετικέτες")
        st.pyplot(fig)
        plt.clf()

        # Cross-Validation Scores
        st.write(f"**Αποτελέσματα {classifier_name} με Cross-Validation:**")
        scores = cross_val_score(
            classifier, X_selected, y, cv=cv_folds, scoring="accuracy"
        )
        st.write(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

        f1_scores = cross_val_score(
            classifier, X_selected, y, cv=cv_folds, scoring="f1_macro"
        )
        st.write(f"F1-Score: {f1_scores.mean():.2f} (+/- {f1_scores.std():.2f})")

        if len(np.unique(y)) == 2:
            roc_auc_scores = cross_val_score(
                classifier, X_selected, y, cv=cv_folds, scoring="roc_auc"
            )
            st.write(
                f"ROC-AUC: {roc_auc_scores.mean():.2f} (+/- {roc_auc_scores.std():.2f})"
            )
        else:
            st.write("ROC-AUC δεν είναι διαθέσιμο για πολυκατηγορική κατηγοριοποίηση.")

    else:
        st.error("Παρακαλώ εκτελέστε πρώτα το Feature Selection.")

# Σελίδα Info
elif page == "Info":
    st.header("Πληροφορίες Εφαρμογής")
    st.write(
        """
    Αυτή η εφαρμογή αναπτύχθηκε για την εξόρυξη και ανάλυση δεδομένων με χρήση του Streamlit.

    **Ομάδα Ανάπτυξης:**
    - Μέλος 1: Ανάπτυξη της φόρτωσης δεδομένων και των οπτικοποιήσεων
    - Μέλος 2: Υλοποίηση του Feature Selection και της Κατηγοριοποίησης
    - Μέλος 3: Ενοποίηση κώδικα, δοκιμές και συγγραφή της τεκμηρίωσης

    **Συγκεκριμένα Tasks:**
    - *Μέλος 1*: Φόρτωση και προβολή δεδομένων, οπτικοποιήσεις με PCA και UMAP
    - *Μέλος 2*: Υλοποίηση του Feature Selection και εκπαίδευση των αλγορίθμων κατηγοριοποίησης
    - *Μέλος 3*: Ενσωμάτωση του Docker, δημιουργία του GitHub repository και συγγραφή της αναφοράς

    **Οδηγίες Χρήσης:**
    1. Μεταβείτε στη σελίδα "Φόρτωση Δεδομένων" για να φορτώσετε το dataset σας.
    2. Στη σελίδα "Visualization" μπορείτε να δείτε οπτικοποιήσεις των δεδομένων.
    3. Στη σελίδα "Feature Selection" μπορείτε να επιλέξετε τα σημαντικότερα χαρακτηριστικά.
    4. Στη σελίδα "Classification" μπορείτε να εκπαιδεύσετε και να αξιολογήσετε αλγορίθμους κατηγοριοποίησης.
    """
    )

