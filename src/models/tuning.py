import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def tune_and_evaluate(
    X, y,
    model_type='logistic',
    n_trials=50,
    n_folds=5,
    test_size=0.2,
    timeout=None,
    use_gpu=False
):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    study = tune_model(
        X_train,
        y_train,
        model_type=model_type,
        n_trials=n_trials,
        n_folds=n_folds,
        timeout=timeout,
        use_gpu=use_gpu
    )
    
    pipeline = build_best_pipeline(study, model_type, use_gpu)
    pipeline.fit(X_train, y_train)
    
    train_preds = pipeline.predict_proba(X_train)[:, 1]
    test_preds = pipeline.predict_proba(X_test)[:, 1]
    
    train_score = roc_auc_score(y_train, train_preds)
    test_score = roc_auc_score(y_test, test_preds)
    
    print(f"Results:")
    print(f"Train ROC-AUC: {train_score:.4f}")
    print(f"Test ROC-AUC:  {test_score:.4f}")
    
    return {
        'study': study,
        'pipeline': pipeline,
        'best_params': study.best_params,
        'cv_score': study.best_value,
        'train_score': train_score,
        'test_score': test_score,
        'X_test': X_test,
        'y_test': y_test,
        'test_preds': test_preds
    }

def tune_model(
    X, y,
    model_type='logistic',
    n_trials=50,
    n_folds=5,
    timeout=None,
    use_gpu=False,
    verbose=True
):

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    objective = create_objective(X, y, model_type, n_folds, use_gpu)
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    
    if verbose:
        print(f"Best params: {study.best_params}")
    
    return study

def create_objective(X, y, model_type='logistic', n_folds=5, use_gpu=False):

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    def objective(trial):

        steps = []

        if model_type in ['logistic', 'random_forest', 'gradient_boosting']:
            steps = [('imputer', SimpleImputer(strategy='median'))]
        
        if model_type == 'logistic':
            steps.append(('scaler', StandardScaler()))
        
        model = _build_model(trial, model_type, use_gpu)
        steps.append(('model', model))
        
        pipeline = Pipeline(steps)
        
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=1 if use_gpu else -1)
            return scores.mean()
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    return objective

def _build_model(trial, model_type, use_gpu=False):
    
    if model_type == 'logistic':
        return LogisticRegression(
            C=trial.suggest_float('C', 1e-4, 10.0, log=True),
            solver=trial.suggest_categorical('solver', ['lbfgs', 'saga']),
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 200),
            max_depth=trial.suggest_int('max_depth', 2, 8),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            random_state=42
        )
    
    elif model_type == 'xgboost':
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0),
            'enable_categorical': True,
            'random_state': 42,
            'eval_metric': 'auc',
            'verbosity': 0
        }
        if use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        
        return XGBClassifier(**params)
    
    elif model_type == 'lightgbm':
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0),
            'random_state': 42,
            'verbose': -1
        }
        if use_gpu:
            params['device'] = 'gpu'
        
        return LGBMClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def build_best_pipeline(study, model_type, use_gpu=False):
    
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    
    if model_type in ['random_forest', 'logistic']:
        best_params['n_jobs'] = -1
        
    if model_type == 'xgboost':
        best_params['verbosity'] = 0
        best_params['eval_metric'] = 'auc'
        best_params['enable_categorical'] = True
        if use_gpu:
            best_params['tree_method'] = 'hist'
            best_params['device'] = 'cuda'
            
    if model_type == 'lightgbm':
        best_params['verbose'] = -1
        if use_gpu:
            best_params['device'] = 'gpu'

    if model_type == 'logistic':
        best_params['max_iter'] = 1000
        model = LogisticRegression(**best_params)
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**best_params)
        
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**best_params)
        
    elif model_type == 'xgboost':
        model = XGBClassifier(**best_params)
        
    elif model_type == 'lightgbm':
        model = LGBMClassifier(**best_params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    steps = []

    if model_type in ['logistic', 'random_forest', 'gradient_boosting']:
        steps.append(('imputer', SimpleImputer(strategy='median')))
    
    if model_type == 'logistic':
        steps.append(('scaler', StandardScaler()))

    steps.append(('model', model))

    return Pipeline(steps)



