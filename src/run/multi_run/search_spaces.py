from src.configs.constants import DLModelNames, MLModelNames

dl_learning_rates = [
    1e-5,
    3e-5,
    1e-4,
]

freeze = [
    True,
    False,
]

dropout = [
    0.1,
    0.3,
    0.5,
]

dl_search_space_by_model_name: dict[DLModelNames, dict] = {
    DLModelNames.PLMASF_MODEL: {
        'trainer': {
            'parameters': {
                'learning_rate': {'values': dl_learning_rates + [0.0002]},
            }
        },
        'model': {
            'parameters': {
                'lstm_num_layers': {'values': [1]},
                'lstm_dropout': {'values': [0.1]},
                'lstm_hidden_size': {'values': [10, 40, 70]},
                'freeze': {
                    'values': freeze,
                },
            }
        },
    },
    DLModelNames.PLMAS_MODEL: {
        'trainer': {
            'parameters': {
                'learning_rate': {'values': dl_learning_rates + [0.0002]},
            }
        },
        'model': {
            'parameters': {
                'lstm_num_layers': {'values': [1, 2]},
                'lstm_dropout': {'values': dropout},
                'freeze': {
                    'values': freeze,
                },
            }
        },
    },
    DLModelNames.ROBERTEYE_MODEL: {
        'model': {
            'parameters': {
                'freeze': {
                    'values': freeze,
                },
                'eye_projection_dropout': {'values': dropout},
            }
        },
        'trainer': {
            'parameters': {
                'learning_rate': {'values': dl_learning_rates},
            }
        },
    },
    DLModelNames.BEYELSTM_MODEL: {
        'model': {
            'parameters': {
                'dropout_rate': {
                    'values': dropout,
                },
                'embedding_dim': {
                    'values': [4, 8],
                },
                'lstm_block_fc1_out_dim': {
                    'values': [50],
                },
                'lstm_block_fc2_out_dim': {
                    'values': [20],
                },
                'gsf_out_dim': {
                    'values': [32],
                },
                'after_cat_fc_hidden_dim': {
                    'values': [32],
                },
                'hidden_dim': {
                    'values': [64, 128],
                },
            }
        },
        'trainer': {
            'parameters': {
                'learning_rate': {'values': [0.001, 0.003, 0.01]},
            }
        },
    },
    DLModelNames.AHN_CNN_MODEL: {
        'model': {
            'parameters': {
                'hidden_dim': {'values': [40, 80]},
                'conv_kernel_size': {'values': [3]},
                'pooling_kernel_size': {'values': [2]},
                'fc_hidden_dim1': {'values': [50]},
                'fc_hidden_dim2': {'values': [20]},
                'fc_dropout': {'values': dropout},
            }
        },
        'trainer': {
            'parameters': {
                'learning_rate': {'values': [1e-3] + dl_learning_rates},
            }
        },
    },
    DLModelNames.AHN_RNN_MODEL: {
        'model': {
            'parameters': {
                'hidden_dim': {'values': [25, 50]},
                'num_lstm_layers': {'values': [1, 2, 4]},
                'fc_hidden_dim': {'values': [20]},
                'fc_dropout': {'values': dropout},
            }
        },
        'trainer': {
            'parameters': {
                'learning_rate': {'values': [1e-3] + dl_learning_rates},
            }
        },
    },
    DLModelNames.MAG_MODEL: {
        'model': {
            'parameters': {
                'mag_dropout': {'values': dropout},
                'mag_injection_index': {  # Starts from 0 and should be no more than max number of layers. If roberta-base and >13 cutoff in train.py
                    'values': [
                        0,
                        23,
                    ],
                },
                'freeze': {
                    'values': freeze,
                },
            }
        },
        'trainer': {
            'parameters': {
                'learning_rate': {'values': dl_learning_rates},
            }
        },
    },
    DLModelNames.POSTFUSION_MODEL: {
        'model': {
            'parameters': {
                'eye_projection_dropout': {'values': dropout},
                'freeze': {
                    'values': freeze,
                },
            }
        },
        'trainer': {
            'parameters': {
                'learning_rate': {'values': dl_learning_rates},
            }
        },
    },
}

ml_search_space_by_model_name: dict[MLModelNames, dict] = {
    MLModelNames.XGBOOST: {
        'model': {
            'parameters': {
                'pca_explained_variance_ratio_threshold': {'values': [0.8, 0.9, 1.0]},
                'sklearn_pipeline_param_clf__learning_rate': {
                    'values': [0.3, 0.01, 0.001]
                },
                'sklearn_pipeline_param_clf__n_estimators': {'values': [10, 100, 1000]},
                'sklearn_pipeline_param_clf__max_depth': {
                    'values': [
                        3,
                        6,
                        9,
                    ]
                },
                'sklearn_pipeline_param_clf__alpha': {'values': [0, 0.01, 0.1]},
            },
        },
    },
    MLModelNames.XGBOOST_REG: {
        'model': {
            'parameters': {
                'pca_explained_variance_ratio_threshold': {'values': [0.8, 0.9, 1.0]},
                'sklearn_pipeline_param_reg__learning_rate': {
                    'values': [0.3, 0.01, 0.001]
                },
                'sklearn_pipeline_param_reg__n_estimators': {'values': [10, 100, 1000]},
                'sklearn_pipeline_param_reg__max_depth': {
                    'values': [
                        3,
                        6,
                        9,
                    ]
                },
                'sklearn_pipeline_param_reg__alpha': {'values': [0, 0.01, 0.1]},
            },
        },
    },
    MLModelNames.LOGISTIC_REGRESSION: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_clf__C': {
                    'values': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
                },
                'sklearn_pipeline_param_clf__fit_intercept': {'values': [True]},
                'sklearn_pipeline_param_clf__penalty': {'values': ['l2', None]},
                'sklearn_pipeline_param_clf__solver': {'values': ['lbfgs']},
                'sklearn_pipeline_param_clf__random_state': {'values': [1]},
                'sklearn_pipeline_param_clf__max_iter': {'values': [1000]},
                'sklearn_pipeline_param_clf__class_weight': {'values': [None]},
                'sklearn_pipeline_param_scaler__with_mean': {'values': [True]},
                'sklearn_pipeline_param_scaler__with_std': {'values': [True]},
            },
        }
    },
    MLModelNames.LINEAR_REG: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_regressor__fit_intercept': {
                    'values': [True, False]
                },
                'sklearn_pipeline_param_scaler__with_mean': {'values': [True]},
                'sklearn_pipeline_param_scaler__with_std': {'values': [True]},
            },
        }
    },
    MLModelNames.DUMMY_CLASSIFIER: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_clf__strategy': {
                    'values': [
                        'stratified',
                        'most_frequent',
                        'prior',
                        'uniform',
                    ]
                },
            },
        }
    },
    MLModelNames.DUMMY_REGRESSOR: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_reg__strategy': {
                    'values': [
                        'mean',
                        'median',
                    ]
                },
            },
        }
    },
    MLModelNames.SVM: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_clf__C': {'values': [0.1, 1.0, 10.0, 100.0]},
                'sklearn_pipeline_param_clf__kernel': {'values': ['rbf', 'linear']},
                'sklearn_pipeline_param_clf__degree': {'values': [3]},
                'sklearn_pipeline_param_clf__gamma': {
                    'values': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001]
                },
                'sklearn_pipeline_param_clf__coef0': {'values': [0.0]},
                'sklearn_pipeline_param_clf__shrinking': {'values': [True]},
                'sklearn_pipeline_param_clf__probability': {'values': [False]},
                'sklearn_pipeline_param_clf__tol': {'values': [0.001]},
                'sklearn_pipeline_param_clf__class_weight': {
                    'values': ['balanced', None]
                },
                # scaler params
                'sklearn_pipeline_param_scaler__with_mean': {'values': [True]},
                'sklearn_pipeline_param_scaler__with_std': {'values': [True]},
            },
        },
    },
    MLModelNames.SVM_REG: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_reg__C': {'values': [0.1, 1.0, 10.0, 100.0]},
                'sklearn_pipeline_param_reg__kernel': {'values': ['rbf', 'linear']},
                'sklearn_pipeline_param_reg__degree': {'values': [3]},
                'sklearn_pipeline_param_reg__gamma': {
                    'values': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001]
                },
                'sklearn_pipeline_param_reg__coef0': {'values': [0.0]},
                'sklearn_pipeline_param_reg__shrinking': {'values': [True]},
                'sklearn_pipeline_param_reg__tol': {'values': [0.001]},
                'sklearn_pipeline_param_reg__epsilon': {'values': [0.1]},
                # scaler params
                'sklearn_pipeline_param_scaler__with_mean': {'values': [True]},
                'sklearn_pipeline_param_scaler__with_std': {'values': [True]},
            },
        },
    },
    MLModelNames.RANDOM_FOREST: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_clf__n_estimators': {'values': [10, 100, 1000]},
                'sklearn_pipeline_param_clf__max_depth': {
                    'values': [
                        3,
                        6,
                        9,
                    ]
                },
                'sklearn_pipeline_param_clf__min_samples_split': {
                    'values': [
                        2,
                        4,
                        8,
                    ]
                },
                'sklearn_pipeline_param_clf__min_samples_leaf': {
                    'values': [
                        1,
                        0.01,
                        0.02,
                    ]
                },
                'sklearn_pipeline_param_clf__max_features': {
                    'values': [
                        'sqrt',
                        'log2',
                        None,
                    ]
                },
            },
        },
    },
    MLModelNames.RANDOM_FOREST_REG: {
        'model': {
            'parameters': {
                'sklearn_pipeline_param_reg__n_estimators': {'values': [10, 100, 1000]},
                'sklearn_pipeline_param_reg__max_depth': {
                    'values': [
                        3,
                        6,
                        9,
                    ]
                },
                'sklearn_pipeline_param_reg__min_samples_split': {
                    'values': [
                        2,
                        4,
                        8,
                    ]
                },
                'sklearn_pipeline_param_reg__min_samples_leaf': {
                    'values': [
                        1,
                        0.01,
                        0.02,
                    ]
                },
                'sklearn_pipeline_param_reg__max_features': {
                    'values': [
                        'sqrt',
                        'log2',
                        None,
                    ]
                },
            },
        },
    },
}

search_space_by_model = dl_search_space_by_model_name | ml_search_space_by_model_name
