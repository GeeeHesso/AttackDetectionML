"""General utilities for the AttackDetectionML code."""

from os.path import join as pjoin

import pandas as pd
from numpy import arange as arg
from pandas import read_csv as rcsv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC


def load_gen_data(country_code, n_year=20, replace_small_value=False):

    gen_info = rcsv(pjoin("raw_data", "gens_info.csv"))
    gen_info = gen_info[gen_info.country == country_code]

    # DOWNLOAD GEN FIRST YEAR
    year = 2016
    index = 1
    print(f">>> importing gen data: year {year}, series {index}")
    gen = rcsv(pjoin("raw_data", f"gens_{year}_{index}.csv"))
    gen.columns = gen.columns.astype(int)
    gen = gen[gen_info.id]
    gen *= 100  # per units /!\

    # DOWNLOAD GEN OTHER YEARS
    for i in range(n_year - 1):  # DOWNLOAD AND ADD N-1 OTHERS YEARS
        year += 1
        if year > 2020:
            index += 1
            year = 2016
        if index > 4:
            break

        print(f">>> importing gen data: year {year}, series {index}")
        df = rcsv(pjoin("raw_data", f"gens_{year}_{index}.csv"))
        df.columns = df.columns.astype(int)
        df = df[gen_info.id]
        df *= 100  # per units /!\

        gen = pd.concat([gen, df], ignore_index=True)

    # Replace small gen production by 0
    if replace_small_value:  # < 300 kW (small hydro)
        gen[gen < 0.3] = 0  # Replace value by zeros

    return gen, gen_info


def load_data(country_code, n_year=20, replace_small_value=False):

    load_info = rcsv(pjoin("raw_data", "loads_info.csv"))
    load_info = load_info[load_info.country == country_code]

    # DOWNLOAD LOAD FIRST YEAR
    year = 2016
    index = 1
    print(f">>> importing load data: year {year}, series {index}")
    load = rcsv(pjoin("raw_data", f"loads_{year}_{index}.csv"))
    load.columns = load.columns.astype(int)
    load = load[load_info.id]
    load *= 100  # per units /!\

    # DOWNLOAD LOAD OTHER YEARS
    for i in range(n_year - 1):  # DOWNLOAD AND ADD N-1 OTHERS YEARS
        year += 1
        if year > 2020:
            index += 1
            year = 2016
        if index > 4:
            break

        print(f">>> importing load data: year {year}, series {index}")
        df = rcsv(pjoin("raw_data", f"loads_{year}_{index}.csv"))
        df.columns = df.columns.astype(int)
        df = df[load_info.id]
        df *= 100

        load = pd.concat([load, df], ignore_index=True)

    # Add suffix to load to avoid duplicated in injection dataset
    load.columns = load.columns.astype(str)
    load = load.add_suffix("_load")

    gen, gen_info = load_gen_data(country_code, n_year, replace_small_value)

    return load, gen, gen_info


def load_models():
    models_dict = {
        "nb": {
            "estimator": GaussianNB,
            "parameters": {
                "CH": {
                    "injection": {"var_smoothing": [1e-3, 1e-6, 1e-9, 1e-12]},
                    "generation": {"var_smoothing": [1e-3, 1e-6, 1e-9, 1e-12]},
                },
                "DE": {
                    "injection": {"var_smoothing": [1e-3, 1e-6, 1e-9, 1e-12]},
                    "generation": {"var_smoothing": [1e-3, 1e-6, 1e-9, 1e-12]},
                },
                "ES": {
                    "injection": {"var_smoothing": [1e-3, 1e-6, 1e-9, 1e-12]},
                    "generation": {"var_smoothing": [1e-3, 1e-6, 1e-9, 1e-12]},
                },
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier,
            "parameters": {
                "CH": {
                    "injection": {
                        "n_neighbors": [
                            *arg(1, 10),
                            *arg(10, 100, 10),
                            *arg(100, 200, 20),
                            *arg(200, 501, 50),
                        ]
                    },
                    "generation": {
                        "n_neighbors": [
                            *arg(1, 10),
                            *arg(10, 100, 10),
                            *arg(100, 200, 20),
                            *arg(200, 501, 50),
                        ]
                    },
                },
                "DE": {
                    "injection": {
                        "n_neighbors": [
                            *arg(1, 10),
                            *arg(10, 100, 10),
                            *arg(100, 200, 20),
                            *arg(200, 501, 50),
                        ]
                    },
                    "generation": {
                        "n_neighbors": [
                            *arg(1, 10),
                            *arg(10, 100, 10),
                            *arg(100, 200, 20),
                            *arg(200, 501, 50),
                        ]
                    },
                },
                "ES": {
                    "injection": {
                        "n_neighbors": [
                            *arg(1, 10),
                            *arg(10, 100, 10),
                            *arg(100, 200, 20),
                            *arg(200, 501, 50),
                        ]
                    },
                    "generation": {
                        "n_neighbors": [
                            *arg(1, 10),
                            *arg(10, 100, 10),
                            *arg(100, 200, 20),
                            *arg(200, 501, 50),
                        ]
                    },
                },
            },
        },
        "svc": {
            "estimator": SVC,  # training the SVC on a single generator takes ~8 hours for each hyperparameter, so we cannot take too many
            "parameters": {
                "CH": {
                    "injection": {"C": [3e2, 1e3, 3e3, 1e4, 3e4]},
                    "generation": {"C": [3e2, 1e3, 3e3, 1e4, 3e4]},
                },
                "DE": {  # For CH, best are always 1e4
                    "injection": {"C": [3e2, 1e3, 3e3, 1e4, 3e4]},
                    "generation": {"C": [3e2, 1e3, 3e3, 1e4, 3e4]},
                },
                "ES": {
                    "injection": {"C": [3e2, 1e3, 3e3, 1e4, 3e4]},
                    "generation": {"C": [3e2, 1e3, 3e3, 1e4, 3e4]},
                },
            },
        },
        "gbc": {
            "estimator": GradientBoostingClassifier,
            "parameters": {
                "CH": {
                    "injection": {
                        "n_estimators": [
                            *arg(10, 201, 20),
                            *arg(250, 500, 50),
                            *arg(500, 1001, 100),
                        ]
                    },
                    "generation": {
                        "n_estimators": [
                            *arg(10, 201, 20),
                            *arg(250, 500, 50),
                            *arg(500, 1001, 100),
                        ]
                    },
                },
                "DE": {
                    "injection": {
                        "n_estimators": [
                            *arg(10, 201, 20),
                            *arg(250, 500, 50),
                            *arg(500, 1001, 100),
                        ]
                    },
                    "generation": {
                        "n_estimators": [
                            *arg(10, 201, 20),
                            *arg(250, 500, 50),
                            *arg(500, 1001, 100),
                        ]
                    },
                },
                "ES": {
                    "injection": {
                        "n_estimators": [
                            *arg(10, 201, 20),
                            *arg(250, 500, 50),
                            *arg(500, 1001, 100),
                        ]
                    },
                    "generation": {
                        "n_estimators": [
                            *arg(10, 201, 20),
                            *arg(250, 500, 50),
                            *arg(500, 1001, 100),
                        ]
                    },
                },
            },
        },
        "rf": {
            "estimator": RandomForestClassifier,
            "parameters": {
                "CH": {
                    "injection": {"n_estimators": [*arg(2, 11, 2), *arg(20, 101, 10)]},
                    "generation": {"n_estimators": [*arg(2, 11, 2), *arg(20, 101, 10)]},
                },
                "DE": {
                    "injection": {"n_estimators": [*arg(2, 11, 2), *arg(20, 101, 10)]},
                    "generation": {"n_estimators": [*arg(2, 11, 2), *arg(20, 101, 10)]},
                },
                "ES": {
                    "injection": {"n_estimators": [*arg(2, 11, 2), *arg(20, 101, 10)]},
                    "generation": {"n_estimators": [*arg(2, 11, 2), *arg(20, 101, 10)]},
                },
            },
        },
        "mlpc": {
            "estimator": MLPClassifier,
            "parameters": {
                "CH": {
                    "injection": {
                        "hidden_layer_sizes": [
                            (200, 150, 100, 50),
                            (150, 100, 50, 10),
                            (600, 300, 100),
                            (500, 200, 50),
                            (200, 150, 50, 10),
                            (200, 50, 10),
                            (200,),
                            (200, 100),
                            (200, 50),
                            (200, 100, 50),
                            (150,),
                            (150, 100),
                            (150, 50),
                            (150, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                            (50, 10),
                        ]
                    },
                    "generation": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (30,),
                            (30, 15),
                            (30, 15, 8),
                            (15,),
                            (15, 7),
                            (15, 10, 5),
                            (10,),
                            (36, 25, 12, 5),
                            (30, 20, 10, 5),
                        ]
                    },
                },
                "DE": {
                    "injection": {
                        "hidden_layer_sizes": [
                            (200, 150, 100, 50),
                            (150, 100, 50, 10),
                            (200, 150, 50, 10),
                            (200, 50, 10),
                            (600, 300, 100),
                            (500, 200, 50),
                            (200,),
                            (200, 100),
                            (200, 50),
                            (200, 100, 50),
                            (150,),
                            (150, 100),
                            (150, 50),
                            (150, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                            (50, 10),
                        ]
                    },
                    "generation": {
                        "hidden_layer_sizes": [
                            (120,),
                            (120, 60),
                            (120, 60, 30),
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (10,),
                            (120, 80, 40, 10),
                            (100, 50, 25, 10),
                        ]
                    },
                },
                "ES": {
                    "injection": {
                        "hidden_layer_sizes": [
                            (1000, 500, 250, 50),
                            (800, 300, 100, 10),
                            (900, 450, 200, 50),
                            (500, 250, 100, 10),
                            (900, 400, 100),
                            (900, 300, 50),
                            (900,),
                            (900, 100),
                            (500,),
                            (500, 100),
                            (500, 50),
                            (500, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                            (50, 10),
                        ]
                    },
                    "generation": {
                        "hidden_layer_sizes": [
                            (80, 40, 20, 10),
                            (80, 50, 25, 10),
                            (96, 60, 30, 10),
                            (96, 50, 25, 10),
                            (96,),
                            (96, 50),
                            (96, 50, 20),
                            (80,),
                            (80, 40),
                            (80, 40, 10),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (40,),
                            (40, 10),
                            (40, 20, 10),
                            (10,),
                        ]
                    },
                },
            },
        },
        "mlpr": {
            "estimator": MLPRegressor,
            "parameters": {
                "CH": {
                    "injection_4_t": {
                        "hidden_layer_sizes": [
                            (200, 150, 50, 10),
                            (200, 50, 10),
                            (200,),
                            (200, 100),
                            (200, 50),
                            (200, 100, 50),
                            (150,),
                            (150, 100),
                            (150, 50),
                            (150, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                            (50, 10),
                        ]
                    },
                    "injection_4_hist": {
                        "hidden_layer_sizes": [
                            (700, 350, 150, 70),
                            (600, 300, 150, 70),
                            (700, 300, 100),
                            (700, 350, 50),
                            (600, 300, 100),
                            (500, 200, 50),
                            (600, 300),
                            (600, 100),
                            (200,),
                            (200, 100),
                        ]
                    },
                    "injection_24_t": {
                        "hidden_layer_sizes": [
                            (200, 150, 50, 10),
                            (200, 50, 10),
                            (200,),
                            (200, 100),
                            (200, 50),
                            (200, 100, 50),
                            (150,),
                            (150, 100),
                            (150, 50),
                            (150, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                            (50, 10),
                        ]
                    },
                    "injection_24_hist": {
                        "hidden_layer_sizes": [
                            (4000, 1000, 500, 100),
                            (500, 200, 100, 50),
                            (1000, 500, 100),
                            (2000, 200, 100),
                            (2000, 700, 150),
                            (2000, 100),
                            (700, 350, 150, 70),
                            (600, 300, 150, 70),
                            (600, 300, 100),
                            (500, 200, 50),
                            (200, 150, 50, 10),
                            (200, 50, 10),
                            (200,),
                            (200, 100),
                        ]
                    },
                    "generation_4_t": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (40,),
                            (40, 20),
                            (40, 20, 10),
                            (30,),
                            (30, 15),
                            (30, 15, 8),
                            (20,),
                            (20, 10),
                            (20, 10, 5),
                            (10,),
                            (36, 25, 12, 5),
                            (30, 20, 10, 5),
                        ]
                    },
                    "generation_4_hist": {
                        "hidden_layer_sizes": [
                            (140, 100, 50, 10),
                            (120, 70, 40, 10),
                            (140,),
                            (140, 70),
                            (140, 50),
                            (140, 90, 40),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (70,),
                            (70, 20),
                            (50,),
                            (50, 10),
                        ]
                    },
                    "generation_24_t": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (40,),
                            (40, 20),
                            (40, 20, 10),
                            (30,),
                            (30, 15),
                            (30, 15, 8),
                            (20,),
                            (20, 10),
                            (20, 10, 5),
                            (10,),
                            (50, 25, 12, 5),
                            (50, 20, 10, 5),
                        ]
                    },
                    "generation_24_hist": {
                        "hidden_layer_sizes": [
                            (700, 350, 150, 70),
                            (600, 300, 150, 70),
                            (700, 300, 100),
                            (700, 350, 50),
                            (600, 300, 100),
                            (500, 200, 50),
                            (600, 300),
                            (600, 100),
                            (200,),
                            (200, 100),
                        ]
                    },
                },
                "DE": {
                    "injection_4_t": {
                        "hidden_layer_sizes": [
                            (600, 300, 150, 10),
                            (400,),
                            (400, 100),
                            (400, 50),
                            (400, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                        ]
                    },
                    "injection_4_hist": {
                        "hidden_layer_sizes": [
                            (1000, 500, 100, 10),
                            (1000, 300, 100),
                            (1000, 350, 50),
                            (600, 300, 100),
                            (500, 100, 10),
                            (600, 300),
                            (600, 100),
                            (200,),
                            (200, 100),
                        ]
                    },
                    "injection_24_t": {
                        "hidden_layer_sizes": [
                            (600, 300, 150, 10),
                            (300, 50, 10),
                            (400,),
                            (400, 100),
                            (400, 50),
                            (400, 100, 50),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                            (50,),
                        ]
                    },
                    "injection_24_hist": {
                        "hidden_layer_sizes": [
                            (4000, 1000, 500, 100),
                            (500, 200, 100, 50),
                            (4000, 500, 100),
                            (1000, 100, 10),
                            (2000, 100),
                            (1000, 500, 50, 10),
                            (1000, 50, 10),
                            (500,),
                            (500, 100),
                        ]
                    },
                    "generation_4_t": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (20,),
                            (20, 10),
                            (100, 50, 20, 10),
                        ]
                    },
                    "generation_4_hist": {
                        "hidden_layer_sizes": [
                            (500, 100, 50, 10),
                            (500,),
                            (500, 100),
                            (500, 50),
                            (500, 200, 10),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                        ]
                    },
                    "generation_24_t": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (20,),
                            (20, 10),
                            (100, 50, 20, 10),
                        ]
                    },
                    "generation_24_hist": {
                        "hidden_layer_sizes": [
                            (1000, 600, 200, 50),
                            (1000, 300, 100),
                            (600, 300, 100),
                            (500, 100, 10),
                            (600, 300),
                            (600, 100),
                            (200,),
                            (200, 100),
                        ]
                    },
                },
                "ES": {
                    "injection_4_t": {
                        "hidden_layer_sizes": [
                            (1000, 500, 100, 10),
                            (1000, 300, 100),
                            (1000, 350, 50),
                            (600, 300, 100),
                            (500, 100, 10),
                            (600, 300),
                            (600, 100),
                            (200,),
                            (200, 100),
                        ]
                    },
                    "injection_4_hist": {
                        "hidden_layer_sizes": [
                            (4000, 1000, 500, 100),
                            (500, 200, 100, 50),
                            (4000, 500, 100),
                            (1000, 100, 10),
                            (2000, 100),
                            (1000, 500, 50, 10),
                            (1000, 50, 10),
                            (500,),
                            (500, 100),
                        ]
                    },
                    "injection_24_t": {
                        "hidden_layer_sizes": [
                            (1000, 500, 100, 10),
                            (600, 300, 150, 10),
                            (1000, 300, 100),
                            (1000, 350, 50),
                            (600, 300, 100),
                            (500, 100, 10),
                            (400,),
                            (400, 100),
                            (400, 50),
                            (400, 100, 50),
                            (200,),
                            (200, 100),
                            (100,),
                            (100, 50),
                        ]
                    },
                    "injection_24_hist": {
                        "hidden_layer_sizes": [
                            (4000, 1000, 500, 100),
                            (500, 200, 100, 50),
                            (10000, 2000, 500, 100),
                            (4000, 500, 100),
                            (10000, 2000, 100),
                            (1000, 100, 10),
                            (2000, 100),
                            (1000, 500, 50, 10),
                            (1000, 50, 10),
                            (500,),
                            (500, 100),
                        ]
                    },
                    "generation_4_t": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (20,),
                            (20, 10),
                            (100, 50, 20, 10),
                        ]
                    },
                    "generation_4_hist": {
                        "hidden_layer_sizes": [
                            (350, 100, 50, 10),
                            (350,),
                            (350, 100),
                            (350, 50),
                            (350, 200, 10),
                            (100,),
                            (100, 50),
                            (100, 50, 10),
                        ]
                    },
                    "generation_24_t": {
                        "hidden_layer_sizes": [
                            (100,),
                            (100, 50),
                            (100, 50, 20),
                            (50,),
                            (50, 25),
                            (50, 25, 10),
                            (20,),
                            (20, 10),
                            (100, 50, 20, 10),
                        ]
                    },
                    "generation_24_hist": {
                        "hidden_layer_sizes": [
                            (2000, 600, 200, 50),
                            (2000, 1000, 100),
                            (1500, 500, 100),
                            (1500, 100, 10),
                            (500, 200),
                            (500, 100),
                            (500,),
                            (1000, 100),
                            (100,),
                        ]
                    },
                },
            },
        },
    }

    return models_dict


def get_gen_names(country_code):

    gen_names = {
        "CH": {
            923: "Cavergno",
            918: "Innertkirchen",
            933: "Löbbia",
            934: "Pradella",
            173: "Riddes",
            932: "Rothenbrunnen",
            924: "Sedrun",
            931: "Sils",
            915: "Stalden",
            927: "Tavanasa",
        },
        "DE": {
            253: "Heilbronn (coal)",
            235: "Karlsruhe W1 (coal)",
            255: "Kupferzell (coal)",
            969: "Großgartach (coal)",
            234: "Karlsruhe W2 (coal)",
            940: "Mitte (gas)",
            301: "Charlottenburg (gas)",
            957: "Dörpen (gas)",
            299: "Thyrow 1 (gas)",
            300: "Thyrow 2 (gas)",
            208: "Landesbergen (gas)",
        },
        "ES": {
            1010: "Aguayo (hydro)",
            997: "Belesar (hydro)",
            81: "Riba Roja (hydro)",
            82: "Mequinenza (hydro)",
            905: "Sallente (hydro)",
            142: "Guillena (hydro)",
            993: "S. Pedro (hydro)",
            152: "G. y Galan (hydro)",
            956: "Isona (hydro)",
            998: "P.G. Rodriguez (gas)",
            1004: "Compostilla (coal)",
            1005: "La Robla (coal)",
        },
    }
    return gen_names[country_code]


def get_p_nom(country_code):

    gen_p_nom = {
        "CH": {
            923: 104.0,
            918: 240.0,
            933: 95.0,
            934: 288.0,
            173: 225.0,
            932: 127.0,
            924: 147.0,
            931: 247.0,
            915: 180.0,
            927: 176.0,
        },
        "DE": {
            253: 118.0,
            235: 501.0,
            255: 721.0,
            969: 1441.0,
            234: 876.0,
            940: 444.0,
            301: 211.0,
            957: 130.0,
            299: 296.0,
            300: 432.0,
            208: 491.0,
        },
        "ES": {
            1010: 361.0,
            997: 331.01,
            81: 261.28,
            82: 322.36,
            905: 219.14,
            142: 207.99,
            993: 181.02,
            152: 110.4,
            956: 107.40,
            998: 855.67,
            1004: 1005.14,
            1005: 619.06,
        },
    }

    return gen_p_nom[country_code]
