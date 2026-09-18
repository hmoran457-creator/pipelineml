"""Entrenamiento del modelo predictivo de genero musical.

Lee la vista `public.vw_cliente_genero` de la base Chinook (Supabase) y entrena
un clasificador multiclase que predice el genero musical que compraria un
cliente a partir de su perfil: tipo de correo, proveedor de correo, pais y
ciudad de origen.

El artefacto resultante es un bundle joblib con el pipeline entrenado, las
clases conocidas y las metricas de la corrida, para que el API pueda
reportarlas sin reentrenar.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["tipo_correo", "proveedor_correo", "pais", "ciudad"]
TARGET = "genero"
VIEW = "public.vw_cliente_genero"
RANDOM_STATE = 42


def _load_env():
    """Carga el .env desde el contenedor o desde el repo (desarrollo local)."""
    for candidate in ("/app/.env", str(Path(__file__).resolve().parents[3] / ".env")):
        if os.path.isfile(candidate):
            load_dotenv(candidate, override=False)
            print(f" 📄 .env cargado desde: {candidate}", flush=True)
            return True
    print(" ⚠️  No se encontro archivo .env; se usaran variables de entorno del proceso", flush=True)
    return False


def _fetch_dataset():
    """Devuelve la vista de entrenamiento como DataFrame."""
    user = os.getenv("SUPABASE_USER")
    password = os.getenv("SUPABASE_PASSWORD")
    host = os.getenv("SUPABASE_HOST")
    port = os.getenv("SUPABASE_PORT")
    dbname = os.getenv("SUPABASE_DBNAME")

    faltantes = [
        name
        for name, value in (
            ("SUPABASE_USER", user),
            ("SUPABASE_PASSWORD", password),
            ("SUPABASE_HOST", host),
            ("SUPABASE_PORT", port),
            ("SUPABASE_DBNAME", dbname),
        )
        if not value
    ]
    if faltantes:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltantes)}")

    query = f"SELECT {', '.join(FEATURES + [TARGET])} FROM {VIEW};"
    print(f" 🔌 Conectando a {host}:{port}/{dbname}", flush=True)

    with psycopg2.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        dbname=dbname,
        connect_timeout=15,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def _techo_teorico(df):
    """Accuracy maxima alcanzable: predecir el genero modal de cada perfil unico.

    Sirve de referencia honesta: si el modelo se acerca a este numero, el limite
    lo imponen las features disponibles y no el algoritmo.
    """
    conteos = df.groupby(FEATURES + [TARGET], observed=True).size()
    return float(conteos.groupby(level=FEATURES, observed=True).max().sum() / len(df))


def _build_pipeline(estimator):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocesador = ColumnTransformer(
        transformers=[("categoricas", encoder, FEATURES)],
        remainder="drop",
    )
    return Pipeline([("preprocesador", preprocesador), ("clasificador", estimator)])


def _candidatos():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
        ),
    }


class TrainModel:
    @staticmethod
    def entrenarModelo():
        _load_env()

        destino = os.getenv("MODELO_ENTRENADO")
        if not destino:
            print(" ❌ MODELO_ENTRENADO no esta definido. Abortando.", flush=True)
            return None

        try:
            df = _fetch_dataset()
        except Exception as error:
            print(f" ❌ Error al conectar o recuperar datos: {error}", flush=True)
            return None

        print(f" 📊 Filas recuperadas: {len(df)}", flush=True)
        if df.empty:
            print(" ❌ La vista no devolvio filas. Abortando entrenamiento.", flush=True)
            return None

        df = df.dropna(subset=FEATURES + [TARGET])
        for columna in FEATURES + [TARGET]:
            df[columna] = df[columna].astype(str).str.strip()

        # Un clasificador necesita al menos 2 ejemplos por clase para poder
        # estratificar el split; las clases mas raras se descartan.
        conteo = df[TARGET].value_counts()
        descartadas = conteo[conteo < 2]
        if not descartadas.empty:
            print(f" ⚠️  Generos descartados por tener <2 ejemplos: {list(descartadas.index)}", flush=True)
            df = df[df[TARGET].isin(conteo[conteo >= 2].index)]

        if df[TARGET].nunique() < 2:
            print(" ❌ Se necesita mas de un genero para entrenar. Abortando.", flush=True)
            return None

        X = df[FEATURES]
        y = df[TARGET]
        minimo_por_clase = int(y.value_counts().min())

        print(
            f" 🎯 Dataset listo: {len(df)} filas · {y.nunique()} generos · "
            f"{X['pais'].nunique()} paises · {X['ciudad'].nunique()} ciudades",
            flush=True,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Seleccion de modelo por validacion cruzada sobre el conjunto de train.
        cv = max(2, min(5, minimo_por_clase))
        resultados = {}
        for nombre, estimador in _candidatos().items():
            pipeline = _build_pipeline(estimador)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
            resultados[nombre] = float(scores.mean())
            print(f" 🔬 {nombre}: accuracy CV={scores.mean():.4f} (+/- {scores.std():.4f})", flush=True)

        mejor_nombre = max(resultados, key=resultados.get)
        print(f" 🏆 Modelo seleccionado: {mejor_nombre}", flush=True)

        pipeline = _build_pipeline(_candidatos()[mejor_nombre])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        baseline = _build_pipeline(DummyClassifier(strategy="most_frequent"))
        baseline.fit(X_train, y_train)
        baseline_accuracy = float(accuracy_score(y_test, baseline.predict(X_test)))

        # El API devuelve un ranking, asi que el top-3 es la metrica que mejor
        # refleja el uso real: acertar el genero dentro de las 3 sugerencias.
        try:
            top3 = float(
                top_k_accuracy_score(y_test, y_proba, k=3, labels=pipeline.classes_)
            )
        except Exception:
            top3 = None

        techo = _techo_teorico(df)

        metricas = {
            "modelo": mejor_nombre,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "top3_accuracy": top3,
            "techo_teorico": techo,
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "baseline_accuracy": baseline_accuracy,
            "accuracy_cv": resultados,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        print(
            f" 📈 accuracy={metricas['accuracy']:.4f} · "
            f"top3={'n/d' if top3 is None else format(top3, '.4f')} · "
            f"f1_macro={metricas['f1_macro']:.4f}",
            flush=True,
        )
        print(
            f" 📐 baseline={baseline_accuracy:.4f} < modelo={metricas['accuracy']:.4f} "
            f"<= techo={techo:.4f}",
            flush=True,
        )
        if metricas["accuracy"] <= baseline_accuracy:
            print(" ⚠️  El modelo no supera al baseline de clase mayoritaria.", flush=True)
        elif techo > baseline_accuracy:
            aprovechado = (metricas["accuracy"] - baseline_accuracy) / (techo - baseline_accuracy)
            metricas["margen_aprovechado"] = float(aprovechado)
            print(
                f" 🎯 El modelo aprovecha el {aprovechado:.1%} del margen que permiten las features.",
                flush=True,
            )

        # Reentrenamiento final sobre todos los datos disponibles.
        modelo_final = _build_pipeline(_candidatos()[mejor_nombre])
        modelo_final.fit(X, y)

        bundle = {
            "pipeline": modelo_final,
            "features": FEATURES,
            "target": TARGET,
            "classes": sorted(modelo_final.classes_.tolist()),
            "categorias_conocidas": {
                columna: sorted(X[columna].unique().tolist()) for columna in FEATURES
            },
            "metricas": metricas,
            "n_muestras": int(len(df)),
            "entrenado_en": datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
        temporal = f"{destino}.tmp"
        joblib.dump(bundle, temporal)
        os.replace(temporal, destino)  # escritura atomica: el API nunca lee un pkl a medias

        print(f" ✅ Modelo entrenado y guardado en {destino}", flush=True)
        return bundle
