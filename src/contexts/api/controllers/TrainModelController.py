"""Endpoint de prediccion del genero musical de un cliente.

Carga el bundle entrenado por `TrainModel` y devuelve el genero mas probable
junto con el ranking completo de probabilidades. El bundle se mantiene en
memoria y se recarga solo cuando el archivo cambia en disco, de modo que el
reentrenamiento por cron surte efecto sin reiniciar el contenedor.
"""

import os
import threading

import joblib
import pandas as pd
from fastapi import HTTPException

from src.contexts.api.models import PredictorRequest

_FEATURES_POR_DEFECTO = ["tipo_correo", "proveedor_correo", "pais", "ciudad"]


class _ModelCache:
    """Cachea el bundle en memoria e invalida cuando cambia el mtime del archivo."""

    def __init__(self):
        self._bundle = None
        self._mtime = None
        self._lock = threading.Lock()

    def get(self):
        ruta = os.getenv("MODELO_ENTRENADO")
        if not ruta:
            raise HTTPException(
                status_code=503,
                detail="MODELO_ENTRENADO no esta configurado en el entorno del API",
            )
        if not os.path.isfile(ruta):
            raise HTTPException(
                status_code=503,
                detail="El modelo aun no ha sido entrenado. Ejecute el pipeline de entrenamiento.",
            )

        mtime = os.path.getmtime(ruta)
        with self._lock:
            if self._bundle is None or self._mtime != mtime:
                try:
                    self._bundle = joblib.load(ruta)
                except Exception as error:
                    raise HTTPException(
                        status_code=503, detail=f"No se pudo cargar el modelo: {error}"
                    )
                self._mtime = mtime
                print(f" 🔄 Modelo cargado desde {ruta}", flush=True)
            return self._bundle


_cache = _ModelCache()


class TrainModelController:
    def execute(self, request: PredictorRequest):
        bundle = _cache.get()

        # Compatibilidad: si el .pkl es un estimador suelto y no un bundle.
        if isinstance(bundle, dict):
            pipeline = bundle.get("pipeline")
            features = bundle.get("features", _FEATURES_POR_DEFECTO)
            categorias = bundle.get("categorias_conocidas", {})
            metricas = bundle.get("metricas", {})
            entrenado_en = bundle.get("entrenado_en")
        else:
            pipeline, features, categorias, metricas, entrenado_en = (
                bundle,
                _FEATURES_POR_DEFECTO,
                {},
                {},
                None,
            )

        if pipeline is None:
            raise HTTPException(status_code=503, detail="El bundle del modelo no contiene un pipeline")

        perfil = {
            "tipo_correo": request.tipo_correo.value,
            "proveedor_correo": request.proveedor_correo,
            "pais": request.pais,
            "ciudad": request.ciudad,
        }

        # Avisar cuando el perfil trae categorias que el modelo nunca vio:
        # OneHotEncoder(handle_unknown="ignore") las codifica en ceros, asi que
        # la prediccion sigue siendo valida pero menos informada.
        advertencias = [
            f"'{perfil[columna]}' no aparece en los datos de entrenamiento para '{columna}'"
            for columna in features
            if categorias.get(columna) and perfil.get(columna) not in categorias[columna]
        ]

        entrada = pd.DataFrame([{columna: perfil[columna] for columna in features}])

        try:
            prediccion = pipeline.predict(entrada)[0]
            probabilidades = pipeline.predict_proba(entrada)[0]
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Error al predecir: {error}")

        ranking = sorted(
            (
                {"genero": str(clase), "probabilidad": round(float(prob), 4)}
                for clase, prob in zip(pipeline.classes_, probabilidades)
            ),
            key=lambda item: item["probabilidad"],
            reverse=True,
        )
        top = ranking[: request.top_n]

        print(f" 🎵 {perfil} -> {prediccion}", flush=True)

        return {
            "status": "OK",
            "genero_predicho": str(prediccion),
            "confianza": top[0]["probabilidad"] if top else None,
            "top_generos": top,
            "perfil_evaluado": perfil,
            "advertencias": advertencias,
            "modelo": {
                "algoritmo": metricas.get("modelo"),
                "accuracy": metricas.get("accuracy"),
                "baseline_accuracy": metricas.get("baseline_accuracy"),
                "entrenado_en": entrenado_en,
            },
        }


class ModelInfoController:
    """Expone el estado y las metricas del modelo cargado (util para validar el deploy)."""

    def execute(self):
        bundle = _cache.get()
        if not isinstance(bundle, dict):
            return {"status": "OK", "detalle": "El artefacto no es un bundle con metadatos"}
        return {
            "status": "OK",
            "entrenado_en": bundle.get("entrenado_en"),
            "n_muestras": bundle.get("n_muestras"),
            "features": bundle.get("features"),
            "generos": bundle.get("classes"),
            "metricas": bundle.get("metricas"),
        }
