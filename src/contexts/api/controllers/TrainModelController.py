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


def _canonizar(valor, conocidas):
    """Empareja `valor` con la categoria vista en entrenamiento, ignorando mayusculas.

    El OneHotEncoder compara cadenas exactas, asi que "madison" no coincidiria
    con "Madison" y se codificaria como todo ceros. Devuelve (valor_canonico,
    es_conocido).
    """
    if not conocidas:
        return valor, True  # sin catalogo no se puede afirmar que sea desconocido
    if valor in conocidas:
        return valor, True
    objetivo = valor.casefold()
    for candidato in conocidas:
        if candidato.casefold() == objetivo:
            return candidato, True
    return valor, False


class _ModelCache:
    """Cachea el bundle en memoria e invalida cuando cambia el mtime del archivo."""

    def __init__(self):
        self._bundle = None
        self._mtime = None
        self._lock = threading.Lock()

    def get(self):
        ruta = os.getenv("MODELO_ENTRENADO")
        if not ruta:
            print(" ❌ MODELO_ENTRENADO no esta configurado en el entorno del API", flush=True)
            raise HTTPException(
                status_code=503, detail="El servicio de prediccion no esta configurado."
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
                    # El detalle va al log; al cliente solo un mensaje generico,
                    # para no exponer rutas del sistema de archivos.
                    print(f" ❌ No se pudo cargar el modelo desde {ruta}: {error}", flush=True)
                    raise HTTPException(
                        status_code=503, detail="El modelo no se pudo cargar."
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
            features = bundle.get("features") or _FEATURES_POR_DEFECTO
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
            print(" ❌ El bundle del modelo no contiene un pipeline", flush=True)
            raise HTTPException(status_code=503, detail="El modelo no esta disponible.")

        perfil = {
            "tipo_correo": request.tipo_correo.value,
            "proveedor_correo": request.proveedor_correo,
            "pais": request.pais,
            "ciudad": request.ciudad,
        }

        # El bundle manda sobre las features: si se entreno con otro conjunto, se
        # rechaza limpiamente en vez de reventar con KeyError mas abajo.
        faltantes = [columna for columna in features if columna not in perfil]
        if faltantes:
            print(f" ❌ El modelo espera features no disponibles: {faltantes}", flush=True)
            raise HTTPException(
                status_code=503,
                detail="El modelo entrenado no es compatible con este API.",
            )

        # Se normaliza contra las categorias vistas en entrenamiento y se avisa de
        # las desconocidas: OneHotEncoder(handle_unknown="ignore") las codifica en
        # ceros, asi que la prediccion sigue siendo valida pero menos informada.
        advertencias = []
        for columna in features:
            canonico, conocido = _canonizar(perfil[columna], categorias.get(columna))
            perfil[columna] = canonico
            if not conocido:
                advertencias.append(
                    f"'{canonico}' no aparece en los datos de entrenamiento para '{columna}'"
                )

        entrada = pd.DataFrame([{columna: perfil[columna] for columna in features}])

        try:
            prediccion = pipeline.predict(entrada)[0]
            probabilidades = pipeline.predict_proba(entrada)[0]
        except Exception as error:
            print(f" ❌ Error al predecir sobre {perfil}: {error}", flush=True)
            raise HTTPException(status_code=500, detail="No se pudo generar la prediccion.")

        # Se ordena por la probabilidad cruda y se redondea solo para la salida:
        # redondear antes empataria clases y podria dejar en primer lugar un genero
        # distinto del que devuelve predict().
        ranking = sorted(
            zip((str(clase) for clase in pipeline.classes_), (float(p) for p in probabilidades)),
            key=lambda par: par[1],
            reverse=True,
        )
        top = [
            {"genero": genero, "probabilidad": round(prob, 4)}
            for genero, prob in ranking[: request.top_n]
        ]

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
