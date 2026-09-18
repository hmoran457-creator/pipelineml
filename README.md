# pipelineml — Predicción de género musical (Chinook)

Pipeline de ML en contenedores que predice el **género musical** que consumiría un
cliente que entra a la tienda, a partir de su **tipo de correo electrónico**,
**país** y **ciudad** de origen.

Desplegado en una instancia EC2 (AWS, `us-east-2`) con IP elástica pública.

```
http://16.58.217.109:8000
```

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/health-check` | GET | Estado del servicio |
| `/api/model/info` | GET | Métricas y metadatos del modelo cargado |
| `/api/model` | POST | Predicción de género musical |
| `/docs` | GET | Swagger UI (auto-generado por FastAPI) |

---

## Arquitectura

```
┌──────────────────────── EC2 · 16.58.217.109 ────────────────────────┐
│                                                                     │
│   ┌────────────────────┐              ┌────────────────────┐        │
│   │  train-model       │              │  api               │        │
│   │  (cron)            │              │  (FastAPI+uvicorn) │        │
│   │                    │              │                    │        │
│   │  01:00 diario      │              │  :8000  ──────────────► público
│   │  + al arrancar     │              │                    │        │
│   └─────────┬──────────┘              └─────────▲──────────┘        │
│             │                                   │                   │
│             │      volumen compartido ./assets  │                   │
│             └────► modelo_entrenado.pkl ────────┘                   │
│                    (recarga en caliente por mtime)                   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ psycopg2
                              ▼
              Supabase · Chinook · public.vw_cliente_genero
```

El API **no** reinicia cuando el cron reentrena: detecta el cambio de `mtime`
del `.pkl` y recarga el modelo en la siguiente petición. La escritura del
artefacto es atómica (`.tmp` + `os.replace`), así que nunca se lee un archivo
a medio escribir.

---

## 1. La vista

`sql/vw_cliente_genero.sql` crea `public.vw_cliente_genero` en la base Chinook:

| Campo | Ejemplo | Rol |
|---|---|---|
| `tipo_correo` | `gratuito`, `corporativo`, `isp`, `gubernamental`, `educativo` | feature |
| `proveedor_correo` | `gmail`, `yahoo`, `apple`, `embraer` | feature (adicional) |
| `pais` | `Brazil` | feature |
| `ciudad` | `São José dos Campos` | feature |
| `genero` | `Rock` | **objetivo** |

**2,240 filas · 24 géneros · 24 países · 53 ciudades · 20 proveedores · 0 nulos.**

### Por qué el grano es "una fila por compra"

A nivel cliente el problema degenera: de los 59 clientes, **45 (76%) tienen Rock**
como género dominante y solo aparecen 4 géneros distintos. Un modelo entrenado
así aprende a responder "Rock" siempre y acierta 76% — una métrica vistosa que
no predice nada.

A nivel compra (`invoice_line`) hay 2,240 ejemplos y 24 géneros, con Rock en
37.3%. El modelo tiene margen real de aprendizaje.

### Clasificación del tipo de correo

Se deriva del dominio, y **la misma lógica vive en dos sitios que deben coincidir**:
la vista SQL y `PredictorRequest.py` (para poder aceptar un email crudo en el API).

| Regla | Resultado |
|---|---|
| dominio contiene `.gov` / `.gob` / `.mil` | `gubernamental` |
| dominio contiene `.edu` / `.ac` | `educativo` |
| proveedor ∈ {gmail, yahoo, hotmail, aol, uol, …} | `gratuito` |
| proveedor ∈ {shaw, rogers, comcast, surfeu, …} | `isp` |
| resto | `corporativo` |

---

## 2. El modelo

`src/contexts/train_model/TrainModel.py`:

1. Lee la vista por el *transaction pooler* de Supabase.
2. `OneHotEncoder(handle_unknown="ignore")` sobre las 4 features categóricas.
3. Compara **RandomForest** vs **LogisticRegression** por validación cruzada y
   elige el mejor.
4. Evalúa sobre un *hold-out* estratificado del 20%.
5. Reentrena el ganador con el 100% de los datos y guarda el bundle.

### Resultados

| Métrica | Valor |
|---|---|
| Modelo elegido | LogisticRegression |
| **Accuracy** | **0.3973** |
| **Top-3 accuracy** | **0.6741** |
| Baseline (siempre "Rock") | 0.3728 |
| **Techo teórico** | **0.4000** |
| F1 macro | 0.0478 |
| Margen aprovechado | **90.2%** |

### Lectura honesta de estos números

El accuracy de 0.397 parece bajo, y lo es — pero **0.400 es el máximo alcanzable**
con estas features. El "techo teórico" es la exactitud del mejor clasificador
posible: aquel que para cada combinación única de (tipo_correo, proveedor, país,
ciudad) responde el género más frecuente de esa combinación. Nada puede superarlo.

Con baseline 0.3728 y techo 0.4000, todo el margen disponible es de 2.7 puntos,
y el modelo captura el 90% de ellos. **El límite lo imponen los datos, no el
algoritmo**: Rock es el género dominante en 44 de las 53 ciudades, así que saber
de dónde viene un cliente casi no informa sobre qué música compra.

El `f1_macro` bajo (0.048) confirma lo mismo desde otro ángulo: el modelo acierta
las clases frecuentes e ignora las raras, que es la respuesta óptima cuando no
hay señal para distinguirlas.

Por eso el API devuelve un **ranking de probabilidades** y no una sola etiqueta:
con top-3, el género real aparece el **67%** de las veces, que sí es útil para
recomendar en una tienda. `techo_teorico` y `margen_aprovechado` se recalculan en
cada entrenamiento y se exponen en `/api/model/info`.

---

## 3. Uso del API

**Opción A — enviando el correo** (el tipo y el proveedor se derivan solos):

```bash
curl -X POST http://16.58.217.109:8000/api/model \
  -H 'Content-Type: application/json' \
  -d '{"email":"luisg@embraer.com.br","pais":"Brazil","ciudad":"São José dos Campos"}'
```

**Opción B — con los campos explícitos:**

```bash
curl -X POST http://16.58.217.109:8000/api/model \
  -H 'Content-Type: application/json' \
  -d '{"tipo_correo":"gratuito","proveedor_correo":"gmail","pais":"USA","ciudad":"Madison","top_n":5}'
```

Respuesta:

```json
{
  "status": "OK",
  "genero_predicho": "Rock",
  "confianza": 0.3746,
  "top_generos": [
    {"genero": "Rock",  "probabilidad": 0.3746},
    {"genero": "Latin", "probabilidad": 0.2829},
    {"genero": "Reggae","probabilidad": 0.0653}
  ],
  "perfil_evaluado": {"tipo_correo":"corporativo","proveedor_correo":"embraer","pais":"Brazil","ciudad":"São José dos Campos"},
  "advertencias": [],
  "modelo": {"algoritmo":"LogisticRegression","accuracy":0.3973,"baseline_accuracy":0.3728,"entrenado_en":"..."}
}
```

`advertencias` avisa cuando el perfil trae un país o ciudad que el modelo nunca
vio: la predicción sigue siendo válida (esas categorías se codifican en ceros)
pero está menos informada.

### Campos de entrada

| Campo | Tipo | Req. | Notas |
|---|---|---|---|
| `pais` | string | ✅ | |
| `ciudad` | string | ✅ | |
| `email` | string | ⚠️ | requerido si no se envían los dos siguientes |
| `tipo_correo` | enum | ⚠️ | `gratuito` `corporativo` `gubernamental` `isp` `educativo` |
| `proveedor_correo` | string | ⚠️ | |
| `top_n` | int 1–10 | ❌ | por defecto 3 |

Entradas inválidas devuelven **422** con el detalle del campo que falló.

---

## 4. Despliegue

```bash
git clone https://github.com/hmoran457-creator/pipelineml.git
cd pipelineml
cp .env-example .env      # y completar credenciales
docker compose up -d --build
```

Requiere que la vista exista en la base (`sql/vw_cliente_genero.sql`) y que el
puerto 8000 esté abierto en el security group.

| Comando | Acción |
|---|---|
| `make run-production` | Levanta en segundo plano |
| `make logs` | Sigue los logs |
| `make down` | Detiene todo |

### Variables de entorno (`.env`)

| Variable | Descripción |
|---|---|
| `MODELO_ENTRENADO` | Ruta del `.pkl` dentro del contenedor |
| `OUTPUT_LOG` / `ERROR_LOG` | Rutas de log del cron |
| `SUPABASE_USER` | `postgres.<project-ref>` |
| `SUPABASE_HOST` | Host del pooler (según región) |
| `SUPABASE_PORT` | `6543` (transaction pooler) |
| `SUPABASE_DBNAME` | `postgres` |
| `SUPABASE_PASSWORD` | Contraseña de la base |

`.env` está en `.gitignore` y excluido del build (`.dockerignore`): el contenedor
lo recibe montado, no incrustado en la imagen.

> **Nota sobre el cron:** los procesos lanzados por cron no heredan el entorno
> del contenedor, por eso `TrainModel` lee el archivo `.env` explícitamente en
> lugar de depender solo de `env_file`.

---

## Estructura

```
sql/vw_cliente_genero.sql                      la vista
src/contexts/train_model/TrainModel.py         entrenamiento
src/contexts/api/models/PredictorRequest.py    validación de entrada
src/contexts/api/controllers/TrainModelController.py   predicción
src/apps/api_app/ApiApp.py                     rutas FastAPI
src/tasks/crontab.TrainModel                   agenda de reentrenamiento
docker-compose.yml                             los dos servicios
```
