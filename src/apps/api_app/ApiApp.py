from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.contexts.api.controllers import HealthCheckController
from src.contexts.api.controllers import TrainModelController
from src.contexts.api.controllers import ModelInfoController


class ApiApp:
    def __init__(self):
        self.app = FastAPI(
            title="API Prediccion de Genero Musical",
            description="Predice el genero musical de un cliente de la tienda a partir de su tipo de correo, pais y ciudad.",
            version="1.0.0",
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.setup_routes()

    def setup_routes(self):
        self.app.add_api_route(
            "/api/health-check",
            HealthCheckController().execute, 
            methods=["GET"],
        )
        
       
        self.app.add_api_route(
            "/api/model",
            TrainModelController().execute,
            methods=["POST"],
        )

        self.app.add_api_route(
            "/api/model/info",
            ModelInfoController().execute,
            methods=["GET"],
        )

    def start(self):
        print(f"\n 🚀 init ApiApp")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)
