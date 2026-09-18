"""Contrato de entrada del endpoint de prediccion de genero musical.

El cliente puede enviar su perfil ya desglosado (`tipo_correo` + `proveedor_correo`)
o simplemente su `email`, del cual ambos campos se derivan con la misma logica
que usa la vista `public.vw_cliente_genero`. Asi el API y el entrenamiento
comparten exactamente el mismo criterio de clasificacion.
"""

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Estas listas replican la clasificacion de la vista vw_cliente_genero.
PROVEEDORES_GRATUITOS = {
    "gmail", "hotmail", "yahoo", "outlook", "live", "msn", "aol", "rediff",
    "jubii", "wp", "sapo", "uol", "terra", "bol", "ig", "gmx", "yandex",
    "zoho", "protonmail", "icloud", "mail",
}
PROVEEDORES_ISP = {
    "shaw", "rogers", "surfeu", "comcast", "telus", "bell", "videotron",
    "sympatico", "verizon", "att", "cox", "charter", "orange", "wanadoo",
    "t-online", "virgin", "sky", "btinternet",
}
ALIAS_PROVEEDOR = {"yachoo": "yahoo"}  # typo presente en el dataset Chinook

_RE_GUBERNAMENTAL = re.compile(r"\.(gov|gob|mil)(\.|$)")
_RE_EDUCATIVO = re.compile(r"\.(edu|ac)(\.|$)")
_RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class TipoCorreoEnum(str, Enum):
    GRATUITO = "gratuito"
    CORPORATIVO = "corporativo"
    GUBERNAMENTAL = "gubernamental"
    ISP = "isp"
    EDUCATIVO = "educativo"


def proveedor_desde_dominio(dominio: str) -> str:
    """Extrae el proveedor (gmail, yahoo, apple...) a partir del dominio."""
    raiz = dominio.strip().lower().split(".")[0]
    return ALIAS_PROVEEDOR.get(raiz, raiz)


def tipo_desde_dominio(dominio: str) -> TipoCorreoEnum:
    """Clasifica un dominio de correo igual que lo hace la vista SQL."""
    dominio = dominio.strip().lower()
    if _RE_GUBERNAMENTAL.search(dominio):
        return TipoCorreoEnum.GUBERNAMENTAL
    if _RE_EDUCATIVO.search(dominio):
        return TipoCorreoEnum.EDUCATIVO

    proveedor = proveedor_desde_dominio(dominio)
    if proveedor in PROVEEDORES_GRATUITOS:
        return TipoCorreoEnum.GRATUITO
    if proveedor in PROVEEDORES_ISP:
        return TipoCorreoEnum.ISP
    return TipoCorreoEnum.CORPORATIVO


class PredictorRequest(BaseModel):
    """Perfil del cliente que entra a la tienda."""

    pais: str = Field(..., min_length=1, max_length=80, description="Pais de origen del cliente")
    ciudad: str = Field(..., min_length=1, max_length=80, description="Ciudad de origen del cliente")
    tipo_correo: Optional[TipoCorreoEnum] = Field(
        None, description="Tipo de correo; se deriva de `email` si se omite"
    )
    proveedor_correo: Optional[str] = Field(
        None, max_length=60, description="Proveedor de correo; se deriva de `email` si se omite"
    )
    email: Optional[str] = Field(
        None, description="Correo del cliente; alternativa a tipo_correo + proveedor_correo"
    )
    top_n: int = Field(3, ge=1, le=10, description="Cuantos generos devolver, ordenados por probabilidad")

    @model_validator(mode="after")
    def derivar_campos_de_correo(self):
        """Completa tipo_correo y proveedor_correo desde el email cuando falten."""
        if self.email is not None:
            email = self.email.strip()
            if not _RE_EMAIL.match(email):
                raise ValueError(f"'{self.email}' no es un correo electronico valido")
            dominio = email.split("@", 1)[1]
            if self.tipo_correo is None:
                self.tipo_correo = tipo_desde_dominio(dominio)
            if self.proveedor_correo is None:
                self.proveedor_correo = proveedor_desde_dominio(dominio)

        if self.tipo_correo is None or self.proveedor_correo is None:
            raise ValueError(
                "Debe enviar `email`, o bien `tipo_correo` y `proveedor_correo` explicitamente"
            )
        return self

    @field_validator("pais", "ciudad", "proveedor_correo", mode="before")
    @classmethod
    def normalizar_texto(cls, valor: Any) -> Any:
        """Recorta espacios; el pais y la ciudad conservan su capitalizacion original."""
        if isinstance(valor, str):
            valor = valor.strip()
            if not valor:
                raise ValueError("El campo no puede estar vacio")
        return valor

    @field_validator("proveedor_correo", mode="after")
    @classmethod
    def normalizar_proveedor(cls, valor: Optional[str]) -> Optional[str]:
        if valor is None:
            return None
        return ALIAS_PROVEEDOR.get(valor.lower(), valor.lower())

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"email": "luisg@embraer.com.br", "pais": "Brazil", "ciudad": "Sao Jose dos Campos"},
                {"tipo_correo": "gratuito", "proveedor_correo": "gmail", "pais": "USA", "ciudad": "Madison"},
            ]
        }
    }
