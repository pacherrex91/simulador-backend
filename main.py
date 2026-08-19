from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
import math
import os

from dotenv import load_dotenv
import numpy_financial as npf

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

load_dotenv()

logger = logging.getLogger("simulador")

app = FastAPI(
    title="Simulador de Negocios - Motor Financiero",
    version="3.4",
)

DEFAULT_ORIGINS = [
    "https://simulador-negocios.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        ",".join(DEFAULT_ORIGINS),
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=os.getenv(
        "ALLOWED_ORIGIN_REGEX",
        r"https://.*\.vercel\.app",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELOS DE ENTRADA
# ============================================================


class ItemDinamico(BaseModel):
    id: str
    nombre: str
    monto: float
    categoria: str
    vida_util: int = 0
    residual: float = 0.0


class Inversion(BaseModel):
    insumos: float = 0.0
    equipos: float = 0.0
    empaques: float = 0.0
    permisos: float = 0.0
    otros: float = 0.0


class GastosFijos(BaseModel):
    marketing: float = 0.0
    logistica: float = 0.0
    sueldo_emprendedor: float = 0.0
    otros: float = 0.0


class Ventas(BaseModel):
    pesimista: int
    base: int
    optimista: int
    crecimiento_mensual: float


class DatosSimulacion(BaseModel):
    nombre_idea: str
    sector: str = ""
    moneda: str = "S/"
    capital_disponible: float = 10000.0

    inversion_dinamica: list[ItemDinamico] = Field(default_factory=list)
    gastos_dinamicos: list[ItemDinamico] = Field(default_factory=list)

    inversion: Optional[Inversion] = None
    gastos_fijos: Optional[GastosFijos] = None

    precio_venta: float
    costo_directo: float
    ventas: Ventas

    regimen_tributario: str = "NRUS"
    inflacion_anual: float = 3.0

    financiamiento_monto: float = 0.0
    financiamiento_tasa_mensual: float = 0.0
    financiamiento_plazo: int = 12

    tasa_descuento: float = 12.0
    meses_reserva: int = 3
    estacionalidad: list[float] = Field(default_factory=lambda: [0.0] * 12)

    # Supuestos tributarios editables. Se agregan con valores por defecto
    # para mantener compatibilidad con el frontend V3.3 actual.
    precio_incluye_igv: bool = True
    costo_directo_incluye_igv: bool = True
    gastos_fijos_incluyen_igv: bool = True
    aplica_igv: bool = True

    # Referencia tributaria 2026.
    anio_tributario: int = 2026
    uit_referencia: float = 5500.0

    # Supuesto económico de recuperación del capital de trabajo al mes 36.
    porcentaje_recuperacion_capital_trabajo: float = 100.0


# ============================================================
# CONSTANTES Y UTILIDADES
# ============================================================


IGV_TASA = 0.18
VIDA_UTIL_EQUIPOS_DEFAULT = 60
RESIDUAL_EQUIPOS_DEFAULT = 0.10

NRUS_LIMITE_MENSUAL = 8000.0
NRUS_LIMITE_ANUAL = 96000.0
NRUS_ACTIVOS_FIJOS_LIMITE = 70000.0

RER_LIMITE_ANUAL = 525000.0
RER_ACTIVOS_FIJOS_LIMITE = 126000.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "motor": "financiero-consolidado",
        "version": "3.4",
    }


def _normalizar_inversion(datos: DatosSimulacion) -> list[ItemDinamico]:
    if datos.inversion_dinamica:
        return datos.inversion_dinamica

    if datos.inversion:
        inv = datos.inversion
        return [
            ItemDinamico(
                id="legacy-insumos",
                nombre="Insumos",
                monto=inv.insumos,
                categoria="Insumos",
            ),
            ItemDinamico(
                id="legacy-equipos",
                nombre="Equipos",
                monto=inv.equipos,
                categoria="Equipos",
            ),
            ItemDinamico(
                id="legacy-empaques",
                nombre="Empaques",
                monto=inv.empaques,
                categoria="Empaques",
            ),
            ItemDinamico(
                id="legacy-permisos",
                nombre="Permisos",
                monto=inv.permisos,
                categoria="Permisos",
            ),
            ItemDinamico(
                id="legacy-otros",
                nombre="Otros",
                monto=inv.otros,
                categoria="Otros",
            ),
        ]

    return []


def _normalizar_gastos(datos: DatosSimulacion) -> list[ItemDinamico]:
    if datos.gastos_dinamicos:
        return datos.gastos_dinamicos

    if datos.gastos_fijos:
        gf = datos.gastos_fijos
        return [
            ItemDinamico(
                id="legacy-marketing",
                nombre="Marketing",
                monto=gf.marketing,
                categoria="Marketing",
            ),
            ItemDinamico(
                id="legacy-logistica",
                nombre="Logística",
                monto=gf.logistica,
                categoria="Proveedores",
            ),
            ItemDinamico(
                id="legacy-sueldo",
                nombre="Sueldo Emprendedor",
                monto=gf.sueldo_emprendedor,
                categoria="Personal",
            ),
            ItemDinamico(
                id="legacy-otros",
                nombre="Otros Fijos",
                monto=gf.otros,
                categoria="Otros",
            ),
        ]

    return []


def _normalizar_regimen(regimen: str) -> str:
    valor = (regimen or "").strip().upper()

    equivalencias = {
        "NRUS": "NRUS",
        "NUEVO RUS": "NRUS",
        "NUEVO REGIMEN UNICO SIMPLIFICADO": "NRUS",
        "RER": "RER",
        "REGIMEN ESPECIAL": "RER",
        "REGIMEN ESPECIAL DE RENTA": "RER",
        "RMT": "RMT",
        "MYPE": "RMT",
        "REGIMEN MYPE TRIBUTARIO": "RMT",
        "MYPE TRIBUTARIO": "RMT",
        "RG": "RG",
        "GENERAL": "RG",
        "REGIMEN GENERAL": "RG",
    }

    # Normalización simple de tildes frecuentes.
    valor = (
        valor.replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
        .replace("Á", "A")
    )

    return equivalencias.get(valor, "DESCONOCIDO")


def _es_personal(item: ItemDinamico) -> bool:
    texto = f"{item.nombre} {item.categoria}".lower()
    return any(
        palabra in texto
        for palabra in [
            "personal",
            "sueldo",
            "salario",
            "remuneracion",
            "remuneración",
            "planilla",
        ]
    )


def _es_capital_trabajo(item: ItemDinamico) -> bool:
    texto = f"{item.nombre} {item.categoria}".lower()
    return any(
        palabra in texto
        for palabra in [
            "insumo",
            "empaque",
            "inventario",
            "stock",
            "mercaderia",
            "mercadería",
        ]
    )


def _es_activo_depreciable(item: ItemDinamico) -> bool:
    texto = f"{item.nombre} {item.categoria}".lower()
    return any(
        palabra in texto
        for palabra in [
            "equipo",
            "maquinaria",
            "mobiliario",
            "activo fijo",
            "activo_fijo",
        ]
    )


def _tasa_mensual_equivalente(tasa_anual_pct: float) -> float:
    tasa_anual = tasa_anual_pct / 100.0

    if tasa_anual <= -1:
        return 0.0

    return (1 + tasa_anual) ** (1 / 12) - 1


def _calcular_van(tasa_mensual: float, flujos: list[float]) -> float:
    try:
        valor = float(npf.npv(tasa_mensual, flujos))
        return valor if math.isfinite(valor) else 0.0
    except Exception:
        return 0.0


def _calcular_tir(flujos: list[float]) -> dict:
    try:
        tir_mensual = float(npf.irr(flujos))

        if not math.isfinite(tir_mensual) or tir_mensual <= -1:
            return {
                "mensual": None,
                "anual": None,
                "interpretable": False,
                "alerta": "No se pudo obtener una TIR interpretable.",
            }

        tir_anual = ((1 + tir_mensual) ** 12 - 1) * 100
        tir_mensual_pct = tir_mensual * 100

        cambios_signo = 0
        signo_anterior = 0

        for flujo in flujos:
            if abs(flujo) < 1e-9:
                continue

            signo_actual = 1 if flujo > 0 else -1

            if signo_anterior and signo_actual != signo_anterior:
                cambios_signo += 1

            signo_anterior = signo_actual

        interpretable = cambios_signo == 1

        alerta = None

        if not interpretable:
            alerta = (
                "La serie tiene múltiples cambios de signo; "
                "la TIR puede ser ambigua. Prioriza VAN y flujos."
            )
        elif tir_anual > 1000:
            alerta = (
                "La TIR anualizada es extremadamente alta por la relación "
                "entre inversión inicial y flujos tempranos. Prioriza VAN, "
                "payback y sensibilidad para decidir."
            )

        return {
            "mensual": tir_mensual_pct,
            "anual": tir_anual,
            "interpretable": interpretable,
            "alerta": alerta,
        }

    except Exception:
        return {
            "mensual": None,
            "anual": None,
            "interpretable": False,
            "alerta": "No se pudo calcular la TIR.",
        }


def _calcular_payback(
    inversion_inicial: float,
    flujos_mensuales: list[float],
) -> Optional[float]:
    if inversion_inicial <= 0:
        return 0.0

    acumulado = -inversion_inicial

    for mes, flujo in enumerate(flujos_mensuales, start=1):
        anterior = acumulado
        acumulado += flujo

        if anterior < 0 <= acumulado and flujo > 0:
            fraccion = abs(anterior) / flujo
            return (mes - 1) + fraccion

    return None


def _calcular_bc_separado(
    tasa_mensual: float,
    beneficios: list[float],
    costos: list[float],
) -> float:
    pv_beneficios = 0.0
    pv_costos = 0.0

    total_periodos = max(len(beneficios), len(costos))

    for periodo in range(total_periodos):
        factor = (1 + tasa_mensual) ** periodo

        if factor <= 0:
            continue

        beneficio = beneficios[periodo] if periodo < len(beneficios) else 0.0
        costo = costos[periodo] if periodo < len(costos) else 0.0

        pv_beneficios += max(0.0, beneficio) / factor
        pv_costos += max(0.0, costo) / factor

    return pv_beneficios / pv_costos if pv_costos > 0 else 0.0


def _calcular_bc_desde_flujos(
    tasa_mensual: float,
    flujos: list[float],
) -> float:
    beneficios = []
    costos = []

    for flujo in flujos:
        beneficios.append(max(0.0, flujo))
        costos.append(max(0.0, -flujo))

    return _calcular_bc_separado(
        tasa_mensual,
        beneficios,
        costos,
    )


def _descomponer_igv(
    monto_ingresado: float,
    incluye_igv: bool,
    aplica_igv: bool,
) -> tuple[float, float, float]:
    monto = max(0.0, float(monto_ingresado))

    if not aplica_igv:
        return monto, 0.0, monto

    if incluye_igv:
        base = monto / (1 + IGV_TASA)
        igv = monto - base
        caja = monto
    else:
        base = monto
        igv = base * IGV_TASA
        caja = base + igv

    return base, igv, caja


def _impuesto_rmt_anual(renta_neta: float, uit: float) -> float:
    renta = max(0.0, renta_neta)
    tramo_10 = 15 * uit

    if renta <= tramo_10:
        return renta * 0.10

    return (
        tramo_10 * 0.10
        + (renta - tramo_10) * 0.295
    )


def _impuesto_rg_anual(renta_neta: float) -> float:
    return max(0.0, renta_neta) * 0.295


def _max_anual(valores: list[float]) -> float:
    maximo = 0.0

    for inicio in range(0, len(valores), 12):
        maximo = max(
            maximo,
            sum(valores[inicio:inicio + 12]),
        )

    return maximo


def _determinar_regimen_calculo(
    regimen_seleccionado: str,
    ventas_limite: list[float],
    adquisiciones_limite: list[float],
    activos_fijos_estimados: float,
    uit: float,
) -> dict:
    advertencias: list[str] = []
    compatible = True
    regimen_calculo = regimen_seleccionado
    calculo_referencial = False

    max_venta_mensual = max(ventas_limite, default=0.0)
    max_adq_mensual = max(adquisiciones_limite, default=0.0)
    max_venta_anual = _max_anual(ventas_limite)
    max_adq_anual = _max_anual(adquisiciones_limite)

    if regimen_seleccionado == "NRUS":
        razones = []

        if max_venta_mensual > NRUS_LIMITE_MENSUAL:
            razones.append(
                f"ventas mensuales proyectadas superan S/ {NRUS_LIMITE_MENSUAL:,.0f}"
            )

        if max_adq_mensual > NRUS_LIMITE_MENSUAL:
            razones.append(
                f"adquisiciones mensuales estimadas superan S/ {NRUS_LIMITE_MENSUAL:,.0f}"
            )

        if max_venta_anual > NRUS_LIMITE_ANUAL:
            razones.append(
                f"ventas anuales proyectadas superan S/ {NRUS_LIMITE_ANUAL:,.0f}"
            )

        if max_adq_anual > NRUS_LIMITE_ANUAL:
            razones.append(
                f"adquisiciones anuales estimadas superan S/ {NRUS_LIMITE_ANUAL:,.0f}"
            )

        if activos_fijos_estimados > NRUS_ACTIVOS_FIJOS_LIMITE:
            razones.append(
                f"activos fijos estimados superan S/ {NRUS_ACTIVOS_FIJOS_LIMITE:,.0f}"
            )

        if razones:
            compatible = False
            regimen_calculo = "RMT_REFERENCIAL"
            calculo_referencial = True
            advertencias.append(
                "El NRUS seleccionado resulta incompatible con la proyección: "
                + "; ".join(razones)
                + ". Para no subestimar tributos, el motor usa RMT solo como "
                "referencia de cálculo; no cambia tu régimen automáticamente."
            )

    elif regimen_seleccionado == "RER":
        razones = []

        if max_venta_anual > RER_LIMITE_ANUAL:
            razones.append(
                f"ingresos anuales proyectados superan S/ {RER_LIMITE_ANUAL:,.0f}"
            )

        if max_adq_anual > RER_LIMITE_ANUAL:
            razones.append(
                f"adquisiciones anuales estimadas superan S/ {RER_LIMITE_ANUAL:,.0f}"
            )

        if activos_fijos_estimados > RER_ACTIVOS_FIJOS_LIMITE:
            razones.append(
                f"activos fijos estimados superan S/ {RER_ACTIVOS_FIJOS_LIMITE:,.0f}"
            )

        advertencias.append(
            "La validación del RER no puede confirmar desde este simulador "
            "restricciones por actividad ni número de trabajadores."
        )

        if razones:
            compatible = False
            regimen_calculo = "RMT_REFERENCIAL"
            calculo_referencial = True
            advertencias.append(
                "El RER seleccionado resulta incompatible con la proyección: "
                + "; ".join(razones)
                + ". Para no subestimar tributos, el motor usa RMT solo como "
                "referencia de cálculo; no cambia tu régimen automáticamente."
            )

    elif regimen_seleccionado == "RMT":
        limite_rmt = 1700 * uit

        if max_venta_anual > limite_rmt:
            compatible = False
            regimen_calculo = "RG_REFERENCIAL"
            calculo_referencial = True
            advertencias.append(
                f"Los ingresos anuales proyectados superan 1700 UIT "
                f"(S/ {limite_rmt:,.0f} con la UIT indicada). "
                "El motor usa Régimen General como referencia de cálculo."
            )

    elif regimen_seleccionado == "RG":
        pass

    else:
        compatible = False
        regimen_calculo = "RMT_REFERENCIAL"
        calculo_referencial = True
        advertencias.append(
            "El régimen tributario ingresado no fue reconocido. "
            "El motor usa RMT únicamente como referencia y marca "
            "la tributación como no validada."
        )

    return {
        "regimen_seleccionado": regimen_seleccionado,
        "regimen_calculo": regimen_calculo,
        "compatible": compatible,
        "calculo_referencial": calculo_referencial,
        "advertencias": advertencias,
        "max_venta_mensual": max_venta_mensual,
        "max_adq_mensual": max_adq_mensual,
        "max_venta_anual": max_venta_anual,
        "max_adq_anual": max_adq_anual,
    }


def _regimen_base(regimen_calculo: str) -> str:
    if regimen_calculo.startswith("RMT"):
        return "RMT"

    if regimen_calculo.startswith("RG"):
        return "RG"

    return regimen_calculo


# ============================================================
# SIMULADOR
# ============================================================


@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_items = _normalizar_inversion(datos)
    gasto_items = _normalizar_gastos(datos)

    advertencias_generales: list[str] = []

    inversion_total = sum(
        max(0.0, float(item.monto))
        for item in inversion_items
    )

    gastos_fijos_base = sum(
        max(0.0, float(item.monto))
        for item in gasto_items
    )

    gastos_personal_base = sum(
        max(0.0, float(item.monto))
        for item in gasto_items
        if _es_personal(item)
    )

    gastos_no_personal_base = max(
        0.0,
        gastos_fijos_base - gastos_personal_base,
    )

    capital_trabajo_inicial = sum(
        max(0.0, float(item.monto))
        for item in inversion_items
        if _es_capital_trabajo(item)
    )

    recuperacion_pct = max(
        0.0,
        min(
            100.0,
            float(datos.porcentaje_recuperacion_capital_trabajo),
        ),
    )

    recuperacion_capital_trabajo = (
        capital_trabajo_inicial
        * recuperacion_pct
        / 100.0
    )

    if datos.anio_tributario != 2026:
        advertencias_generales.append(
            "Las reglas tributarias automáticas de esta versión están "
            "calibradas con parámetros de referencia 2026. Revisa límites, "
            "UIT y tasas si proyectas otro año."
        )

    if abs(datos.uit_referencia - 5500.0) > 0.01:
        advertencias_generales.append(
            "La UIT fue modificada respecto de la referencia 2026 de S/ 5,500."
        )

    if datos.ventas.optimista < datos.ventas.base:
        advertencias_generales.append(
            "Las ventas optimistas son menores que las ventas base. "
            "Para evitar recortes artificiales, la capacidad se ajusta "
            "al menos al nivel inicial del escenario."
        )

    if datos.ventas.base < datos.ventas.pesimista:
        advertencias_generales.append(
            "Las ventas base son menores que las pesimistas. "
            "Revisa los escenarios ingresados."
        )

    if datos.precio_venta <= 0:
        advertencias_generales.append(
            "El precio de venta es cero o negativo; los indicadores "
            "de rentabilidad no serán útiles."
        )

    if datos.costo_directo < 0:
        advertencias_generales.append(
            "El costo directo negativo se tratará como cero."
        )

    if datos.inflacion_anual <= -100:
        advertencias_generales.append(
            "La inflación anual menor o igual a -100% no es válida; "
            "se usará 0% para evitar resultados matemáticos imposibles."
        )

    # --- DEPRECIACIÓN ECONÓMICA ---
    depreciacion_mensual = 0.0
    valor_activos_inicial = 0.0
    valor_residual_activos_final = 0.0
    valor_activos_mes_36 = 0.0

    for item in inversion_items:
        monto_item = max(0.0, float(item.monto))

        vida_util_efectiva = int(item.vida_util)
        residual_efectivo = float(item.residual)

        if (
            _es_activo_depreciable(item)
            and vida_util_efectiva <= 0
        ):
            vida_util_efectiva = VIDA_UTIL_EQUIPOS_DEFAULT

            if residual_efectivo <= 0:
                residual_efectivo = (
                    monto_item
                    * RESIDUAL_EQUIPOS_DEFAULT
                )

        if vida_util_efectiva > 0:
            residual_efectivo = max(
                0.0,
                min(residual_efectivo, monto_item),
            )

            base_depreciable = max(
                0.0,
                monto_item - residual_efectivo,
            )

            depreciacion_item = (
                base_depreciable
                / vida_util_efectiva
            )

            depreciacion_mensual += depreciacion_item
            valor_activos_inicial += monto_item
            valor_residual_activos_final += residual_efectivo

            meses_depreciados = min(
                36,
                vida_util_efectiva,
            )

            valor_item_mes_36 = max(
                residual_efectivo,
                monto_item
                - (
                    depreciacion_item
                    * meses_depreciados
                ),
            )

            valor_activos_mes_36 += (
                valor_item_mes_36
            )

    valor_terminal_total = (
        valor_activos_mes_36
        + recuperacion_capital_trabajo
    )

    # --- FINANCIAMIENTO ---
    monto_financiar = max(
        0.0,
        float(datos.financiamiento_monto),
    )

    tasa_prestamo = max(
        0.0,
        float(datos.financiamiento_tasa_mensual)
        / 100.0,
    )

    plazo = max(
        0,
        int(datos.financiamiento_plazo),
    )

    cuota_prestamo = 0.0

    if monto_financiar > 0 and plazo <= 0:
        advertencias_generales.append(
            "Existe monto de financiamiento pero el plazo es 0; "
            "no se puede construir un cronograma de deuda válido."
        )

    if monto_financiar > 0 and plazo > 0:
        if tasa_prestamo > 0:
            cuota_prestamo = (
                monto_financiar
                * (
                    tasa_prestamo
                    * (1 + tasa_prestamo) ** plazo
                )
                / (
                    (1 + tasa_prestamo) ** plazo
                    - 1
                )
            )
        else:
            cuota_prestamo = (
                monto_financiar / plazo
            )

    # --- MES 0 Y LIQUIDEZ ---
    capital_disponible = max(
        0.0,
        float(datos.capital_disponible),
    )

    recursos_disponibles = (
        capital_disponible
        + monto_financiar
    )

    deficit_apertura = max(
        0.0,
        inversion_total - recursos_disponibles,
    )

    caja_despues_apertura = max(
        0.0,
        recursos_disponibles - inversion_total,
    )

    reserva_emergencia = (
        gastos_fijos_base
        * max(0, int(datos.meses_reserva))
    )

    deficit_reserva = max(
        0.0,
        reserva_emergencia - caja_despues_apertura,
    )

    cobertura_meses = (
        caja_despues_apertura
        / gastos_fijos_base
        if gastos_fijos_base > 0
        else 0.0
    )

    if deficit_apertura > 0:
        estado_liquidez = (
            "🔴 FINANCIAMIENTO INSUFICIENTE"
        )
        alerta_liquidez = (
            f"Faltan {datos.moneda} "
            f"{deficit_apertura:,.2f} "
            "para financiar la apertura."
        )

    elif cobertura_meses < 1:
        estado_liquidez = "🟠 LIQUIDEZ CRÍTICA"
        alerta_liquidez = (
            "Puedes abrir, pero la caja restante "
            f"cubre solo {cobertura_meses:.1f} "
            "meses de gastos fijos."
        )

    elif cobertura_meses < max(
        0,
        int(datos.meses_reserva),
    ):
        estado_liquidez = "🟡 LIQUIDEZ AJUSTADA"
        alerta_liquidez = (
            "Puedes abrir, pero aún faltan "
            f"{datos.moneda} "
            f"{deficit_reserva:,.2f} "
            "para alcanzar la reserva objetivo."
        )

    else:
        estado_liquidez = "🟢 LIQUIDEZ SALUDABLE"
        alerta_liquidez = (
            "La apertura y la reserva objetivo "
            "están cubiertas."
        )

    # Compatibilidad con la V3.3.
    capital_invertible = caja_despues_apertura

    # --- APORTE PROPIO VS DEUDA ---
    prestamo_aplicado_inversion = min(
        monto_financiar,
        inversion_total,
    )

    aporte_propio_requerido = max(
        0.0,
        inversion_total
        - prestamo_aplicado_inversion,
    )

    aporte_propio_efectivo = min(
        capital_disponible,
        aporte_propio_requerido,
    )

    metricas_inversionista_validas = (
        deficit_apertura <= 0
        and aporte_propio_efectivo > 0
    )

    # --- DEDICACIÓN ESTIMADA ---
    sector_str = datos.sector.lower()

    es_digital = any(
        keyword in sector_str
        for keyword in [
            "tech",
            "e-commerce",
            "digital",
            "online",
            "software",
            "web",
            "tecnología",
            "tecnologia",
        ]
    )

    es_alimentos = any(
        keyword in sector_str
        for keyword in [
            "gastro",
            "alimento",
            "restaurante",
            "food",
            "cafeteria",
            "cafetería",
        ]
    )

    if es_digital:
        horas_semana, presencia = 25, 5
    elif es_alimentos:
        horas_semana, presencia = 60, 90
    else:
        horas_semana, presencia = 45, 70

    sueldo_emprendedor = gastos_personal_base

    if sueldo_emprendedor > 2000:
        horas_semana = min(
            80,
            horas_semana + 15,
        )
        presencia = min(
            100,
            presencia + 10,
        )

    tasa_descuento_mensual = (
        _tasa_mensual_equivalente(
            datos.tasa_descuento
        )
    )

    regimen_seleccionado = (
        _normalizar_regimen(
            datos.regimen_tributario
        )
    )

    # ========================================================
    # PROYECCIÓN DE ESCENARIOS
    # ========================================================

    def proyectar_escenario(
        ventas_iniciales: float,
        crecimiento: float,
        precio_mult: float = 1.0,
        costo_mult: float = 1.0,
    ):
        precio_final = max(
            0.0,
            datos.precio_venta * precio_mult,
        )

        costo_final = max(
            0.0,
            datos.costo_directo * costo_mult,
        )

        # Primera pasada: cantidades y montos ingresados.
        filas_previas: list[dict] = []
        ventas_limite: list[float] = []
        adquisiciones_limite: list[float] = []

        for mes in range(1, 37):
            idx_estacional = (mes - 1) % 12

            ajuste_estacional = (
                datos.estacionalidad[idx_estacional]
                if idx_estacional < len(
                    datos.estacionalidad
                )
                else 0.0
            )

            factor_estacional = max(
                0.0,
                1.0
                + (ajuste_estacional / 100.0),
            )

            capacidad_ventas = max(
                0.0,
                float(datos.ventas.optimista),
                float(ventas_iniciales),
            )

            crecimiento_decimal = (
                crecimiento / 100.0
            )

            if crecimiento_decimal <= -1:
                ventas_tendencia = (
                    ventas_iniciales
                    if mes == 1
                    else 0.0
                )
            else:
                ventas_tendencia = (
                    ventas_iniciales
                    * (
                        (1 + crecimiento_decimal)
                        ** (mes - 1)
                    )
                )

            ventas_mes = min(
                max(
                    0.0,
                    ventas_tendencia
                    * factor_estacional,
                ),
                capacidad_ventas,
            )

            inflacion_anual_decimal = (
                datos.inflacion_anual / 100.0
            )

            if inflacion_anual_decimal <= -1:
                inflacion_mensual = 0.0
            else:
                inflacion_mensual = (
                    (1 + inflacion_anual_decimal)
                    ** (1 / 12)
                    - 1
                )

            factor_inflacion = (
                (1 + inflacion_mensual)
                ** (mes - 1)
            )

            ventas_monto_ingresado = (
                ventas_mes * precio_final
            )

            costo_unitario_ingresado = (
                costo_final
                * factor_inflacion
            )

            costos_variables_ingresados = (
                ventas_mes
                * costo_unitario_ingresado
            )

            gastos_personal_ingresados = (
                gastos_personal_base
                * factor_inflacion
            )

            gastos_no_personal_ingresados = (
                gastos_no_personal_base
                * factor_inflacion
            )

            filas_previas.append(
                {
                    "mes": mes,
                    "ventas_unidades": ventas_mes,
                    "factor_inflacion": factor_inflacion,
                    "ventas_ingresadas": ventas_monto_ingresado,
                    "costos_variables_ingresados": costos_variables_ingresados,
                    "gastos_personal_ingresados": gastos_personal_ingresados,
                    "gastos_no_personal_ingresados": gastos_no_personal_ingresados,
                }
            )

            # Para RER/RMT/RG los límites se aproximan con montos
            # netos de IGV cuando corresponde. Para NRUS se mantienen
            # los montos brutos ingresados, porque su control se realiza
            # sobre ingresos/adquisiciones brutos.
            if (
                regimen_seleccionado in {"RER", "RMT", "RG"}
                and bool(datos.aplica_igv)
            ):
                ventas_limite_mes, _, _ = _descomponer_igv(
                    ventas_monto_ingresado,
                    bool(datos.precio_incluye_igv),
                    True,
                )

                costos_limite_mes, _, _ = _descomponer_igv(
                    costos_variables_ingresados,
                    bool(datos.costo_directo_incluye_igv),
                    True,
                )

                gastos_limite_mes, _, _ = _descomponer_igv(
                    gastos_no_personal_ingresados,
                    bool(datos.gastos_fijos_incluyen_igv),
                    True,
                )

                ventas_limite.append(
                    ventas_limite_mes
                )

                adquisiciones_limite.append(
                    costos_limite_mes
                    + gastos_limite_mes
                )

            else:
                ventas_limite.append(
                    ventas_monto_ingresado
                )

                adquisiciones_limite.append(
                    costos_variables_ingresados
                    + gastos_no_personal_ingresados
                )

        tributacion = (
            _determinar_regimen_calculo(
                regimen_seleccionado,
                ventas_limite,
                adquisiciones_limite,
                valor_activos_inicial,
                max(
                    1.0,
                    float(datos.uit_referencia),
                ),
            )
        )

        regimen_calculo = _regimen_base(
            tributacion["regimen_calculo"]
        )

        usa_igv = (
            bool(datos.aplica_igv)
            and regimen_calculo
            in {"RER", "RMT", "RG"}
        )

        # PROYECTO: parte de toda la inversión.
        caja_acumulada_proyecto = (
            -inversion_total
        )

        # INVERSIONISTA: parte solo del aporte propio.
        caja_acumulada_inversionista = (
            -aporte_propio_efectivo
        )

        flujo_proyecto_mensual: list[float] = []
        flujo_inversionista_mensual: list[float] = []

        caja_mensual_proyecto: list[float] = []
        caja_mensual_inversionista: list[float] = []

        beneficios_proyecto = [0.0]
        costos_proyecto = [inversion_total]

        p_y_g: list[dict] = []

        ingresos_contables_totales = 0.0
        utilidad_neta_total = 0.0
        igv_pagado_total = 0.0
        impuesto_renta_total = 0.0
        cuota_nrus_total = 0.0

        saldo_prestamo = monto_financiar
        credito_igv_arrastre = 0.0

        renta_acumulada_anio = 0.0
        impuesto_anual_reconocido = 0.0

        advertencias_escenario = list(
            tributacion["advertencias"]
        )

        if usa_igv:
            advertencias_escenario.append(
                "El IGV se modela por débito fiscal menos crédito fiscal "
                "operativo estimado. No se valida si cada gasto cuenta con "
                "comprobante válido para crédito fiscal."
            )

        if regimen_calculo in {"RMT", "RG"}:
            advertencias_escenario.append(
                "El Impuesto a la Renta del RMT/RG se estima sobre una "
                "renta operativa simplificada. No sustituye una liquidación "
                "tributaria ni valida todos los gastos deducibles."
            )

        for fila in filas_previas:
            mes = fila["mes"]

            if (mes - 1) % 12 == 0:
                renta_acumulada_anio = 0.0
                impuesto_anual_reconocido = 0.0

            ventas_ingresadas = (
                fila["ventas_ingresadas"]
            )

            costos_variables_ingresados = (
                fila[
                    "costos_variables_ingresados"
                ]
            )

            gastos_personal_ingresados = (
                fila[
                    "gastos_personal_ingresados"
                ]
            )

            gastos_no_personal_ingresados = (
                fila[
                    "gastos_no_personal_ingresados"
                ]
            )

            # Ventas: contabilidad, IGV y caja.
            (
                ingresos_contables,
                igv_debito,
                cobros_ventas,
            ) = _descomponer_igv(
                ventas_ingresadas,
                bool(datos.precio_incluye_igv),
                usa_igv,
            )

            # Costos variables: contabilidad, IGV y caja.
            (
                costos_variables_contables,
                igv_credito_variable,
                pagos_variables,
            ) = _descomponer_igv(
                costos_variables_ingresados,
                bool(
                    datos.costo_directo_incluye_igv
                ),
                usa_igv,
            )

            # Gastos no personales potencialmente gravados.
            (
                gastos_no_personal_contables,
                igv_credito_fijo,
                pagos_no_personal,
            ) = _descomponer_igv(
                gastos_no_personal_ingresados,
                bool(
                    datos.gastos_fijos_incluyen_igv
                ),
                usa_igv,
            )

            # Personal se mantiene sin IGV.
            gastos_fijos_contables = (
                gastos_personal_ingresados
                + gastos_no_personal_contables
            )

            pagos_gastos_fijos = (
                gastos_personal_ingresados
                + pagos_no_personal
            )

            margen_bruto = (
                ingresos_contables
                - costos_variables_contables
            )

            ebit = (
                margen_bruto
                - gastos_fijos_contables
                - depreciacion_mensual
            )

            # --- IMPUESTO A LA RENTA / NRUS ---
            impuesto_renta = 0.0
            cuota_nrus = 0.0
            pago_a_cuenta_referencial = 0.0

            if regimen_calculo == "NRUS":
                adquisiciones_mes = (
                    costos_variables_ingresados
                    + gastos_no_personal_ingresados
                )

                parametro_nrus = max(
                    ventas_ingresadas,
                    adquisiciones_mes,
                )

                if parametro_nrus <= 5000:
                    cuota_nrus = 20.0
                elif parametro_nrus <= 8000:
                    cuota_nrus = 50.0
                else:
                    # No debería ocurrir si quedó como NRUS compatible.
                    cuota_nrus = 50.0
                    advertencias_escenario.append(
                        "Se detectó un mes fuera del límite NRUS "
                        "durante la proyección."
                    )

            elif regimen_calculo == "RER":
                impuesto_renta = (
                    ingresos_contables * 0.015
                )

            elif regimen_calculo in {"RMT", "RG"}:
                renta_acumulada_anio += ebit

                renta_neta_acumulada = max(
                    0.0,
                    renta_acumulada_anio,
                )

                if regimen_calculo == "RMT":
                    impuesto_anual_actual = (
                        _impuesto_rmt_anual(
                            renta_neta_acumulada,
                            max(
                                1.0,
                                float(
                                    datos.uit_referencia
                                ),
                            ),
                        )
                    )

                    ventas_anuales_ref = (
                        tributacion[
                            "max_venta_anual"
                        ]
                    )

                    limite_300_uit = (
                        300
                        * max(
                            1.0,
                            float(
                                datos.uit_referencia
                            ),
                        )
                    )

                    tasa_pago_cuenta = (
                        0.01
                        if ventas_anuales_ref
                        <= limite_300_uit
                        else 0.015
                    )

                    pago_a_cuenta_referencial = (
                        ingresos_contables
                        * tasa_pago_cuenta
                    )

                    if (
                        ventas_anuales_ref
                        > limite_300_uit
                    ):
                        aviso = (
                            "Al superar 300 UIT, el pago a cuenta del RMT "
                            "puede depender de coeficiente o 1.5%. "
                            "El simulador usa 1.5% como referencia."
                        )

                        if aviso not in advertencias_escenario:
                            advertencias_escenario.append(
                                aviso
                            )

                else:
                    impuesto_anual_actual = (
                        _impuesto_rg_anual(
                            renta_neta_acumulada
                        )
                    )

                impuesto_renta = max(
                    0.0,
                    impuesto_anual_actual
                    - impuesto_anual_reconocido,
                )

                impuesto_anual_reconocido += (
                    impuesto_renta
                )

            impuestos_resultado = (
                impuesto_renta
                + cuota_nrus
            )

            utilidad_neta = (
                ebit - impuestos_resultado
            )

            # --- IGV CAJA ---
            igv_credito_operativo = (
                igv_credito_variable
                + igv_credito_fijo
            )

            credito_disponible = (
                credito_igv_arrastre
                + igv_credito_operativo
            )

            igv_pagado = max(
                0.0,
                igv_debito - credito_disponible,
            )

            credito_igv_arrastre = max(
                0.0,
                credito_disponible - igv_debito,
            )

            # --- FLUJO ECONÓMICO DEL PROYECTO ---
            flujo_caja_proyecto = (
                cobros_ventas
                - pagos_variables
                - pagos_gastos_fijos
                - impuesto_renta
                - cuota_nrus
                - igv_pagado
            )

            # --- CRONOGRAMA DE PRÉSTAMO ---
            pago_prestamo = 0.0
            interes_prestamo = 0.0
            amortizacion_prestamo = 0.0

            if (
                mes <= plazo
                and saldo_prestamo > 0
                and cuota_prestamo > 0
            ):
                interes_prestamo = (
                    saldo_prestamo
                    * tasa_prestamo
                )

                amortizacion_prestamo = max(
                    0.0,
                    cuota_prestamo
                    - interes_prestamo,
                )

                amortizacion_prestamo = min(
                    amortizacion_prestamo,
                    saldo_prestamo,
                )

                pago_prestamo = (
                    interes_prestamo
                    + amortizacion_prestamo
                )

                saldo_prestamo = max(
                    0.0,
                    saldo_prestamo
                    - amortizacion_prestamo,
                )

            flujo_caja_inversionista = (
                flujo_caja_proyecto
                - pago_prestamo
            )

            valor_terminal_mes = 0.0

            if mes == 36:
                valor_terminal_mes = (
                    valor_terminal_total
                )

                flujo_caja_proyecto += (
                    valor_terminal_mes
                )

                flujo_caja_inversionista += (
                    valor_terminal_mes
                )

            flujo_proyecto_mensual.append(
                flujo_caja_proyecto
            )

            flujo_inversionista_mensual.append(
                flujo_caja_inversionista
            )

            caja_acumulada_proyecto += (
                flujo_caja_proyecto
            )

            caja_acumulada_inversionista += (
                flujo_caja_inversionista
            )

            caja_mensual_proyecto.append(
                round(
                    caja_acumulada_proyecto,
                    2,
                )
            )

            caja_mensual_inversionista.append(
                round(
                    caja_acumulada_inversionista,
                    2,
                )
            )

            # Beneficio/costo del proyecto en términos de caja.
            beneficios_proyecto.append(
                cobros_ventas
                + valor_terminal_mes
            )

            costos_proyecto.append(
                pagos_variables
                + pagos_gastos_fijos
                + impuesto_renta
                + cuota_nrus
                + igv_pagado
            )

            ingresos_contables_totales += (
                ingresos_contables
            )

            utilidad_neta_total += utilidad_neta
            igv_pagado_total += igv_pagado
            impuesto_renta_total += impuesto_renta
            cuota_nrus_total += cuota_nrus

            p_y_g.append(
                {
                    "mes": mes,
                    "ventas_unidades": round(
                        fila["ventas_unidades"],
                        1,
                    ),
                    "ventas_brutas_caja": round(
                        cobros_ventas,
                        2,
                    ),
                    "ingresos": round(
                        ingresos_contables,
                        2,
                    ),
                    "costos_variables": round(
                        costos_variables_contables,
                        2,
                    ),
                    "margen_bruto": round(
                        margen_bruto,
                        2,
                    ),
                    "gastos_fijos": round(
                        gastos_fijos_contables,
                        2,
                    ),
                    "depreciacion": round(
                        depreciacion_mensual,
                        2,
                    ),
                    "ebit": round(
                        ebit,
                        2,
                    ),
                    "impuesto_renta": round(
                        impuesto_renta,
                        2,
                    ),
                    "cuota_nrus": round(
                        cuota_nrus,
                        2,
                    ),
                    "pago_a_cuenta_rmt_referencial": round(
                        pago_a_cuenta_referencial,
                        2,
                    ),
                    "igv_debito": round(
                        igv_debito,
                        2,
                    ),
                    "igv_credito_operativo": round(
                        igv_credito_operativo,
                        2,
                    ),
                    "igv_pagado": round(
                        igv_pagado,
                        2,
                    ),
                    "credito_igv_arrastre": round(
                        credito_igv_arrastre,
                        2,
                    ),
                    # Compatibilidad con frontend actual.
                    "impuestos": round(
                        impuestos_resultado,
                        2,
                    ),
                    "utilidad_neta": round(
                        utilidad_neta,
                        2,
                    ),
                    "flujo_caja_neto": round(
                        flujo_caja_proyecto,
                        2,
                    ),
                    "flujo_caja_proyecto": round(
                        flujo_caja_proyecto,
                        2,
                    ),
                    "flujo_caja_inversionista": round(
                        flujo_caja_inversionista,
                        2,
                    ),
                    "valor_terminal": round(
                        valor_terminal_mes,
                        2,
                    ),
                    "pago_prestamo": round(
                        pago_prestamo,
                        2,
                    ),
                    "interes_prestamo": round(
                        interes_prestamo,
                        2,
                    ),
                    "amortizacion_prestamo": round(
                        amortizacion_prestamo,
                        2,
                    ),
                    "saldo_prestamo": round(
                        saldo_prestamo,
                        2,
                    ),
                    "caja_acumulada": round(
                        caja_acumulada_proyecto,
                        2,
                    ),
                    "caja_acumulada_proyecto": round(
                        caja_acumulada_proyecto,
                        2,
                    ),
                    "caja_acumulada_inversionista": round(
                        caja_acumulada_inversionista,
                        2,
                    ),
                }
            )

        # --- MÉTRICAS PROYECTO ---
        flujos_proyecto = [
            -inversion_total
        ] + flujo_proyecto_mensual

        van_proyecto = _calcular_van(
            tasa_descuento_mensual,
            flujos_proyecto,
        )

        tir_proyecto_info = _calcular_tir(
            flujos_proyecto
        )

        roi_proyecto = (
            (
                caja_acumulada_proyecto
                / inversion_total
            )
            * 100
            if inversion_total > 0
            else 0.0
        )

        b_c_proyecto = (
            _calcular_bc_separado(
                tasa_descuento_mensual,
                beneficios_proyecto,
                costos_proyecto,
            )
        )

        payback_proyecto = _calcular_payback(
            inversion_total,
            flujo_proyecto_mensual,
        )

        # --- MÉTRICAS INVERSIONISTA ---
        flujos_inversionista = [
            -aporte_propio_efectivo
        ] + flujo_inversionista_mensual

        van_inversionista: Optional[float] = None
        tir_inversionista_info = {
            "mensual": None,
            "anual": None,
            "interpretable": False,
            "alerta": None,
        }
        roi_inversionista: Optional[float] = None
        b_c_inversionista: Optional[float] = None
        payback_inversionista: Optional[float] = None

        if metricas_inversionista_validas:
            van_inversionista = _calcular_van(
                tasa_descuento_mensual,
                flujos_inversionista,
            )

            tir_inversionista_info = (
                _calcular_tir(
                    flujos_inversionista
                )
            )

            roi_inversionista = (
                (
                    caja_acumulada_inversionista
                    / aporte_propio_efectivo
                )
                * 100
                if aporte_propio_efectivo > 0
                else None
            )

            b_c_inversionista = (
                _calcular_bc_desde_flujos(
                    tasa_descuento_mensual,
                    flujos_inversionista,
                )
            )

            payback_inversionista = (
                _calcular_payback(
                    aporte_propio_efectivo,
                    flujo_inversionista_mensual,
                )
            )

        margen_neto = (
            (
                utilidad_neta_total
                / ingresos_contables_totales
            )
            * 100
            if ingresos_contables_totales > 0
            else 0.0
        )

        tir_proyecto_anual = (
            tir_proyecto_info["anual"]
        )

        tir_inversionista_anual = (
            tir_inversionista_info["anual"]
        )

        return {
            # Compatibilidad V3.3:
            # campos principales = PROYECTO.
            "caja_mensual": caja_mensual_proyecto,
            "caja_mes_a_mes": caja_mensual_proyecto,
            "p_y_g": p_y_g,
            "van": round(
                van_proyecto,
                2,
            ),
            "tir": (
                round(
                    tir_proyecto_anual,
                    2,
                )
                if tir_proyecto_anual
                is not None
                else None
            ),
            "roi": round(
                roi_proyecto,
                2,
            ),
            "b_c": round(
                b_c_proyecto,
                2,
            ),
            "payback_meses": (
                round(
                    payback_proyecto,
                    2,
                )
                if payback_proyecto
                is not None
                else None
            ),
            "caja_final": round(
                caja_acumulada_proyecto,
                2,
            ),
            "mes_recuperacion": (
                math.ceil(payback_proyecto)
                if payback_proyecto
                is not None
                else "No recupera"
            ),
            "margen_neto": round(
                margen_neto,
                2,
            ),
            "tributacion": {
                **tributacion,
                "usa_igv": usa_igv,
                "igv_pagado_total": round(
                    igv_pagado_total,
                    2,
                ),
                "impuesto_renta_total": round(
                    impuesto_renta_total,
                    2,
                ),
                "cuota_nrus_total": round(
                    cuota_nrus_total,
                    2,
                ),
                "advertencias": list(
                    dict.fromkeys(
                        advertencias_escenario
                    )
                ),
            },
            "proyecto": {
                "van": round(
                    van_proyecto,
                    2,
                ),
                "tir": (
                    round(
                        tir_proyecto_anual,
                        2,
                    )
                    if tir_proyecto_anual
                    is not None
                    else None
                ),
                "tir_mensual": (
                    round(
                        tir_proyecto_info[
                            "mensual"
                        ],
                        4,
                    )
                    if tir_proyecto_info[
                        "mensual"
                    ]
                    is not None
                    else None
                ),
                "tir_interpretable": (
                    tir_proyecto_info[
                        "interpretable"
                    ]
                ),
                "alerta_tir": (
                    tir_proyecto_info[
                        "alerta"
                    ]
                ),
                "roi": round(
                    roi_proyecto,
                    2,
                ),
                "b_c": round(
                    b_c_proyecto,
                    2,
                ),
                "payback_meses": (
                    round(
                        payback_proyecto,
                        2,
                    )
                    if payback_proyecto
                    is not None
                    else None
                ),
                "inversion_inicial": round(
                    inversion_total,
                    2,
                ),
                "caja_final": round(
                    caja_acumulada_proyecto,
                    2,
                ),
                "mes_recuperacion": (
                    math.ceil(
                        payback_proyecto
                    )
                    if payback_proyecto
                    is not None
                    else "No recupera"
                ),
                "caja_mensual": (
                    caja_mensual_proyecto
                ),
            },
            "inversionista": {
                "metricas_validas": (
                    metricas_inversionista_validas
                ),
                "aporte_propio": round(
                    aporte_propio_efectivo,
                    2,
                ),
                "aporte_propio_requerido": round(
                    aporte_propio_requerido,
                    2,
                ),
                "prestamo": round(
                    monto_financiar,
                    2,
                ),
                "van": (
                    round(
                        van_inversionista,
                        2,
                    )
                    if van_inversionista
                    is not None
                    else None
                ),
                "tir": (
                    round(
                        tir_inversionista_anual,
                        2,
                    )
                    if tir_inversionista_anual
                    is not None
                    else None
                ),
                "tir_mensual": (
                    round(
                        tir_inversionista_info[
                            "mensual"
                        ],
                        4,
                    )
                    if tir_inversionista_info[
                        "mensual"
                    ]
                    is not None
                    else None
                ),
                "tir_interpretable": (
                    tir_inversionista_info[
                        "interpretable"
                    ]
                ),
                "alerta_tir": (
                    tir_inversionista_info[
                        "alerta"
                    ]
                ),
                "roi": (
                    round(
                        roi_inversionista,
                        2,
                    )
                    if roi_inversionista
                    is not None
                    else None
                ),
                "b_c": (
                    round(
                        b_c_inversionista,
                        2,
                    )
                    if b_c_inversionista
                    is not None
                    else None
                ),
                "payback_meses": (
                    round(
                        payback_inversionista,
                        2,
                    )
                    if payback_inversionista
                    is not None
                    else None
                ),
                "caja_final": round(
                    caja_acumulada_inversionista,
                    2,
                ),
                "mes_recuperacion": (
                    math.ceil(
                        payback_inversionista
                    )
                    if payback_inversionista
                    is not None
                    else "No recupera"
                ),
                "caja_mensual": (
                    caja_mensual_inversionista
                ),
            },
        }

    escenario_base = proyectar_escenario(
        datos.ventas.base,
        datos.ventas.crecimiento_mensual,
    )

    escenario_pesimista = proyectar_escenario(
        datos.ventas.pesimista,
        0.0,
    )

    escenario_optimista = proyectar_escenario(
        datos.ventas.optimista,
        0.0,
    )

    # --- PUNTO DE EQUILIBRIO ---
    # Se calcula con el mes 1 del escenario base para respetar
    # la forma en que el motor trató IGV y gastos.
    py_mes_1 = (
        escenario_base["p_y_g"][0]
        if escenario_base["p_y_g"]
        else None
    )

    if py_mes_1 and py_mes_1["ventas_unidades"] > 0:
        unidades_m1 = py_mes_1["ventas_unidades"]

        ingreso_unitario_contable = (
            py_mes_1["ingresos"]
            / unidades_m1
        )

        costo_unitario_contable = (
            py_mes_1["costos_variables"]
            / unidades_m1
        )

        margen_unitario = (
            ingreso_unitario_contable
            - costo_unitario_contable
        )

        gastos_fijos_equilibrio = (
            py_mes_1["gastos_fijos"]
        )
    else:
        margen_unitario = (
            datos.precio_venta
            - max(
                0.0,
                datos.costo_directo,
            )
        )
        gastos_fijos_equilibrio = (
            gastos_fijos_base
        )

    punto_equilibrio = (
        999999
        if margen_unitario <= 0
        else math.ceil(
            gastos_fijos_equilibrio
            / margen_unitario
        )
    )

    punto_equilibrio_con_deuda = (
        999999
        if margen_unitario <= 0
        else math.ceil(
            (
                gastos_fijos_equilibrio
                + cuota_prestamo
            )
            / margen_unitario
        )
    )

    margen_seguridad = (
        max(
            0.0,
            (
                (
                    datos.ventas.base
                    - punto_equilibrio
                )
                / datos.ventas.base
            )
            * 100,
        )
        if datos.ventas.base > 0
        and punto_equilibrio < 999999
        else 0.0
    )

    # --- MATRIZ DE SENSIBILIDAD ---
    matriz_sensibilidad = []

    for p_m in [
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
    ]:
        fila = {
            "precio_mult": p_m,
            "valores": [],
        }

        for v_m in [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
        ]:
            res_sens = proyectar_escenario(
                datos.ventas.base * v_m,
                datos.ventas.crecimiento_mensual,
                p_m,
                1.0,
            )

            fila["valores"].append(
                {
                    "vol_mult": v_m,
                    "van": res_sens["van"],
                    "tir": res_sens["tir"],
                }
            )

        matriz_sensibilidad.append(
            fila
        )

    van_base = (
        escenario_base["proyecto"]["van"]
    )

    tir_base = (
        escenario_base["proyecto"]["tir"]
    )

    tributacion_base = (
        escenario_base["tributacion"]
    )

    # --- RIESGO Y SCORE ---
    prob_perdida = 0

    if (
        escenario_pesimista[
            "proyecto"
        ]["caja_final"]
        < 0
    ):
        prob_perdida += 35

    if (
        escenario_base[
            "proyecto"
        ]["caja_final"]
        < 0
    ):
        prob_perdida += 45

    if punto_equilibrio > datos.ventas.base:
        prob_perdida += 20

    score = 100 - prob_perdida

    if escenario_base["proyecto"]["roi"] < 10:
        score -= 20
    elif escenario_base["proyecto"]["roi"] > 50:
        score += 10

    payback_base = (
        escenario_base[
            "proyecto"
        ]["payback_meses"]
    )

    if payback_base is None:
        score -= 30
    elif payback_base > 18:
        score -= 15

    if margen_seguridad < 15:
        score -= 10

    if deficit_apertura > 0:
        score -= 25

    if cobertura_meses < 1:
        score -= 15
    elif cobertura_meses < max(
        0,
        int(datos.meses_reserva),
    ):
        score -= 5

    if not tributacion_base["compatible"]:
        score -= 20

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    decision_confiable = (
        deficit_apertura <= 0
        and tributacion_base["compatible"]
        and datos.precio_venta > 0
    )

    if deficit_apertura > 0:
        recomendacion = {
            "estado": "🔴 REPLANTEAR FINANCIAMIENTO",
            "msg": (
                "Los recursos disponibles no cubren la inversión inicial."
            ),
        }

    elif not tributacion_base["compatible"]:
        recomendacion = {
            "estado": "🟡 EVALUAR",
            "msg": (
                "La rentabilidad es solo referencial porque el régimen "
                "tributario seleccionado no es compatible con la proyección."
            ),
        }

    elif cobertura_meses < 1:
        recomendacion = {
            "estado": "🟡 EVALUAR",
            "msg": (
                "El proyecto puede abrir, pero queda con liquidez crítica. "
                "Refuerza capital de trabajo o reserva antes de invertir."
            ),
        }

    elif van_base <= 0:
        recomendacion = {
            "estado": "🔴 NO INVERTIR",
            "msg": (
                "El VAN del proyecto es no positivo bajo los supuestos actuales."
            ),
        }

    elif score >= 75:
        recomendacion = {
            "estado": "🟢 INVERTIR",
            "msg": (
                "El VAN es positivo, la tributación proyectada es compatible "
                "y el perfil de riesgo es favorable bajo los supuestos actuales."
            ),
        }

    else:
        recomendacion = {
            "estado": "🟡 EVALUAR",
            "msg": (
                "El proyecto genera valor, pero requiere revisar riesgo, "
                "liquidez o sensibilidad antes de decidir."
            ),
        }

    # --- MES EN QUE VENTAS ALCANZAN PUNTO DE EQUILIBRIO ---
    mes_alcanza_equilibrio: int | str = (
        "No alcanza"
    )

    for registro in escenario_base["p_y_g"]:
        if (
            registro["ventas_unidades"]
            >= punto_equilibrio
        ):
            mes_alcanza_equilibrio = (
                registro["mes"]
            )
            break

    ganancia_promedio_anio = round(
        (
            escenario_pesimista[
                "proyecto"
            ]["caja_mensual"][11]
            + escenario_base[
                "proyecto"
            ]["caja_mensual"][11]
            + escenario_optimista[
                "proyecto"
            ]["caja_mensual"][11]
        )
        / 3,
        2,
    )

    limitaciones_modelo = [
        (
            "La vida útil de 60 meses y residual de 10% son supuestos "
            "económicos por defecto para equipos sin detalle."
        ),
        (
            "La capacidad mensual se aproxima con las ventas optimistas "
            "de la plantilla."
        ),
        (
            "El precio de venta no aumenta automáticamente con inflación; "
            "costos variables y gastos fijos sí."
        ),
        (
            "La recuperación del capital de trabajo al mes 36 es un "
            "supuesto editable."
        ),
        (
            "Las reglas tributarias automáticas usan parámetros de "
            "referencia 2026 y no sustituyen asesoría contable o tributaria."
        ),
        (
            "En RMT/RG los pagos a cuenta se muestran como referencia, "
            "mientras el flujo usa una estimación simplificada del impuesto "
            "anual para evitar doble conteo."
        ),
        (
            "En RMT/RG no se modela de forma completa el efecto fiscal "
            "de los intereses de la deuda sobre el flujo del inversionista."
        ),
    ]

    advertencias_finales = list(
        dict.fromkeys(
            advertencias_generales
            + tributacion_base[
                "advertencias"
            ]
        )
    )

    return {
        "metricas": {
            "inversion_total": round(
                inversion_total,
                2,
            ),
            "capital_trabajo_inicial": round(
                capital_trabajo_inicial,
                2,
            ),
            "recuperacion_capital_trabajo": round(
                recuperacion_capital_trabajo,
                2,
            ),
            "valor_activos_inicial": round(
                valor_activos_inicial,
                2,
            ),
            "vida_util_equipos_default": (
                VIDA_UTIL_EQUIPOS_DEFAULT
            ),
            "valor_residual_activos_final": round(
                valor_residual_activos_final,
                2,
            ),
            "depreciacion_mensual": round(
                depreciacion_mensual,
                2,
            ),
            "valor_activos_mes_36": round(
                valor_activos_mes_36,
                2,
            ),
            "valor_terminal_total": round(
                valor_terminal_total,
                2,
            ),
            "tasa_descuento_anual": round(
                datos.tasa_descuento,
                4,
            ),
            "tasa_descuento_mensual": round(
                tasa_descuento_mensual
                * 100,
                4,
            ),
            "gastos_fijos": round(
                gastos_fijos_base,
                2,
            ),
            "margen_unitario": round(
                margen_unitario,
                2,
            ),
            "punto_equilibrio": (
                punto_equilibrio
            ),
            "punto_equilibrio_con_deuda": (
                punto_equilibrio_con_deuda
            ),
            "margen_seguridad": round(
                margen_seguridad,
                1,
            ),
            "reserva_emergencia": round(
                reserva_emergencia,
                2,
            ),
            "capital_invertible": round(
                capital_invertible,
                2,
            ),
            "alerta_liquidez": (
                alerta_liquidez
            ),
            "recursos_disponibles": round(
                recursos_disponibles,
                2,
            ),
            "deficit_apertura": round(
                deficit_apertura,
                2,
            ),
            "caja_despues_apertura": round(
                caja_despues_apertura,
                2,
            ),
            "deficit_reserva": round(
                deficit_reserva,
                2,
            ),
            "cobertura_meses": round(
                cobertura_meses,
                2,
            ),
            "estado_liquidez": (
                estado_liquidez
            ),
            # Compatibilidad V3.3:
            # métricas principales = PROYECTO.
            "van": van_base,
            "tir": tir_base,
            "roi": (
                escenario_base[
                    "proyecto"
                ]["roi"]
            ),
            "b_c": (
                escenario_base[
                    "proyecto"
                ]["b_c"]
            ),
            "payback_meses": (
                escenario_base[
                    "proyecto"
                ]["payback_meses"]
            ),
            "proyecto": {
                "van": escenario_base[
                    "proyecto"
                ]["van"],
                "tir": escenario_base[
                    "proyecto"
                ]["tir"],
                "tir_mensual": escenario_base[
                    "proyecto"
                ]["tir_mensual"],
                "tir_interpretable": escenario_base[
                    "proyecto"
                ]["tir_interpretable"],
                "alerta_tir": escenario_base[
                    "proyecto"
                ]["alerta_tir"],
                "roi": escenario_base[
                    "proyecto"
                ]["roi"],
                "b_c": escenario_base[
                    "proyecto"
                ]["b_c"],
                "payback_meses": escenario_base[
                    "proyecto"
                ]["payback_meses"],
                "inversion_inicial": escenario_base[
                    "proyecto"
                ]["inversion_inicial"],
                "caja_final": escenario_base[
                    "proyecto"
                ]["caja_final"],
                "mes_recuperacion": escenario_base[
                    "proyecto"
                ]["mes_recuperacion"],
            },
            "inversionista": {
                "metricas_validas": escenario_base[
                    "inversionista"
                ]["metricas_validas"],
                "aporte_propio": escenario_base[
                    "inversionista"
                ]["aporte_propio"],
                "aporte_propio_requerido": escenario_base[
                    "inversionista"
                ]["aporte_propio_requerido"],
                "prestamo": escenario_base[
                    "inversionista"
                ]["prestamo"],
                "van": escenario_base[
                    "inversionista"
                ]["van"],
                "tir": escenario_base[
                    "inversionista"
                ]["tir"],
                "tir_mensual": escenario_base[
                    "inversionista"
                ]["tir_mensual"],
                "tir_interpretable": escenario_base[
                    "inversionista"
                ]["tir_interpretable"],
                "alerta_tir": escenario_base[
                    "inversionista"
                ]["alerta_tir"],
                "roi": escenario_base[
                    "inversionista"
                ]["roi"],
                "b_c": escenario_base[
                    "inversionista"
                ]["b_c"],
                "payback_meses": escenario_base[
                    "inversionista"
                ]["payback_meses"],
                "caja_final": escenario_base[
                    "inversionista"
                ]["caja_final"],
                "mes_recuperacion": escenario_base[
                    "inversionista"
                ]["mes_recuperacion"],
            },
            "tributacion": {
                **tributacion_base,
                "parametros_referencia": {
                    "anio": datos.anio_tributario,
                    "uit": round(
                        float(datos.uit_referencia),
                        2,
                    ),
                    "igv_tasa_pct": 18.0,
                    "nrus_limite_mensual": NRUS_LIMITE_MENSUAL,
                    "nrus_limite_anual": NRUS_LIMITE_ANUAL,
                    "rer_limite_anual": RER_LIMITE_ANUAL,
                    "rmt_limite_uit": 1700,
                    "rmt_tramo_10_pct_uit": 15,
                },
            },
            "score": score,
            "decision_confiable": (
                decision_confiable
            ),
            "recomendacion": (
                recomendacion
            ),
            "prestamo": {
                "monto": round(
                    monto_financiar,
                    2,
                ),
                "cuota": round(
                    cuota_prestamo,
                    2,
                ),
                "cuota_mensual": round(
                    cuota_prestamo,
                    2,
                ),
            },
            "mes_alcanza_equilibrio": (
                mes_alcanza_equilibrio
            ),
            "dedicacion": {
                "horas_semana": (
                    horas_semana
                ),
                "porcentaje_presencial": (
                    presencia
                ),
            },
            "advertencias": (
                advertencias_finales
            ),
            "limitaciones_modelo": (
                limitaciones_modelo
            ),
        },
        "base": escenario_base,
        "pesimista": escenario_pesimista,
        "optimista": escenario_optimista,
        "riesgo": {
            "probabilidad_perdida": min(
                100,
                prob_perdida,
            ),
            "ganancia_promedio_anio": (
                ganancia_promedio_anio
            ),
        },
        "matriz_sensibilidad": (
            matriz_sensibilidad
        ),
    }


# ============================================================
# GEMINI - GOOGLE GEN AI SDK
# ============================================================


def _crear_cliente_gemini():
    if genai is None:
        raise RuntimeError(
            "Falta instalar el paquete google-genai. "
            "Ejecuta: pip install -r requirements.txt"
        )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "No se encontró GEMINI_API_KEY "
            "en las variables de entorno."
        )

    return genai.Client(
        api_key=api_key
    )


def _modelo_gemini() -> str:
    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash",
    )


def _mensaje_error_ia(error: Exception, contexto: str) -> str:
    """
    Convierte errores técnicos de Gemini en mensajes seguros y claros.
    El detalle completo queda en el log del servidor, no se expone al usuario.
    """
    error_str = str(error)
    error_lower = error_str.lower()

    logger.error(
        "Error Gemini en %s: %s",
        contexto,
        error_str,
    )

    if (
        "429" in error_lower
        or "quota" in error_lower
        or "resource_exhausted" in error_lower
        or "rate limit" in error_lower
    ):
        return (
            "⚠️ Se alcanzó temporalmente el límite de consultas de IA. "
            "Espera un momento y vuelve a intentarlo. "
            "El análisis financiero del simulador sigue disponible."
        )

    if (
        "503" in error_lower
        or "unavailable" in error_lower
        or "high demand" in error_lower
        or "overloaded" in error_lower
    ):
        return (
            "⚠️ El asistente de IA está temporalmente ocupado por alta demanda. "
            "Inténtalo nuevamente en unos momentos. "
            "El análisis financiero del simulador sigue disponible."
        )

    if (
        "api key" in error_lower
        or "api_key" in error_lower
        or "permission_denied" in error_lower
        or "401" in error_lower
        or "403" in error_lower
    ):
        return (
            "⚠️ La conexión con el asistente de IA requiere revisión de credenciales. "
            "El análisis financiero del simulador sigue disponible."
        )

    if (
        "google-genai" in error_lower
        or "no module named" in error_lower
    ):
        return (
            "⚠️ El módulo de IA del servidor no está disponible en este momento. "
            "El análisis financiero del simulador sigue disponible."
        )

    return (
        "⚠️ El asistente de IA no pudo responder en este momento. "
        "Vuelve a intentarlo más tarde. "
        "El análisis financiero del simulador sigue disponible."
    )


@app.post("/consejero")
async def obtener_consejo(datos: dict):
    client = None

    try:
        client = _crear_cliente_gemini()

        rol = datos.get("rol")
        metricas = datos.get(
            "metricas",
            {},
        )

        prompt = (
            f"Proyecto: {datos.get('idea')} "
            f"(Sector: {datos.get('sector')}).\n"
            f"VAN proyecto: "
            f"{metricas.get('van', 'N/A')}.\n"
            f"TIR proyecto: "
            f"{metricas.get('tir', 'N/A')}%.\n"
            f"ROI proyecto: "
            f"{metricas.get('roi', 'N/A')}%.\n"
            f"Liquidez: "
            f"{metricas.get('estado_liquidez', 'N/A')}.\n"
            f"Score: "
            f"{metricas.get('score', 'N/A')}/100.\n\n"
        )

        if rol == "auditor":
            prompt += (
                "Actúa como auditor financiero estricto. "
                "Prioriza VAN, flujo de caja, liquidez, "
                "sensibilidad y supuestos. Si la TIR es "
                "extraordinariamente alta, no la uses como "
                "único argumento. Dame 3 recomendaciones concretas."
            )

        elif rol == "marketing":
            prompt += (
                "Actúa como director de marketing. "
                "Diseña una estrategia práctica para validar "
                "y sostener las ventas base sin asumir "
                "crecimiento ilimitado."
            )

        elif rol == "operaciones":
            prompt += (
                "Actúa como jefe de operaciones. "
                "Detecta cuellos de botella de capacidad, "
                "liquidez y costos."
            )

        else:
            prompt += (
                "Actúa como asesor de negocios. "
                "Resume los principales riesgos y acciones."
            )

        response = await (
            client.aio.models.generate_content(
                model=_modelo_gemini(),
                contents=prompt,
            )
        )

        return {
            "consejo": (
                response.text
                or "Sin respuesta de IA."
            )
        }

    except Exception as e:
        return {
            "consejo": _mensaje_error_ia(
                e,
                "consejero",
            )
        }

    finally:
        if client is not None:
            try:
                await client.aio.aclose()
            except Exception:
                pass


@app.post("/chat")
async def chat_ia(datos: dict):
    client = None

    try:
        client = _crear_cliente_gemini()

        historial = datos.get(
            "history",
            [],
        )

        pregunta = datos.get(
            "question",
            "",
        )

        metricas = datos.get(
            "metricas",
            {},
        )

        contexto = (
            f"Negocio: {datos.get('idea')} "
            f"({datos.get('sector')}). "
            f"VAN proyecto: "
            f"{metricas.get('van', 'N/A')}. "
            f"TIR proyecto: "
            f"{metricas.get('tir', 'N/A')}%. "
            f"ROI proyecto: "
            f"{metricas.get('roi', 'N/A')}%. "
            f"Estado de liquidez: "
            f"{metricas.get('estado_liquidez', 'N/A')}. "
        )

        if genai_types is None:
            raise RuntimeError(
                "google-genai no está disponible."
            )

        contenidos = []

        for msg in historial:
            rol = (
                "user"
                if msg.get("role") == "user"
                else "model"
            )

            contenidos.append(
                genai_types.Content(
                    role=rol,
                    parts=[
                        genai_types.Part.from_text(
                            text=msg.get(
                                "content",
                                "",
                            )
                        )
                    ],
                )
            )

        contenidos.append(
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(
                        text=(
                            contexto
                            + "\n\nPregunta: "
                            + pregunta
                        )
                    )
                ],
            )
        )

        config = (
            genai_types.GenerateContentConfig(
                system_instruction=(
                    "Eres un asesor financiero y de negocios "
                    "de la plataforma Decisiones de Inversión IA. "
                    "No presentes una TIR extrema como prueba suficiente "
                    "de viabilidad. Prioriza VAN, caja, riesgo, liquidez, "
                    "sensibilidad y supuestos."
                )
            )
        )

        response = await (
            client.aio.models.generate_content(
                model=_modelo_gemini(),
                contents=contenidos,
                config=config,
            )
        )

        return {
            "respuesta": (
                response.text
                or "Sin respuesta de IA."
            )
        }

    except Exception as e:
        return {
            "respuesta": _mensaje_error_ia(
                e,
                "chat",
            )
        }

    finally:
        if client is not None:
            try:
                await client.aio.aclose()
            except Exception:
                pass
