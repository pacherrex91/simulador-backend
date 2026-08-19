from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import math
import os

from dotenv import load_dotenv
import google.generativeai as genai
import numpy_financial as npf

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health")
def health():
    return {"status": "ok"}


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
                categoria="Otros",
            ),
            ItemDinamico(
                id="legacy-permisos",
                nombre="Permisos",
                monto=inv.permisos,
                categoria="Otros",
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


def _calcular_tir_anual(flujos: list[float]) -> Optional[float]:
    try:
        tir_mensual = float(npf.irr(flujos))

        if not math.isfinite(tir_mensual) or tir_mensual <= -1:
            return None

        return ((1 + tir_mensual) ** 12 - 1) * 100
    except Exception:
        return None


def _calcular_beneficio_costo(
    tasa_mensual: float,
    flujos: list[float],
) -> float:
    beneficios_vp = 0.0
    costos_vp = 0.0

    for periodo, flujo in enumerate(flujos):
        factor = (1 + tasa_mensual) ** periodo

        if factor <= 0:
            continue

        valor_presente = flujo / factor

        if valor_presente > 0:
            beneficios_vp += valor_presente
        elif valor_presente < 0:
            costos_vp += abs(valor_presente)

    return beneficios_vp / costos_vp if costos_vp > 0 else 0.0


@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_items = _normalizar_inversion(datos)
    gasto_items = _normalizar_gastos(datos)

    inversion_total = sum(
        max(0.0, float(item.monto))
        for item in inversion_items
    )
    gastos_fijos_base = sum(
        max(0.0, float(item.monto))
        for item in gasto_items
    )

    # --- DEPRECIACIÓN ECONÓMICA DE ACTIVOS ---
    VIDA_UTIL_EQUIPOS_DEFAULT = 60
    RESIDUAL_EQUIPOS_DEFAULT = 0.10

    depreciacion_mensual = 0.0
    valor_activos_inicial = 0.0
    valor_residual_activos_final = 0.0
    valor_activos_mes_36 = 0.0

    for item in inversion_items:
        monto_item = max(0.0, float(item.monto))
        categoria = item.categoria.strip().lower()
        nombre = item.nombre.strip().lower()

        es_equipo = (
            "equipo" in categoria
            or "equipo" in nombre
            or "maquinaria" in categoria
            or "maquinaria" in nombre
        )

        vida_util_efectiva = item.vida_util
        residual_efectivo = item.residual

        if es_equipo and vida_util_efectiva <= 0:
            vida_util_efectiva = VIDA_UTIL_EQUIPOS_DEFAULT

            if residual_efectivo <= 0:
                residual_efectivo = monto_item * RESIDUAL_EQUIPOS_DEFAULT

        if vida_util_efectiva > 0:
            residual_efectivo = max(
                0.0,
                min(float(residual_efectivo), monto_item),
            )

            base_depreciable = max(
                0.0,
                monto_item - residual_efectivo,
            )

            depreciacion_item = (
                base_depreciable / vida_util_efectiva
            )

            depreciacion_mensual += depreciacion_item
            valor_activos_inicial += monto_item
            valor_residual_activos_final += residual_efectivo

            meses_depreciados = min(36, vida_util_efectiva)

            valor_item_mes_36 = max(
                residual_efectivo,
                monto_item - (depreciacion_item * meses_depreciados),
            )

            valor_activos_mes_36 += valor_item_mes_36

    # --- FINANCIAMIENTO ---
    monto_financiar = max(0.0, datos.financiamiento_monto)
    tasa_prestamo = max(
        0.0,
        datos.financiamiento_tasa_mensual / 100,
    )
    plazo = max(0, datos.financiamiento_plazo)

    cuota_prestamo = 0.0

    if monto_financiar > 0 and plazo > 0:
        if tasa_prestamo > 0:
            cuota_prestamo = (
                monto_financiar
                * (
                    tasa_prestamo
                    * (1 + tasa_prestamo) ** plazo
                )
                / (
                    (1 + tasa_prestamo) ** plazo - 1
                )
            )
        else:
            cuota_prestamo = monto_financiar / plazo

    # --- MES 0: APERTURA Y LIQUIDEZ ---
    recursos_disponibles = (
        max(0.0, datos.capital_disponible)
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
        * max(0, datos.meses_reserva)
    )

    deficit_reserva = max(
        0.0,
        reserva_emergencia - caja_despues_apertura,
    )

    cobertura_meses = (
        caja_despues_apertura / gastos_fijos_base
        if gastos_fijos_base > 0
        else 0.0
    )

    if deficit_apertura > 0:
        estado_liquidez = "🔴 FINANCIAMIENTO INSUFICIENTE"
        alerta_liquidez = (
            f"Faltan {datos.moneda} {deficit_apertura:,.2f} "
            "para financiar la apertura."
        )

    elif cobertura_meses < 1:
        estado_liquidez = "🟠 LIQUIDEZ CRÍTICA"
        alerta_liquidez = (
            f"Puedes abrir, pero la caja restante cubre solo "
            f"{cobertura_meses:.1f} meses de gastos fijos."
        )

    elif cobertura_meses < max(0, datos.meses_reserva):
        estado_liquidez = "🟡 LIQUIDEZ AJUSTADA"
        alerta_liquidez = (
            f"Puedes abrir, pero aún faltan "
            f"{datos.moneda} {deficit_reserva:,.2f} "
            "para alcanzar la reserva objetivo."
        )

    else:
        estado_liquidez = "🟢 LIQUIDEZ SALUDABLE"
        alerta_liquidez = (
            "La apertura y la reserva objetivo están cubiertas."
        )

    # Compatibilidad con la V3.3.
    capital_invertible = caja_despues_apertura

    # El proyecto cuesta inversion_total sin importar
    # si se financia con capital propio o deuda.
    prestamo_aplicado_inversion = min(
        monto_financiar,
        inversion_total,
    )

    aporte_propio_requerido = max(
        0.0,
        inversion_total - prestamo_aplicado_inversion,
    )

    aporte_propio_efectivo = min(
        max(0.0, datos.capital_disponible),
        aporte_propio_requerido,
    )

    metricas_inversionista_validas = (
        deficit_apertura <= 0
        and aporte_propio_efectivo > 0
    )

    # --- PUNTO DE EQUILIBRIO ---
    margen_unitario = datos.precio_venta - datos.costo_directo

    punto_equilibrio = (
        999999
        if margen_unitario <= 0
        else int(gastos_fijos_base / margen_unitario) + 1
    )

    punto_equilibrio_con_deuda = (
        999999
        if margen_unitario <= 0
        else int(
            (gastos_fijos_base + cuota_prestamo)
            / margen_unitario
        )
        + 1
    )

    margen_seguridad = (
        max(
            0,
            (
                (datos.ventas.base - punto_equilibrio)
                / datos.ventas.base
            )
            * 100,
        )
        if datos.ventas.base > 0
        else 0
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

    sueldo_emprendedor = 0.0

    if datos.gastos_fijos:
        sueldo_emprendedor = (
            datos.gastos_fijos.sueldo_emprendedor
        )
    else:
        for item in gasto_items:
            if (
                "sueldo" in item.nombre.lower()
                or item.categoria.lower() == "personal"
            ):
                sueldo_emprendedor += item.monto

    if sueldo_emprendedor > 2000:
        horas_semana = min(80, horas_semana + 15)
        presencia = min(100, presencia + 10)

    tasa_descuento_mensual = _tasa_mensual_equivalente(
        datos.tasa_descuento
    )

    # --- PROYECCIÓN DE ESCENARIOS ---
    def proyectar_escenario(
        ventas_iniciales: float,
        crecimiento: float,
        precio_mult: float = 1.0,
        costo_mult: float = 1.0,
    ):
        # PROYECTO: parte de toda la inversión inicial.
        caja_acumulada_proyecto = -inversion_total

        # INVERSIONISTA: parte solo de su aporte propio.
        caja_acumulada_inversionista = (
            -aporte_propio_efectivo
        )

        flujo_proyecto_mensual: list[float] = []
        flujo_inversionista_mensual: list[float] = []

        caja_mensual_proyecto: list[float] = []
        caja_mensual_inversionista: list[float] = []

        p_y_g: list[dict] = []

        mes_recuperacion_proyecto: int | str = "No recupera"
        mes_recuperacion_inversionista: int | str = "No recupera"

        precio_final = datos.precio_venta * precio_mult
        costo_final = datos.costo_directo * costo_mult

        ingresos_totales = 0.0
        utilidad_neta_total = 0.0

        saldo_prestamo = monto_financiar

        for mes in range(1, 37):
            idx_estacional = (mes - 1) % 12

            ajuste_estacional = (
                datos.estacionalidad[idx_estacional]
                if idx_estacional < len(datos.estacionalidad)
                else 0.0
            )

            factor_estacional = (
                1.0 + (ajuste_estacional / 100.0)
            )

            # Capacidad mensual de referencia:
            # ventas optimistas de la plantilla.
            capacidad_ventas = max(
                0.0,
                float(datos.ventas.optimista),
            )

            ventas_tendencia = (
                ventas_iniciales
                * (
                    (1 + (crecimiento / 100))
                    ** (mes - 1)
                )
            )

            ventas_mes = min(
                ventas_tendencia * factor_estacional,
                capacidad_ventas,
            )

            # Inflación anual -> mensual equivalente.
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

            # Precio sin aumento automático.
            ingresos = ventas_mes * precio_final

            # Costos variables con inflación.
            costo_unitario_inflado = (
                costo_final * factor_inflacion
            )

            costos_variables = (
                ventas_mes * costo_unitario_inflado
            )

            margen_bruto = ingresos - costos_variables

            # Gastos fijos con inflación.
            gastos_fijos_inflados = (
                gastos_fijos_base * factor_inflacion
            )

            ebit = (
                margen_bruto
                - gastos_fijos_inflados
                - depreciacion_mensual
            )

            # TEMPORAL:
            # La lógica tributaria se mantiene como estaba.
            # Se corregirá en el paso tributario específico.
            if datos.regimen_tributario == "NRUS":
                impuestos = 20 if ingresos <= 5000 else 50
            elif datos.regimen_tributario == "RER":
                impuestos = ingresos * 0.015
            else:
                impuestos = max(0, ebit * 0.295)

            utilidad_neta = ebit - impuestos

            # Flujo económico del PROYECTO:
            # no resta intereses ni principal.
            flujo_caja_proyecto = (
                utilidad_neta + depreciacion_mensual
            )

            # --- CRONOGRAMA DEL PRÉSTAMO ---
            pago_prestamo = 0.0
            interes_prestamo = 0.0
            amortizacion_prestamo = 0.0

            if (
                mes <= plazo
                and saldo_prestamo > 0
                and cuota_prestamo > 0
            ):
                interes_prestamo = (
                    saldo_prestamo * tasa_prestamo
                )

                amortizacion_prestamo = max(
                    0.0,
                    cuota_prestamo - interes_prestamo,
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

            # Flujo del INVERSIONISTA:
            # proyecto después del servicio de deuda.
            flujo_caja_inversionista = (
                flujo_caja_proyecto - pago_prestamo
            )

            # Valor terminal: una sola vez en mes 36.
            if mes == 36:
                flujo_caja_proyecto += valor_activos_mes_36
                flujo_caja_inversionista += valor_activos_mes_36

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
                round(caja_acumulada_proyecto, 2)
            )

            caja_mensual_inversionista.append(
                round(caja_acumulada_inversionista, 2)
            )

            ingresos_totales += ingresos
            utilidad_neta_total += utilidad_neta

            p_y_g.append(
                {
                    "mes": mes,
                    "ventas_unidades": round(ventas_mes, 1),
                    "ingresos": round(ingresos, 2),
                    "costos_variables": round(
                        costos_variables,
                        2,
                    ),
                    "margen_bruto": round(margen_bruto, 2),
                    "gastos_fijos": round(
                        gastos_fijos_inflados,
                        2,
                    ),
                    "depreciacion": round(
                        depreciacion_mensual,
                        2,
                    ),
                    "ebit": round(ebit, 2),
                    "impuestos": round(impuestos, 2),
                    "utilidad_neta": round(
                        utilidad_neta,
                        2,
                    ),
                    # Compatibilidad: el flujo principal
                    # ahora representa al proyecto.
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

            if (
                caja_acumulada_proyecto >= 0
                and mes_recuperacion_proyecto == "No recupera"
            ):
                mes_recuperacion_proyecto = mes

            if (
                caja_acumulada_inversionista >= 0
                and mes_recuperacion_inversionista == "No recupera"
            ):
                mes_recuperacion_inversionista = mes

        # --- MÉTRICAS DEL PROYECTO ---
        flujos_proyecto = [
            -inversion_total
        ] + flujo_proyecto_mensual

        van_proyecto = _calcular_van(
            tasa_descuento_mensual,
            flujos_proyecto,
        )

        tir_proyecto = _calcular_tir_anual(
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

        b_c_proyecto = _calcular_beneficio_costo(
            tasa_descuento_mensual,
            flujos_proyecto,
        )

        # --- MÉTRICAS DEL INVERSIONISTA ---
        flujos_inversionista = [
            -aporte_propio_efectivo
        ] + flujo_inversionista_mensual

        van_inversionista: Optional[float] = None
        tir_inversionista: Optional[float] = None
        roi_inversionista: Optional[float] = None
        b_c_inversionista: Optional[float] = None

        if metricas_inversionista_validas:
            van_inversionista = _calcular_van(
                tasa_descuento_mensual,
                flujos_inversionista,
            )

            tir_inversionista = _calcular_tir_anual(
                flujos_inversionista
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

            b_c_inversionista = _calcular_beneficio_costo(
                tasa_descuento_mensual,
                flujos_inversionista,
            )

        margen_neto = (
            (
                utilidad_neta_total
                / ingresos_totales
            )
            * 100
            if ingresos_totales > 0
            else 0.0
        )

        return {
            # Compatibilidad V3.3:
            # los campos principales pasan a ser del PROYECTO.
            "caja_mensual": caja_mensual_proyecto,
            "caja_mes_a_mes": caja_mensual_proyecto,
            "p_y_g": p_y_g,
            "van": round(van_proyecto, 2),
            "tir": (
                round(tir_proyecto, 2)
                if tir_proyecto is not None
                else None
            ),
            "roi": round(roi_proyecto, 2),
            "b_c": round(b_c_proyecto, 2),
            "caja_final": round(
                caja_acumulada_proyecto,
                2,
            ),
            "mes_recuperacion": (
                mes_recuperacion_proyecto
            ),
            "margen_neto": round(margen_neto, 2),

            # Nuevos datos separados.
            "proyecto": {
                "van": round(van_proyecto, 2),
                "tir": (
                    round(tir_proyecto, 2)
                    if tir_proyecto is not None
                    else None
                ),
                "roi": round(roi_proyecto, 2),
                "b_c": round(b_c_proyecto, 2),
                "inversion_inicial": round(
                    inversion_total,
                    2,
                ),
                "caja_final": round(
                    caja_acumulada_proyecto,
                    2,
                ),
                "mes_recuperacion": (
                    mes_recuperacion_proyecto
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
                    round(van_inversionista, 2)
                    if van_inversionista is not None
                    else None
                ),
                "tir": (
                    round(tir_inversionista, 2)
                    if tir_inversionista is not None
                    else None
                ),
                "roi": (
                    round(roi_inversionista, 2)
                    if roi_inversionista is not None
                    else None
                ),
                "b_c": (
                    round(b_c_inversionista, 2)
                    if b_c_inversionista is not None
                    else None
                ),
                "caja_final": round(
                    caja_acumulada_inversionista,
                    2,
                ),
                "mes_recuperacion": (
                    mes_recuperacion_inversionista
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
        0,
    )

    # Al iniciar ya en la capacidad optimista,
    # el límite impide crecimiento adicional.
    escenario_optimista = proyectar_escenario(
        datos.ventas.optimista,
        0,
    )

    # --- MATRIZ DE SENSIBILIDAD ---
    matriz_sensibilidad = []

    for p_m in [0.8, 0.9, 1.0, 1.1, 1.2]:
        fila = {
            "precio_mult": p_m,
            "valores": [],
        }

        for v_m in [0.8, 0.9, 1.0, 1.1, 1.2]:
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

        matriz_sensibilidad.append(fila)

    van_base = escenario_base["proyecto"]["van"]
    tir_base = escenario_base["proyecto"]["tir"]

    # --- RIESGO Y SCORE ---
    prob_perdida = 0

    if escenario_pesimista["proyecto"]["caja_final"] < 0:
        prob_perdida += 35

    if escenario_base["proyecto"]["caja_final"] < 0:
        prob_perdida += 45

    if punto_equilibrio > datos.ventas.base:
        prob_perdida += 20

    score = 100 - prob_perdida

    if escenario_base["proyecto"]["roi"] < 10:
        score -= 20
    elif escenario_base["proyecto"]["roi"] > 50:
        score += 10

    if isinstance(
        escenario_base["proyecto"]["mes_recuperacion"],
        int,
    ):
        if (
            escenario_base["proyecto"]["mes_recuperacion"]
            > 18
        ):
            score -= 15
    else:
        score -= 30

    if margen_seguridad < 15:
        score -= 10

    if deficit_apertura > 0:
        score -= 20

    score = max(0, min(100, score))

    if score >= 75 and van_base > 0:
        recomendacion = {
            "estado": "🟢 INVERTIR",
            "msg": (
                "VAN del proyecto positivo y Score alto. "
                "Genera valor bajo los supuestos actuales."
            ),
        }
    elif score >= 45 or van_base > -2000:
        recomendacion = {
            "estado": "🟡 EVALUAR",
            "msg": (
                "Riesgo moderado o VAN ajustado. "
                "Optimiza costos y valida supuestos."
            ),
        }
    else:
        recomendacion = {
            "estado": "🔴 NO INVERTIR",
            "msg": (
                "El proyecto destruye valor bajo "
                "los supuestos actuales."
            ),
        }

    # --- MES EN QUE LAS VENTAS ALCANZAN EL PUNTO DE EQUILIBRIO ---
    mes_alcanza_equilibrio: int | str = "No alcanza"

    for mes in range(1, 37):
        idx_estacional = (mes - 1) % 12

        ajuste_estacional = (
            datos.estacionalidad[idx_estacional]
            if idx_estacional < len(datos.estacionalidad)
            else 0.0
        )

        ventas_tendencia = (
            datos.ventas.base
            * (
                (1 + datos.ventas.crecimiento_mensual / 100)
                ** (mes - 1)
            )
        )

        ventas_mes = min(
            ventas_tendencia
            * (1 + ajuste_estacional / 100),
            max(0.0, float(datos.ventas.optimista)),
        )

        if ventas_mes >= punto_equilibrio:
            mes_alcanza_equilibrio = mes
            break

    ganancia_promedio_anio = round(
        (
            escenario_pesimista["proyecto"]["caja_mensual"][11]
            + escenario_base["proyecto"]["caja_mensual"][11]
            + escenario_optimista["proyecto"]["caja_mensual"][11]
        )
        / 3,
        2,
    )

    tasa_descuento_mensual_pct = (
        tasa_descuento_mensual * 100
    )

    return {
        "metricas": {
            "inversion_total": round(inversion_total, 2),

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

            "tasa_descuento_anual": round(
                datos.tasa_descuento,
                4,
            ),
            "tasa_descuento_mensual": round(
                tasa_descuento_mensual_pct,
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
            "punto_equilibrio": punto_equilibrio,
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
            "alerta_liquidez": alerta_liquidez,
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
            "estado_liquidez": estado_liquidez,

            # Compatibilidad V3.3:
            # estos cuatro son ahora métricas DEL PROYECTO.
            "van": van_base,
            "tir": tir_base,
            "roi": escenario_base["proyecto"]["roi"],
            "b_c": escenario_base["proyecto"]["b_c"],

            "proyecto": {
                "van": escenario_base["proyecto"]["van"],
                "tir": escenario_base["proyecto"]["tir"],
                "roi": escenario_base["proyecto"]["roi"],
                "b_c": escenario_base["proyecto"]["b_c"],
                "inversion_inicial": (
                    escenario_base["proyecto"]["inversion_inicial"]
                ),
                "caja_final": (
                    escenario_base["proyecto"]["caja_final"]
                ),
                "mes_recuperacion": (
                    escenario_base["proyecto"]["mes_recuperacion"]
                ),
            },

            "inversionista": {
                "metricas_validas": (
                    escenario_base["inversionista"]["metricas_validas"]
                ),
                "aporte_propio": (
                    escenario_base["inversionista"]["aporte_propio"]
                ),
                "aporte_propio_requerido": (
                    escenario_base["inversionista"][
                        "aporte_propio_requerido"
                    ]
                ),
                "prestamo": (
                    escenario_base["inversionista"]["prestamo"]
                ),
                "van": (
                    escenario_base["inversionista"]["van"]
                ),
                "tir": (
                    escenario_base["inversionista"]["tir"]
                ),
                "roi": (
                    escenario_base["inversionista"]["roi"]
                ),
                "b_c": (
                    escenario_base["inversionista"]["b_c"]
                ),
                "caja_final": (
                    escenario_base["inversionista"]["caja_final"]
                ),
                "mes_recuperacion": (
                    escenario_base["inversionista"][
                        "mes_recuperacion"
                    ]
                ),
            },

            "score": score,
            "recomendacion": recomendacion,

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
                "horas_semana": horas_semana,
                "porcentaje_presencial": presencia,
            },
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

        "matriz_sensibilidad": matriz_sensibilidad,
    }


@app.post("/consejero")
async def obtener_consejo(datos: dict):
    try:
        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        modelo = genai.GenerativeModel(
            "gemini-3.5-flash"
        )

        rol = datos.get("rol")
        metricas = datos.get("metricas", {})

        prompt = (
            f"Proyecto: {datos.get('idea')} "
            f"(Sector: {datos.get('sector')}).\n"
            f"VAN: {metricas.get('van', 'N/A')}. "
            f"TIR: {metricas.get('tir', 'N/A')}%. "
            f"Score: {metricas.get('score', 'N/A')}/100.\n\n"
        )

        if rol == "auditor":
            prompt += (
                "Actúa como auditor financiero estricto. "
                "Dame 3 consejos concretos para reducir "
                "riesgos y mejorar VAN/TIR."
            )

        elif rol == "marketing":
            prompt += (
                "Actúa como director de marketing. "
                "Diseña una estrategia rápida para asegurar "
                "las ventas base."
            )

        elif rol == "operaciones":
            prompt += (
                "Actúa como jefe de operaciones. "
                "Detecta posibles cuellos de botella "
                "con este nivel de liquidez."
            )

        respuesta = modelo.generate_content(prompt)

        return {
            "consejo": respuesta.text
        }

    except Exception as e:
        error_str = str(e)

        if (
            "429" in error_str
            or "quota" in error_str.lower()
        ):
            return {
                "consejo": (
                    "⚠️ **Límite de consultas alcanzado.**\n\n"
                    "Espera 60 segundos y vuelve a intentarlo."
                )
            }

        return {
            "consejo": (
                f"Error de conexión IA: {error_str}"
            )
        }


@app.post("/chat")
async def chat_ia(datos: dict):
    try:
        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        modelo = genai.GenerativeModel(
            "gemini-3.5-flash"
        )

        historial = datos.get("history", [])
        pregunta = datos.get("question", "")

        contexto = (
            f"Contexto del negocio: "
            f"{datos.get('idea')} "
            f"({datos.get('sector')}). "
            f"ROI: "
            f"{datos.get('metricas', {}).get('roi')}%. "
        )

        mensajes = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Eres un experto asesor financiero "
                            "y de negocios de la plataforma "
                            "Decisiones de Inversión IA."
                        )
                    }
                ],
            }
        ]

        for msg in historial:
            rol = (
                "user"
                if msg.get("role") == "user"
                else "model"
            )

            mensajes.append(
                {
                    "role": rol,
                    "parts": [
                        {
                            "text": msg.get(
                                "content",
                                "",
                            )
                        }
                    ],
                }
            )

        mensajes.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": contexto + pregunta
                    }
                ],
            }
        )

        respuesta = modelo.generate_content(
            mensajes
        )

        return {
            "respuesta": respuesta.text
        }

    except Exception as e:
        error_str = str(e)

        if (
            "429" in error_str
            or "quota" in error_str.lower()
        ):
            return {
                "respuesta": (
                    "⚠️ Límite de consultas gratuitas superado. "
                    "Espera 60 segundos antes de enviar "
                    "otro mensaje."
                )
            }

        return {
            "respuesta": (
                f"Error en el chat: {error_str}"
            )
        }
