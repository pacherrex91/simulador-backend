from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
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
            ItemDinamico(id="legacy-insumos", nombre="Insumos", monto=inv.insumos, categoria="Insumos"),
            ItemDinamico(id="legacy-equipos", nombre="Equipos", monto=inv.equipos, categoria="Equipos"),
            ItemDinamico(id="legacy-empaques", nombre="Empaques", monto=inv.empaques, categoria="Otros"),
            ItemDinamico(id="legacy-permisos", nombre="Permisos", monto=inv.permisos, categoria="Otros"),
            ItemDinamico(id="legacy-otros", nombre="Otros", monto=inv.otros, categoria="Otros"),
        ]

    return []


def _normalizar_gastos(datos: DatosSimulacion) -> list[ItemDinamico]:
    if datos.gastos_dinamicos:
        return datos.gastos_dinamicos

    if datos.gastos_fijos:
        gf = datos.gastos_fijos
        return [
            ItemDinamico(id="legacy-marketing", nombre="Marketing", monto=gf.marketing, categoria="Marketing"),
            ItemDinamico(id="legacy-logistica", nombre="Logística", monto=gf.logistica, categoria="Proveedores"),
            ItemDinamico(
                id="legacy-sueldo",
                nombre="Sueldo Emprendedor",
                monto=gf.sueldo_emprendedor,
                categoria="Personal",
            ),
            ItemDinamico(id="legacy-otros", nombre="Otros Fijos", monto=gf.otros, categoria="Otros"),
        ]

    return []


@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_items = _normalizar_inversion(datos)
    gasto_items = _normalizar_gastos(datos)

    inversion_total = sum(item.monto for item in inversion_items)
    gastos_fijos_base = sum(item.monto for item in gasto_items)

    depreciacion_mensual = 0.0
    for item in inversion_items:
        if item.vida_util > 0:
            depreciacion_mensual += (item.monto - item.residual) / item.vida_util

    cuota_prestamo = 0.0
    tasa = datos.financiamiento_tasa_mensual / 100
    plazo = max(0, datos.financiamiento_plazo)
    monto_financiar = max(0.0, datos.financiamiento_monto)

    if monto_financiar > 0 and plazo > 0:
        if tasa > 0:
            cuota_prestamo = (
                monto_financiar
                * (tasa * (1 + tasa) ** plazo)
                / ((1 + tasa) ** plazo - 1)
            )
        else:
            cuota_prestamo = monto_financiar / plazo

        # --- MES 0: APERTURA Y LIQUIDEZ ---
    recursos_disponibles = datos.capital_disponible + monto_financiar

    deficit_apertura = max(
        0.0,
        inversion_total - recursos_disponibles,
    )

    caja_despues_apertura = max(
        0.0,
        recursos_disponibles - inversion_total,
    )

    reserva_emergencia = (
        gastos_fijos_base * max(0, datos.meses_reserva)
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

    # Se mantiene este nombre por compatibilidad con la V3.3 actual.
    capital_invertible = caja_despues_apertura

    margen_unitario = datos.precio_venta - datos.costo_directo
    punto_equilibrio = (
        999999
        if margen_unitario <= 0
        else int((gastos_fijos_base + cuota_prestamo) / margen_unitario) + 1
    )
    margen_seguridad = (
        max(0, ((datos.ventas.base - punto_equilibrio) / datos.ventas.base) * 100)
        if datos.ventas.base > 0
        else 0
    )

    sector_str = datos.sector.lower()
    es_digital = any(
        keyword in sector_str
        for keyword in ["tech", "e-commerce", "digital", "online", "software", "web", "tecnología"]
    )
    es_alimentos = any(
        keyword in sector_str
        for keyword in ["gastro", "alimento", "restaurante", "food", "cafeteria", "cafetería"]
    )

    if es_digital:
        horas_semana, presencia = 25, 5
    elif es_alimentos:
        horas_semana, presencia = 60, 90
    else:
        horas_semana, presencia = 45, 70

    sueldo_emprendedor = 0.0
    if datos.gastos_fijos:
        sueldo_emprendedor = datos.gastos_fijos.sueldo_emprendedor
    else:
        for item in gasto_items:
            if "sueldo" in item.nombre.lower() or item.categoria.lower() == "personal":
                sueldo_emprendedor += item.monto

    if sueldo_emprendedor > 2000:
        horas_semana = min(80, horas_semana + 15)
        presencia = min(100, presencia + 10)

    def proyectar_escenario(
        ventas_iniciales: float,
        crecimiento: float,
        precio_mult: float = 1.0,
        costo_mult: float = 1.0,
    ):
        capital_propio_invertido = min(inversion_total, datos.capital_disponible)
        caja_acumulada = -capital_propio_invertido
        flujo_neto_mensual: list[float] = []
        caja_mensual: list[float] = []
        p_y_g: list[dict] = []

        mes_recuperacion: int | str = "No recupera"
        precio_final = datos.precio_venta * precio_mult
        costo_final = datos.costo_directo * costo_mult
        ingresos_totales = 0.0
        utilidad_neta_total = 0.0

        for mes in range(1, 37):
            idx_estacional = (mes - 1) % 12
            ajuste_estacional = (
                datos.estacionalidad[idx_estacional]
                if idx_estacional < len(datos.estacionalidad)
                else 0.0
            )
            factor_estacional = 1.0 + (ajuste_estacional / 100.0)

                       # Crecimiento con límite de capacidad.
            # Las ventas optimistas de la plantilla funcionan
            # como capacidad mensual de referencia.
            capacidad_ventas = max(0.0, float(datos.ventas.optimista))

            ventas_tendencia = (
                ventas_iniciales
                * ((1 + (crecimiento / 100)) ** (mes - 1))
            )

            ventas_mes = min(
                ventas_tendencia * factor_estacional,
                capacidad_ventas,
            )
                       # Inflación mensual equivalente a partir de la inflación anual.
            inflacion_anual_decimal = datos.inflacion_anual / 100.0
            inflacion_mensual = (
                (1 + inflacion_anual_decimal) ** (1 / 12) - 1
            )

            factor_inflacion = (
                (1 + inflacion_mensual) ** (mes - 1)
            )

            # El precio de venta se mantiene sin aumento automático.
            ingresos = ventas_mes * precio_final

            # Los costos variables sí aumentan con la inflación.
            costo_unitario_inflado = costo_final * factor_inflacion
            costos_variables = ventas_mes * costo_unitario_inflado

            margen_bruto = ingresos - costos_variables

            # Los gastos fijos también aumentan con la inflación.
            gastos_fijos_inflados = gastos_fijos_base * factor_inflacion

            ebit = margen_bruto - gastos_fijos_inflados - depreciacion_mensual

            if datos.regimen_tributario == "NRUS":
                impuestos = 20 if ingresos <= 5000 else 50
            elif datos.regimen_tributario == "RER":
                impuestos = ingresos * 0.015
            else:
                impuestos = max(0, ebit * 0.295)

            utilidad_neta = ebit - impuestos
            cuota_mes = cuota_prestamo if mes <= plazo else 0
            flujo_caja = utilidad_neta + depreciacion_mensual - cuota_mes

            flujo_neto_mensual.append(flujo_caja)
            caja_acumulada += flujo_caja
            caja_mensual.append(round(caja_acumulada, 2))

            ingresos_totales += ingresos
            utilidad_neta_total += utilidad_neta

            p_y_g.append(
                {
                    "mes": mes,
                    "ventas_unidades": round(ventas_mes, 1),
                    "ingresos": round(ingresos, 2),
                    "costos_variables": round(costos_variables, 2),
                    "margen_bruto": round(margen_bruto, 2),
                    "gastos_fijos": round(gastos_fijos_inflados, 2),
                    "depreciacion": round(depreciacion_mensual, 2),
                    "ebit": round(ebit, 2),
                    "impuestos": round(impuestos, 2),
                    "utilidad_neta": round(utilidad_neta, 2),
                    "flujo_caja_neto": round(flujo_caja, 2),
                    "caja_acumulada": round(caja_acumulada, 2),
                }
            )

            if caja_acumulada >= 0 and mes_recuperacion == "No recupera":
                mes_recuperacion = mes

        tasa_mensual = (datos.tasa_descuento / 100) / 12
        flujos_para_van = [-capital_propio_invertido] + flujo_neto_mensual

        try:
            van = float(npf.npv(tasa_mensual, flujos_para_van))
        except Exception:
            van = 0.0

        try:
            tir_mensual = float(npf.irr(flujos_para_van))
            tir = ((1 + tir_mensual) ** 12 - 1) * 100 if tir_mensual > -1 else -100
        except Exception:
            tir = -100.0

        roi = (
            (caja_acumulada / capital_propio_invertido) * 100
            if capital_propio_invertido > 0
            else 0.0
        )
        total_salidas = sum(abs(f) for f in flujos_para_van if f < 0)
        b_c = (
            sum(f for f in flujo_neto_mensual if f > 0) / total_salidas
            if total_salidas > 0
            else 0.0
        )
        margen_neto = (
            (utilidad_neta_total / ingresos_totales) * 100
            if ingresos_totales > 0
            else 0.0
        )

        return {
            "caja_mensual": caja_mensual,
            "p_y_g": p_y_g,
            "van": round(van, 2),
            "tir": round(tir, 2),
            "roi": round(roi, 2),
            "b_c": round(b_c, 2),
            "caja_mes_a_mes": caja_mensual,
            "margen_neto": round(margen_neto, 2),
            "caja_final": round(caja_acumulada, 2),
            "mes_recuperacion": mes_recuperacion,
        }

    escenario_base = proyectar_escenario(
        datos.ventas.base, datos.ventas.crecimiento_mensual
    )
    escenario_pesimista = proyectar_escenario(datos.ventas.pesimista, 0)
    escenario_optimista = proyectar_escenario(
        datos.ventas.optimista, datos.ventas.crecimiento_mensual * 1.5
    )

    matriz_sensibilidad = []
    for p_m in [0.8, 0.9, 1.0, 1.1, 1.2]:
        fila = {"precio_mult": p_m, "valores": []}
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

    van_base = escenario_base["van"]
    tir_base = escenario_base["tir"]

    prob_perdida = 0
    if escenario_pesimista["caja_final"] < 0:
        prob_perdida += 35
    if escenario_base["caja_final"] < 0:
        prob_perdida += 45
    if punto_equilibrio > datos.ventas.base:
        prob_perdida += 20

    score = 100 - prob_perdida
    if escenario_base["roi"] < 10:
        score -= 20
    elif escenario_base["roi"] > 50:
        score += 10

    if isinstance(escenario_base["mes_recuperacion"], int):
        if escenario_base["mes_recuperacion"] > 18:
            score -= 15
    else:
        score -= 30

    if margen_seguridad < 15:
        score -= 10
    if inversion_total > datos.capital_disponible and monto_financiar == 0:
        score -= 20

    score = max(0, min(100, score))

    if score >= 75 and van_base > 0:
        recomendacion = {
            "estado": "🟢 INVERTIR",
            "msg": "VAN positivo y Score alto. Genera valor.",
        }
    elif score >= 45 or van_base > -2000:
        recomendacion = {
            "estado": "🟡 EVALUAR",
            "msg": "Riesgo moderado o VAN ajustado. Optimiza costos.",
        }
    else:
        recomendacion = {
            "estado": "🔴 NO INVERTIR",
            "msg": "Destrucción de valor (VAN negativo). Proyecto inviable.",
        }

    mes_alcanza_equilibrio: int | str = "No alcanza"
    for mes in range(1, 37):
        idx_estacional = (mes - 1) % 12
        ajuste_estacional = (
            datos.estacionalidad[idx_estacional]
            if idx_estacional < len(datos.estacionalidad)
            else 0.0
        )
        ventas_mes = (
            datos.ventas.base
            * ((1 + datos.ventas.crecimiento_mensual / 100) ** (mes - 1))
            * (1 + ajuste_estacional / 100)
        )
        if ventas_mes >= punto_equilibrio:
            mes_alcanza_equilibrio = mes
            break

    ganancia_promedio_anio = round(
        (
            escenario_pesimista["caja_mensual"][11]
            + escenario_base["caja_mensual"][11]
            + escenario_optimista["caja_mensual"][11]
        )
        / 3,
        2,
    )

    return {
        "metricas": {
            "inversion_total": round(inversion_total, 2),
            "gastos_fijos": round(gastos_fijos_base, 2),
            "margen_unitario": round(margen_unitario, 2),
            "punto_equilibrio": punto_equilibrio,
            "margen_seguridad": round(margen_seguridad, 1),
            "reserva_emergencia": round(reserva_emergencia, 2),
            "capital_invertible": round(capital_invertible, 2),
            "alerta_liquidez": alerta_liquidez,
            "recursos_disponibles": round(recursos_disponibles, 2),
            "deficit_apertura": round(deficit_apertura, 2),
            "caja_despues_apertura": round(caja_despues_apertura, 2),
            "deficit_reserva": round(deficit_reserva, 2),
            "cobertura_meses": round(cobertura_meses, 2),
            "estado_liquidez": estado_liquidez,
            "van": van_base,
            "tir": tir_base,
            "roi": escenario_base["roi"],
            "b_c": escenario_base["b_c"],
            "score": score,
            "recomendacion": recomendacion,
            "prestamo": {
                "monto": round(monto_financiar, 2),
                "cuota": round(cuota_prestamo, 2),
                "cuota_mensual": round(cuota_prestamo, 2),
            },
            "mes_alcanza_equilibrio": mes_alcanza_equilibrio,
            "dedicacion": {
                "horas_semana": horas_semana,
                "porcentaje_presencial": presencia,
            },
        },
        "base": escenario_base,
        "pesimista": escenario_pesimista,
        "optimista": escenario_optimista,
        "riesgo": {
            "probabilidad_perdida": min(100, prob_perdida),
            "ganancia_promedio_anio": ganancia_promedio_anio,
        },
        "matriz_sensibilidad": matriz_sensibilidad,
    }


@app.post("/consejero")
async def obtener_consejo(datos: dict):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel("gemini-3.5-flash")
        rol = datos.get("rol")
        metricas = datos.get("metricas", {})
        prompt = (
            f"Proyecto: {datos.get('idea')} (Sector: {datos.get('sector')}).\n"
            f"VAN: {metricas.get('van', 'N/A')}. "
            f"TIR: {metricas.get('tir', 'N/A')}%. "
            f"Score: {metricas.get('score', 'N/A')}/100.\n\n"
        )

        if rol == "auditor":
            prompt += (
                "Actúa como auditor financiero estricto. "
                "Dame 3 consejos concretos para reducir riesgos y mejorar VAN/TIR."
            )
        elif rol == "marketing":
            prompt += (
                "Actúa como director de marketing. "
                "Diseña una estrategia rápida para asegurar las ventas base."
            )
        elif rol == "operaciones":
            prompt += (
                "Actúa como jefe de operaciones. "
                "Detecta posibles cuellos de botella con este nivel de liquidez."
            )

        respuesta = modelo.generate_content(prompt)
        return {"consejo": respuesta.text}
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            return {
                "consejo": (
                    "⚠️ **Límite de consultas alcanzado.**\n\n"
                    "Espera 60 segundos y vuelve a intentarlo."
                )
            }
        return {"consejo": f"Error de conexión IA: {error_str}"}


@app.post("/chat")
async def chat_ia(datos: dict):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel("gemini-3.5-flash")
        historial = datos.get("history", [])
        pregunta = datos.get("question", "")
        contexto = (
            f"Contexto del negocio: {datos.get('idea')} ({datos.get('sector')}). "
            f"ROI: {datos.get('metricas', {}).get('roi')}%. "
        )

        mensajes = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Eres un experto asesor financiero y de negocios "
                            "de la plataforma Decisiones de Inversión IA."
                        )
                    }
                ],
            }
        ]

        for msg in historial:
            rol = "user" if msg.get("role") == "user" else "model"
            mensajes.append(
                {"role": rol, "parts": [{"text": msg.get("content", "")}]}
            )

        mensajes.append(
            {"role": "user", "parts": [{"text": contexto + pregunta}]}
        )

        respuesta = modelo.generate_content(mensajes)
        return {"respuesta": respuesta.text}
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            return {
                "respuesta": (
                    "⚠️ Límite de consultas gratuitas superado. "
                    "Espera 60 segundos antes de enviar otro mensaje."
                )
            }
        return {"respuesta": f"Error en el chat: {error_str}"}
