from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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

class Ventas(BaseModel):
    pesimista: int
    base: int
    optimista: int
    crecimiento_mensual: float

class DatosSimulacion(BaseModel):
    nombre_idea: str
    sector: str
    moneda: str = "S/"
    capital_disponible: float = 10000.0
    inversion_dinamica: list[ItemDinamico] = []
    gastos_dinamicos: list[ItemDinamico] = []
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
    estacionalidad: list[float] = [0.0]*12

@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_total = sum([item.monto for item in datos.inversion_dinamica])
    gastos_fijos_base = sum([item.monto for item in datos.gastos_dinamicos])
    
    depreciacion_mensual = 0.0
    for item in datos.inversion_dinamica:
        if item.vida_util > 0:
            depreciacion_mensual += (item.monto - item.residual) / item.vida_util

    cuota_prestamo = 0
    tasa = datos.financiamiento_tasa_mensual / 100
    plazo = datos.financiamiento_plazo
    monto_financiar = datos.financiamiento_monto
    if monto_financiar > 0 and plazo > 0:
        if tasa > 0: cuota_prestamo = monto_financiar * (tasa * (1 + tasa)**plazo) / ((1 + tasa)**plazo - 1)
        else: cuota_prestamo = monto_financiar / plazo

    reserva_emergencia = gastos_fijos_base * datos.meses_reserva
    capital_invertible = max(0, datos.capital_disponible - reserva_emergencia)
    alerta_liquidez = "⚠️ Tu capital no cubre la inversión y la reserva requerida." if (inversion_total + reserva_emergencia) > (datos.capital_disponible + monto_financiar) else "✅ Liquidez saludable."

    margen_unitario = datos.precio_venta - datos.costo_directo
    punto_equilibrio = 999999 if margen_unitario <= 0 else int((gastos_fijos_base + cuota_prestamo) / margen_unitario) + 1
    margen_seguridad = max(0, ((datos.ventas.base - punto_equilibrio) / datos.ventas.base) * 100) if datos.ventas.base > 0 else 0

    def proyectar_escenario(ventas_iniciales, crecimiento, precio_mult=1.0, costo_mult=1.0):
        caja_acumulada = -(min(inversion_total, datos.capital_disponible))
        flujo_neto_mensual = []
        caja_mensual = []
        p_y_g = []
        
        mes_recuperacion = "No recupera"
        precio_final = datos.precio_venta * precio_mult
        costo_final = datos.costo_directo * costo_mult
        
        for mes in range(1, 37):
            idx_estacional = (mes - 1) % 12
            factor_estacional = 1.0 + (datos.estacionalidad[idx_estacional] / 100.0)
            
            ventas_mes = ventas_iniciales * ((1 + (crecimiento/100)) ** (mes - 1)) * factor_estacional
            ingresos = ventas_mes * precio_final
            costos_variables = ventas_mes * costo_final
            margen_bruto = ingresos - costos_variables
            
            inflacion_mensual = datos.inflacion_anual / 100 / 12
            gastos_fijos_inflados = gastos_fijos_base * ((1 + inflacion_mensual) ** (mes - 1))
            
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
            
            p_y_g.append({
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
                "caja_acumulada": round(caja_acumulada, 2)
            })
            
            if caja_acumulada >= 0 and mes_recuperacion == "No recupera":
                mes_recuperacion = mes

        tasa_mensual = (datos.tasa_descuento / 100) / 12
        flujos_para_van = [-(min(inversion_total, datos.capital_disponible))] + flujo_neto_mensual
        
        try: van = float(npf.npv(tasa_mensual, flujos_para_van))
        except: van = 0.0
        
        try:
            tir_m = float(npf.irr(flujos_para_van))
            tir = ((1 + tir_m)**12 - 1) * 100 if tir_m > -1 else -100
        except: tir = -100

        roi = (caja_acumulada / min(inversion_total, datos.capital_disponible)) * 100 if inversion_total > 0 else 0
        b_c = (sum([f for f in flujo_neto_mensual if f > 0]) / sum([abs(f) for f in flujos_para_van if f < 0])) if flujos_para_van else 0
        
        return {
            "caja_mensual": caja_mensual, "caja_final": round(caja_acumulada, 2), 
            "mes_recuperacion": mes_recuperacion, "van": round(van, 2), "tir": round(tir, 2),
            "roi": round(roi, 2), "b_c": round(b_c, 2), "p_y_g": p_y_g
        }

    escenario_base = proyectar_escenario(datos.ventas.base, datos.ventas.crecimiento_mensual)
    escenario_pesimista = proyectar_escenario(datos.ventas.pesimista, 0)
    escenario_optimista = proyectar_escenario(datos.ventas.optimista, datos.ventas.crecimiento_mensual * 1.5)
    
    matriz_sensibilidad = []
    precios_mult = [0.8, 0.9, 1.0, 1.1, 1.2]
    vols_mult = [0.8, 0.9, 1.0, 1.1, 1.2]
    for p_m in precios_mult:
        fila = {"precio_mult": p_m, "valores": []}
        for v_m in vols_mult:
            res_sens = proyectar_escenario(datos.ventas.base * v_m, datos.ventas.crecimiento_mensual, p_m, 1.0)
            fila["valores"].append({"vol_mult": v_m, "van": res_sens["van"], "tir": res_sens["tir"]})
        matriz_sensibilidad.append(fila)

    van_base = escenario_base["van"]
    tir_base = escenario_base["tir"]
    
    prob_perdida = 0
    if escenario_pesimista["caja_final"] < 0: prob_perdida += 35
    if escenario_base["caja_final"] < 0: prob_perdida += 45
    if punto_equilibrio > datos.ventas.base: prob_perdida += 20
    
    score = 100
    score -= prob_perdida
    if escenario_base["roi"] < 10: score -= 20
    elif escenario_base["roi"] > 50: score += 10
    
    if type(escenario_base["mes_recuperacion"]) == int:
        if escenario_base["mes_recuperacion"] > 18: score -= 15
    else: score -= 30
    
    if margen_seguridad < 15: score -= 10
    if (inversion_total > datos.capital_disponible) and monto_financiar == 0: score -= 20
    score = max(0, min(100, score))

    if score >= 75 and van_base > 0:
        recomendacion = {"estado": "🟢 INVERTIR", "msg": "VAN positivo y Score alto. Genera valor."}
    elif score >= 45 or van_base > -2000:
        recomendacion = {"estado": "🟡 EVALUAR", "msg": "Riesgo moderado o VAN ajustado. Optimiza costos."}
    else:
        recomendacion = {"estado": "🔴 NO INVERTIR", "msg": "Destrucción de valor (VAN negativo). Proyecto inviable."}

    return {
        "metricas": {
            "inversion_total": inversion_total, "gastos_fijos": gastos_fijos_base,
            "margen_unitario": margen_unitario, "punto_equilibrio": punto_equilibrio,
            "reserva_emergencia": reserva_emergencia, "capital_invertible": capital_invertible,
            "alerta_liquidez": alerta_liquidez,
            "van": van_base, "tir": tir_base, "roi": escenario_base["roi"], "b_c": escenario_base["b_c"],
            "score": score,
            "recomendacion": recomendacion, "prestamo": {"monto": round(monto_financiar, 2), "cuota": round(cuota_prestamo, 2)}
        },
        "base": escenario_base, "pesimista": escenario_pesimista, "optimista": escenario_optimista,
        "riesgo": {"probabilidad_perdida": min(100, prob_perdida), "ganancia_promedio_anio": round(sum(escenario_base["caja_mensual"][:12])/12 if len(escenario_base["caja_mensual"])>=12 else 0, 2)},
        "matriz_sensibilidad": matriz_sensibilidad
    }

@app.post("/consejero")
async def obtener_consejo(datos: dict):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel('gemini-3.5-flash')
        rol = datos.get("rol")
        metricas = datos.get("metricas", {})
        prompt = f"Proyecto: {datos.get('idea')} (Sector: {datos.get('sector')}).\nVAN: {metricas.get('van')}. TIR: {metricas.get('tir')}%\n\n"
        if rol == "auditor": prompt += "Actúa como auditor financiero estricto. Dame 3 consejos sobre cómo mejorar el VAN y la TIR."
        elif rol == "marketing": prompt += "Actúa como director de marketing. Diséñame una estrategia rápida para asegurar las ventas base."
        elif rol == "operaciones": prompt += "Actúa como jefe de operaciones. Detecta posibles cuellos de botella con este nivel de liquidez."
        respuesta = modelo.generate_content(prompt)
        return {"consejo": respuesta.text}
    except Exception as e: 
        return {"consejo": f"Error de conexión IA: {str(e)}"}

@app.post("/chat")
async def chat_ia(datos: dict):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel('gemini-3.5-flash')
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
                        "text": "Eres un experto asesor financiero y de negocios de la plataforma Decisiones de Inversión IA."
                    }
                ],
            }
        ]

        for msg in historial:
            rol = "user" if msg["role"] == "user" else "model"
            mensajes.append({"role": rol, "parts": [{"text": msg["content"]}]})

        mensajes.append({"role": "user", "parts": [{"text": contexto + pregunta}]})

        respuesta = modelo.generate_content(mensajes)
        return {"respuesta": respuesta.text}
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            return {
                "respuesta": "⚠️ Límite de consultas gratuitas superado. Por favor, espera 60 segundos antes de enviar otro mensaje."
            }
        return {"respuesta": f"Error en el chat: {error_str}"}

