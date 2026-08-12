from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inversion(BaseModel):
    insumos: float
    equipos: float
    empaques: float
    permisos: float
    otros: float

class GastosFijos(BaseModel):
    marketing: float
    logistica: float
    sueldo_emprendedor: float
    otros: float

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
    inversion: Inversion
    precio_venta: float
    costo_directo: float
    gastos_fijos: GastosFijos
    ventas: Ventas
    regimen_tributario: str = "NRUS"
    inflacion_anual: float = 3.0
    solicitar_prestamo: bool = False
    tea: float = 15.0
    plazo_meses: int = 12

@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_total = sum([
        datos.inversion.insumos, datos.inversion.equipos,
        datos.inversion.empaques, datos.inversion.permisos, datos.inversion.otros
    ])
    
    gastos_fijos_base = sum([
        datos.gastos_fijos.marketing, datos.gastos_fijos.logistica,
        datos.gastos_fijos.sueldo_emprendedor, datos.gastos_fijos.otros
    ])
    
    # Cálculo de Préstamo Bancario (Método Francés)
    monto_prestamo = 0
    cuota_prestamo = 0
    if datos.solicitar_prestamo and inversion_total > datos.capital_disponible:
        monto_prestamo = inversion_total - datos.capital_disponible
        tem = (1 + (datos.tea / 100)) ** (1/12) - 1 # Tasa Efectiva Mensual
        if tem > 0 and datos.plazo_meses > 0:
            cuota_prestamo = monto_prestamo * (tem * (1 + tem)**datos.plazo_meses) / ((1 + tem)**datos.plazo_meses - 1)
        else:
            cuota_prestamo = monto_prestamo / max(1, datos.plazo_meses)
            
    # El punto de equilibrio debe cubrir la cuota del préstamo
    margen_unitario = datos.precio_venta - datos.costo_directo
    impuesto_estimado = 50 if datos.regimen_tributario == "NRUS" else (datos.ventas.base * datos.precio_venta * 0.015)
    gastos_mes_1 = gastos_fijos_base + impuesto_estimado + cuota_prestamo
    punto_equilibrio = 999999 if margen_unitario <= 0 else int(gastos_mes_1 / margen_unitario) + 1
    
    reserva_emergencia = gastos_fijos_base * 3
    capital_invertible = max(0, datos.capital_disponible - reserva_emergencia)
    margen_seguridad = max(0, ((datos.ventas.base - punto_equilibrio) / datos.ventas.base) * 100) if datos.ventas.base > 0 else 0

    def proyectar_escenario(ventas_iniciales, crecimiento):
        # La caja arranca en negativo solo por la parte que el usuario puso de su bolsillo
        capital_propio_invertido = min(inversion_total, datos.capital_disponible)
        caja_acumulada = -capital_propio_invertido
        meses = []
        mes_recuperacion = "No recupera en Año 1"
        ventas_totales_anio = 0
        costos_totales_anio = 0
        
        for mes in range(1, 13):
            ventas_mes = ventas_iniciales * ((1 + (crecimiento/100)) ** (mes - 1))
            ventas_totales_anio += ventas_mes
            ingresos = ventas_mes * datos.precio_venta
            costos_variables = ventas_mes * datos.costo_directo
            
            if datos.regimen_tributario == "NRUS":
                impuestos = 20 if ingresos <= 5000 else 50
            elif datos.regimen_tributario == "RER":
                impuestos = ingresos * 0.015
            else: 
                impuestos = ingresos * 0.01

            inflacion_mensual = datos.inflacion_anual / 100 / 12
            gastos_fijos_inflados = gastos_fijos_base * ((1 + inflacion_mensual) ** (mes - 1))
            
            # Sumamos la cuota del préstamo a los costos de salida de caja
            cuota_mes = cuota_prestamo if mes <= datos.plazo_meses else 0
            
            costos_mes = costos_variables + gastos_fijos_inflados + impuestos + cuota_mes
            costos_totales_anio += costos_mes
            
            utilidad_neta = ingresos - costos_mes
            caja_acumulada += utilidad_neta
            meses.append(round(caja_acumulada, 2))
            
            if caja_acumulada >= 0 and mes_recuperacion == "No recupera en Año 1":
                mes_recuperacion = mes
                
        ingresos_totales = ventas_totales_anio * datos.precio_venta
        margen_neto = ((ingresos_totales - costos_totales_anio) / ingresos_totales) * 100 if ingresos_totales > 0 else 0
                
        return {"caja_mes_a_mes": meses, "caja_final": round(caja_acumulada, 2), "mes_recuperacion": mes_recuperacion, "margen_neto": round(margen_neto, 2)}
        
    escenario_pesimista = proyectar_escenario(datos.ventas.pesimista, 0)
    escenario_base = proyectar_escenario(datos.ventas.base, datos.ventas.crecimiento_mensual)
    escenario_optimista = proyectar_escenario(datos.ventas.optimista, datos.ventas.crecimiento_mensual * 1.5)
    
    prob_perdida = 0
    if escenario_pesimista["caja_final"] < 0: prob_perdida += 35
    if escenario_base["caja_final"] < 0: prob_perdida += 45
    if punto_equilibrio > datos.ventas.base: prob_perdida += 20
    
    capital_propio_invertido = min(inversion_total, datos.capital_disponible)
    roi = (escenario_base["caja_final"] / capital_propio_invertido) * 100 if capital_propio_invertido > 0 else 0
    ganancia_promedio = round((escenario_pesimista["caja_final"] + escenario_base["caja_final"] + escenario_optimista["caja_final"]) / 3, 2)

    score = 100
    score -= prob_perdida
    if roi < 10: score -= 20
    elif roi > 50: score += 10
    if type(escenario_base["mes_recuperacion"]) == int:
        if escenario_base["mes_recuperacion"] > 8: score -= 15
    else: score -= 30
    if margen_seguridad < 15: score -= 10
    if inversion_total > capital_invertible and not datos.solicitar_prestamo: score -= 20
    score = max(0, min(100, score))

    if score >= 75: recomendacion = {"estado": "🟢 INVERTIR", "msg": "Alto potencial y riesgo controlado."}
    elif score >= 45: recomendacion = {"estado": "🟡 ANALIZAR MEJOR", "msg": "Rentabilidad moderada o riesgo alto."}
    else: recomendacion = {"estado": "🔴 NO INVERTIR", "msg": "Alta probabilidad de pérdida o capital insuficiente."}
    
    return {
        "metricas": {
            "inversion_total": inversion_total, "margen_unitario": margen_unitario, "punto_equilibrio": punto_equilibrio,
            "margen_seguridad": round(margen_seguridad, 1), "roi": round(roi, 1), "reserva_emergencia": reserva_emergencia,
            "capital_invertible": capital_invertible, "score": score, "recomendacion": recomendacion,
            "prestamo": {"monto": round(monto_prestamo, 2), "cuota_mensual": round(cuota_prestamo, 2)}
        },
        "pesimista": escenario_pesimista, "base": escenario_base, "optimista": escenario_optimista,
        "riesgo": {"probabilidad_perdida": min(100, prob_perdida), "ganancia_promedio_anio": ganancia_promedio}
    }

@app.post("/consejero")
async def obtener_consejo(datos: dict):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel('gemini-3.5-flash')
        rol = datos.get("rol")
        metricas = datos.get("metricas", {})
        prompt = f"Proyecto: {datos.get('idea')} (Sector: {datos.get('sector')}).\nScore: {metricas.get('score', 'N/A')}/100. Riesgo: Moderado.\n\n"
        if rol == "auditor": prompt += "Actúa como auditor estricto. Dame 3 consejos crudos para reducir costos o mitigar riesgos."
        elif rol == "marketing": prompt += "Actúa como director de marketing. Diséñame una estrategia rápida y textos promocionales."
        elif rol == "operaciones": prompt += "Actúa como asesor operativo. Detecta cuellos de botella en logística o producción."
        respuesta = modelo.generate_content(prompt)
        return {"consejo": respuesta.text}
    except Exception as e: return {"consejo": f"Error de conexión IA: {str(e)}"}

@app.post("/chat")
async def chat_ia(datos: dict):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel('gemini-3.5-flash')
        historial = datos.get("history", [])
        pregunta = datos.get("question", "")
        contexto = f"Contexto del negocio: {datos.get('idea')} ({datos.get('sector')}). ROI: {datos.get('metricas', {}).get('roi')}%. "
        
        mensajes = [{"role": "user", "parts": [{"text": "Eres un experto asesor financiero y de negocios de la plataforma Decisiones de Inversión IA."}]}]
        for msg in historial:
            rol = "user" if msg["role"] == "user" else "model"
            mensajes.append({"role": rol, "parts": [{"text": msg["content"]}]})
        mensajes.append({"role": "user", "parts": [{"text": contexto + pregunta}]})
        
        respuesta = modelo.generate_content(mensajes)
        return {"respuesta": respuesta.text}
    except Exception as e: return {"respuesta": f"Error en el chat: {str(e)}"}
