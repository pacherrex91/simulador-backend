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
    impuestos: float
    otros: float

class Ventas(BaseModel):
    pesimista: int
    base: int
    optimista: int
    crecimiento_mensual: float

class DatosSimulacion(BaseModel):
    nombre_idea: str
    sector: str
    descripcion: str
    capital_disponible: float = 10000.0
    inversion: Inversion
    precio_venta: float
    costo_directo: float
    gastos_fijos: GastosFijos
    ventas: Ventas

@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_total = sum([
        datos.inversion.insumos, datos.inversion.equipos,
        datos.inversion.empaques, datos.inversion.permisos, datos.inversion.otros
    ])
    
    gastos_fijos_mes = sum([
        datos.gastos_fijos.marketing, datos.gastos_fijos.logistica,
        datos.gastos_fijos.sueldo_emprendedor, datos.gastos_fijos.impuestos, datos.gastos_fijos.otros
    ])
    
    margen_unitario = datos.precio_venta - datos.costo_directo
    punto_equilibrio = 999999 if margen_unitario <= 0 else int(gastos_fijos_mes / margen_unitario) + 1
    
    reserva_emergencia = gastos_fijos_mes * 3
    capital_invertible = max(0, datos.capital_disponible - reserva_emergencia)
    
    margen_seguridad = max(0, ((datos.ventas.base - punto_equilibrio) / datos.ventas.base) * 100) if datos.ventas.base > 0 else 0

    def proyectar_escenario(ventas_iniciales, crecimiento):
        caja_acumulada = -inversion_total
        meses = []
        mes_recuperacion = "No recupera en Año 1"
        ventas_totales_anio = 0
        costos_totales_anio = 0
        
        for mes in range(1, 13):
            ventas_mes = ventas_iniciales * ((1 + (crecimiento/100)) ** (mes - 1))
            ventas_totales_anio += ventas_mes
            
            ingresos = ventas_mes * datos.precio_venta
            costos_variables = ventas_mes * datos.costo_directo
            costos_totales_anio += costos_variables + gastos_fijos_mes
            
            utilidad_neta = ingresos - costos_variables - gastos_fijos_mes
            caja_acumulada += utilidad_neta
            meses.append(round(caja_acumulada, 2))
            
            if caja_acumulada >= 0 and mes_recuperacion == "No recupera en Año 1":
                mes_recuperacion = mes
                
        ingresos_totales = ventas_totales_anio * datos.precio_venta
        margen_neto = ((ingresos_totales - costos_totales_anio) / ingresos_totales) * 100 if ingresos_totales > 0 else 0
                
        return {
            "caja_mes_a_mes": meses,
            "caja_final": round(caja_acumulada, 2),
            "mes_recuperacion": mes_recuperacion,
            "margen_neto": round(margen_neto, 2)
        }
        
    escenario_pesimista = proyectar_escenario(datos.ventas.pesimista, 0)
    escenario_base = proyectar_escenario(datos.ventas.base, datos.ventas.crecimiento_mensual)
    escenario_optimista = proyectar_escenario(datos.ventas.optimista, datos.ventas.crecimiento_mensual * 1.5)
    
    # Simulación de riesgo más realista (ponderación)
    prob_perdida = 0
    if escenario_pesimista["caja_final"] < 0: prob_perdida += 35
    if escenario_base["caja_final"] < 0: prob_perdida += 45
    if punto_equilibrio > datos.ventas.base: prob_perdida += 20
    
    roi = (escenario_base["caja_final"] / inversion_total) * 100 if inversion_total > 0 else 0
    ganancia_promedio = round((escenario_pesimista["caja_final"] + escenario_base["caja_final"] + escenario_optimista["caja_final"]) / 3, 2)

    # Score de Inversión Algorítmico (0 - 100)
    score = 100
    score -= prob_perdida
    if roi < 10: score -= 20
    elif roi > 50: score += 10
    if type(escenario_base["mes_recuperacion"]) == int:
        if escenario_base["mes_recuperacion"] > 8: score -= 15
    else:
        score -= 30
    if margen_seguridad < 15: score -= 10
    if inversion_total > capital_invertible: score -= 20
    score = max(0, min(100, score))

    if score >= 75: recomendacion = {"estado": "🟢 INVERTIR", "msg": "Alto potencial, riesgo controlado y viable con tu capital."}
    elif score >= 45: recomendacion = {"estado": "🟡 ANALIZAR MEJOR", "msg": "Rentabilidad moderada o riesgo alto. Optimiza tus costos fijos."}
    else: recomendacion = {"estado": "🔴 NO INVERTIR", "msg": "Alta probabilidad de pérdida o capital insuficiente para operar seguro."}
    
    return {
        "metricas": {
            "inversion_total": inversion_total,
            "gastos_fijos_mes": gastos_fijos_mes,
            "margen_unitario": margen_unitario,
            "punto_equilibrio": punto_equilibrio,
            "margen_seguridad": round(margen_seguridad, 1),
            "roi": round(roi, 1),
            "reserva_emergencia": reserva_emergencia,
            "capital_invertible": capital_invertible,
            "score": score,
            "recomendacion": recomendacion
        },
        "pesimista": escenario_pesimista,
        "base": escenario_base,
        "optimista": escenario_optimista,
        "riesgo": {
            "probabilidad_perdida": min(100, prob_perdida),
            "ganancia_promedio_anio": ganancia_promedio
        }
    }

@app.post("/consejero")
async def obtener_consejo(datos: dict):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        modelo = genai.GenerativeModel('gemini-3.5-flash')
        
        rol = datos.get("rol")
        metricas = datos.get("metricas", {})
        
        prompt = f"Analiza este proyecto: {datos.get('idea')} (Sector: {datos.get('sector')}).\n"
        prompt += f"Punto de equilibrio: {metricas.get('punto_equilibrio', 'N/A')} ventas/mes.\n"
        prompt += f"Score de Inversión: {metricas.get('score', 'N/A')}/100.\n\n"
        
        if rol == "auditor":
            prompt += "Actúa como auditor estricto. Dame 3 consejos crudos para reducir costos o mitigar riesgos."
        elif rol == "marketing":
            prompt += "Actúa como director de marketing. Diséñame una estrategia rápida y textos promocionales."
        elif rol == "operaciones":
            prompt += "Actúa como asesor operativo. Detecta cuellos de botella en la logística o producción."
            
        respuesta = modelo.generate_content(prompt)
        return {"consejo": respuesta.text}
    except Exception as e:
        return {"consejo": f"Error de conexión IA: {str(e)}"}
