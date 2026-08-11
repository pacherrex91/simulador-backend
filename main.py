from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI

app = FastAPI()

# Configuración para permitir que tu página web se comunique con este servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estructura de datos que recibe el simulador
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
    inversion: Inversion
    precio_venta: float
    costo_directo: float
    gastos_fijos: GastosFijos
    ventas: Ventas

# Endpoint principal: El motor financiero
@app.post("/simular")
def simular_negocio(datos: DatosSimulacion):
    inversion_total = sum([
        datos.inversion.insumos,
        datos.inversion.equipos,
        datos.inversion.empaques,
        datos.inversion.permisos,
        datos.inversion.otros
    ])
    
    gastos_fijos_mes = sum([
        datos.gastos_fijos.marketing,
        datos.gastos_fijos.logistica,
        datos.gastos_fijos.sueldo_emprendedor,
        datos.gastos_fijos.impuestos,
        datos.gastos_fijos.otros
    ])
    
    margen_unitario = datos.precio_venta - datos.costo_directo
    
    # Prevención de error matemático si el costo es mayor o igual al precio
    if margen_unitario <= 0:
        punto_equilibrio = 999999 
    else:
        punto_equilibrio = int(gastos_fijos_mes / margen_unitario) + 1
        
    capital_restante = 10000 - inversion_total
    fondo_maniobra_req = gastos_fijos_mes * 2
    
    def proyectar_escenario(ventas_iniciales, crecimiento):
        caja_acumulada = -inversion_total
        meses = []
        mes_recuperacion = "No recupera en Año 1"
        
        for mes in range(1, 13):
            ventas_mes = ventas_iniciales * ((1 + (crecimiento/100)) ** (mes - 1))
            ingresos = ventas_mes * datos.precio_venta
            costos_variables = ventas_mes * datos.costo_directo
            utilidad_neta = ingresos - costos_variables - gastos_fijos_mes
            caja_acumulada += utilidad_neta
            meses.append(round(caja_acumulada, 2))
            
            if caja_acumulada >= 0 and mes_recuperacion == "No recupera en Año 1":
                mes_recuperacion = mes
                
        return {
            "caja_mes_a_mes": meses,
            "caja_final": round(caja_acumulada, 2),
            "mes_recuperacion": mes_recuperacion
        }
        
    escenario_pesimista = proyectar_escenario(datos.ventas.pesimista, 0)
    escenario_base = proyectar_escenario(datos.ventas.base, datos.ventas.crecimiento_mensual)
    escenario_optimista = proyectar_escenario(datos.ventas.optimista, datos.ventas.crecimiento_mensual * 1.5)
    
    prob_perdida = 0
    if escenario_pesimista["caja_final"] < 0: prob_perdida += 40
    if escenario_base["caja_final"] < 0: prob_perdida += 50
    if punto_equilibrio > datos.ventas.base: prob_perdida += 10
    
    return {
        "metricas": {
            "inversion_total": inversion_total,
            "gastos_fijos_mes": gastos_fijos_mes,
            "margen_unitario": margen_unitario,
            "punto_equilibrio": punto_equilibrio,
            "fondo_maniobra_req": fondo_maniobra_req,
            "cubre_fondo": capital_restante >= fondo_maniobra_req,
            "falta_fondo": max(0, fondo_maniobra_req - capital_restante)
        },
        "pesimista": escenario_pesimista,
        "base": escenario_base,
        "optimista": escenario_optimista,
        "riesgo": {
            "probabilidad_perdida": min(100, prob_perdida),
            "ganancia_promedio_anio": round((escenario_pesimista["caja_final"] + escenario_base["caja_final"] + escenario_optimista["caja_final"]) / 3, 2)
        }
    }

# Nuevo Endpoint: El Consejero de Inteligencia Artificial (DeepSeek)
@app.post("/consejero")
async def obtener_consejo(datos: dict):
    try:
        # Iniciamos el cliente de IA apuntando a la API de DeepSeek
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        rol = datos.get("rol")
        metricas = datos.get("metricas", {})
        
        # Construimos el contexto que le daremos a la IA
        prompt = f"Analiza este proyecto: {datos.get('idea')} (Sector: {datos.get('sector')}).\n"
        prompt += f"Ventas necesarias para punto de equilibrio: {metricas.get('punto_equilibrio', 'N/A')}.\n"
        prompt += f"Dinero faltante para reserva de emergencia: S/ {metricas.get('falta_fondo', 'N/A')}.\n\n"
        
        # Definimos el comportamiento según el botón que se presionó
        if rol == "auditor":
            prompt += "Actúa como un auditor financiero estricto. Analiza el riesgo de este negocio y dame 3 consejos crudos, directos y prácticos para reducir costos o mejorar el punto de equilibrio."
        elif rol == "marketing":
            prompt += "Actúa como un director de marketing. Toma los datos de esta simulación y genérame textos promocionales (copy) estructurados, persuasivos y listos para copiar y pegar directamente en WhatsApp y Facebook."
        elif rol == "operaciones":
            prompt += "Actúa como un asesor operativo experimentado. Detecta puntos ciegos en la logística de este negocio y dame recomendaciones para optimizar el tiempo y los recursos."
            
        # Hacemos la llamada al modelo de DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto asesor de negocios. Responde de manera clara, directa y muy bien estructurada."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {"consejo": response.choices[0].message.content}
    except Exception as e:
        return {"consejo": f"Error contactando a la Inteligencia Artificial: {str(e)}"}
