from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Simulador de Negocios - Método 10000 Soles")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS DE DATOS ---
class InversionInicial(BaseModel):
    insumos: float
    equipos: float
    empaques: float
    permisos: float
    otros: float

class GastosMensuales(BaseModel):
    marketing: float
    logistica: float
    sueldo_emprendedor: float
    impuestos: float
    otros: float

class ProyeccionVentas(BaseModel):
    pesimista: int
    base: int
    optimista: int
    crecimiento_mensual: float

class InputsSimulador(BaseModel):
    nombre_idea: str
    sector: str
    descripcion: str
    inversion: InversionInicial
    precio_venta: float
    costo_directo: float
    gastos_fijos: GastosMensuales
    ventas: ProyeccionVentas

# --- MOTOR MATEMÁTICO ---
def calcular_flujo(inputs: InputsSimulador, ventas_iniciales: int, meses: int = 12) -> dict:
    inv_total = sum(inputs.inversion.dict().values())
    gastos_totales_mes = sum(inputs.gastos_fijos.dict().values())
    precio = inputs.precio_venta
    costo_dir = inputs.costo_directo
    crecimiento = inputs.ventas.crecimiento_mensual / 100.0
    
    ingresos = [0.0] * meses
    costos = [0.0] * meses
    caja = [0.0] * meses
    ventas_proyectadas = [0] * meses
    
    caja_acumulada = -inv_total
    
    for i in range(meses):
        # Aplicar crecimiento mensual
        ventas_mes = int(ventas_iniciales * ((1 + crecimiento) ** i))
        ventas_proyectadas[i] = ventas_mes
        
        ingreso_mes = ventas_mes * precio
        costo_mes = (ventas_mes * costo_dir) + gastos_totales_mes
        
        caja_acumulada += (ingreso_mes - costo_mes)
        
        ingresos[i] = round(ingreso_mes, 2)
        costos[i] = round(costo_mes, 2)
        caja[i] = round(caja_acumulada, 2)
        
    return {
        "ventas_mes_a_mes": ventas_proyectadas,
        "caja_mes_a_mes": caja,
        "mes_recuperacion": next((i + 1 for i, v in enumerate(caja) if v >= 0), "No recupera en Año 1")
    }

@app.post("/simular")
async def simular_negocio(inputs: InputsSimulador):
    # 1. Validaciones y Métricas Clave
    inversion_total = sum(inputs.inversion.dict().values())
    gastos_mes = sum(inputs.gastos_fijos.dict().values())
    
    viable_capital = inversion_total <= 10000
    
    # Capital de Trabajo / Fondo de Maniobra
    fondo_maniobra = gastos_mes * 2
    capital_restante = 10000 - inversion_total
    cubre_fondo = capital_restante >= fondo_maniobra
    falta_fondo = round(fondo_maniobra - capital_restante, 2) if not cubre_fondo else 0
    
    # CAC (Costo de Adquisición de Clientes)
    cac = round(inputs.gastos_fijos.marketing / inputs.ventas.base, 2) if inputs.ventas.base > 0 else 0
    
    # 2. Escenario Base
    base = calcular_flujo(inputs, inputs.ventas.base)
    
    # 3. Simulación Monte Carlo (Riesgo)
    iteraciones = 1000
    # Usamos los 3 escenarios del usuario para la distribución estadística
    ventas_mc = np.random.triangular(
        inputs.ventas.pesimista, 
        inputs.ventas.base, 
        inputs.ventas.optimista, 
        iteraciones
    )
    
    fracasos = 0
    caja_promedio_final = 0
    
    for i in range(iteraciones):
        res = calcular_flujo(inputs, int(ventas_mc[i]))
        caja_fin_anio = res["caja_mes_a_mes"][-1]
        caja_promedio_final += caja_fin_anio
        if caja_fin_anio < 0:
            fracasos += 1
            
    prob_fracaso = round((fracasos / iteraciones) * 100, 2)
    ganancia_promedio = round(caja_promedio_final / iteraciones, 2)

    return {
        "metricas": {
            "inversion_total": inversion_total,
            "es_viable_5k": viable_capital,
            "fondo_maniobra_req": fondo_maniobra,
            "cubre_fondo": cubre_fondo,
            "falta_fondo": falta_fondo,
            "cac_estimado": cac,
            "gastos_fijos_mes": gastos_mes
        },
        "base": base,
        "riesgo": {
            "probabilidad_perdida": prob_fracaso,
            "ganancia_promedio_anio": ganancia_promedio
        }
    }
