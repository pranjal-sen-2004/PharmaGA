from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pharma_ga import generate_drug_library, generate_patient, PharmaGA

app = FastAPI(title="PharmaGA API")

# Allow React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Default Vite React port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input from the React frontend
class PatientConfig(BaseModel):
    age: int = 55
    bmi: float = 24.5
    hla_risk: float = 0.15

# Initialize the library once at startup to save compute time
print("Initializing Drug Library...")
DRUG_LIBRARY = generate_drug_library(100, seed=42)

@app.post("/api/run-ga")
def run_genetic_algorithm(config: PatientConfig):
    # 1. Setup Patient based on UI inputs
    patient = generate_patient(1, 'HTN', seed=7)
    patient.age = config.age
    patient.bmi = config.bmi
    patient.hla_risk = config.hla_risk

    # 2. Run the Genetic Algorithm (reduced parameters for faster API response)
    ga = PharmaGA(
        DRUG_LIBRARY, 
        mu=30, 
        lam=30, 
        max_generations=20, # Reduced for web responsiveness
        oracle_budget=200, 
        verbose=False
    )
    archive, archive_fitness = ga.run(patient)

    # 3. Format the output for the React frontend
    results = []
    for i, (regimen, fitness) in enumerate(zip(archive, archive_fitness)):
        drugs = [{"name": d.name, "dose": dose} for d, dose in regimen]
        results.append({
            "id": f"Regimen_{i+1}",
            "drugs": drugs,
            "efficacy": round(float(-fitness[0]), 3), # Invert back to positive
            "toxicity": round(float(fitness[1]), 3),
            "pk_mismatch": round(float(fitness[2]), 3),
            "ddi": round(float(fitness[3]), 3)
        })
        
    return {"pareto_front": results}