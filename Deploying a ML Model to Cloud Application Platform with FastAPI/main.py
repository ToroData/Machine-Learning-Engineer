from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI()

# Modelo Pydantic para la entrada de datos, usando Field para manejar nombres con guiones
class CensusData(BaseModel):
    age: int
    workclass: str = Field(..., alias="work-class")
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "work-class": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

# path to saved artifacts
savepath = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/inference/")
async def make_inference(data: CensusData):
    data_dict = data.dict(by_alias=True)

    sample = pd.DataFrame([data_dict])

    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]

    model = pickle.load(open(os.path.join(savepath,filename[0]), "rb"))
    encoder = pickle.load(open(os.path.join(savepath,filename[1]), "rb"))
    lb = pickle.load(open(os.path.join(savepath,filename[2]), "rb"))
        
    sample, _, _, _ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )
                          
    prediction = model.predict(sample)

    prediction_label = '>50K' if prediction[0] else '<=50K'
    return {"prediction": prediction_label}

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    pass
