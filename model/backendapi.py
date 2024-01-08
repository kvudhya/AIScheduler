from fastapi import FastAPI
from pydantic import BaseModel
from model.findoptiTime import to_do_to_schedule


app = FastAPI()

creating =to_do_to_schedule()

class eventsOfToday(BaseModel):
    summary:str
    scheduled: bool


class add_events(BaseModel):
    event: str


class prediction(BaseModel):
    time: int

@app.get('/todolist')
def get_shit():
    return {'message: it works'}

@app.post('/prediction')
def predict(payload:add_events):
    time = creating.prediction([payload.event])

    return {"time": z}

@app.put("/{item_id}")
def adding_events(item_id:int,item:eventsOfToday):
    return{'item summary': item.summary, 'item_id': item_id}