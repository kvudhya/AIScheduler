import datetime as dt
import pandas as pd
import os.path
#from scemodel import *
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/calendar"]

def main():
    creds = None


    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())


        else:
            flow = InstalledAppFlow.from_client_secrets_file('credintionals.json', SCOPES)
            creds = flow.run_local_server(port=0)


        with open('token.json', 'w') as token:
            token.write(creds.to_json())


    past_events = datamining(creds=creds)
    return creds

def roundingProcess(hours):
    if hours>=1:
        return int(hours)

    else:
        return hours*60



def remove_dup(tasks):
    tasks=list(set(tasks))
    return tasks


def datamining(creds):
    try:
        service = build('calendar','v3', credentials=creds )
        now = dt.datetime.now().isoformat() +'Z'
        #print(dt.datetime.now().isoformat()+'Z')

        #print(dt.datetime(2020,1,1, hour=00, minute=00, second= 00 ).isoformat()+'Z')
        past = dt.datetime(2023,1,1, hour=00, minute=00, second= 00 ).isoformat()+'Z'
        events_result = service.events().list(calendarId='primary', timeMin = past, timeMax= now,
                                               singleEvents= True, orderBy='startTime').execute()

        events = events_result.get('items', [])
        #print(events)
        if not events:
            print("No previous evidents")

            return
        tasks = []
        timeDuration = []
        for event in events:
            #print(event)
            start  = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))

            startTime = dt.datetime.fromisoformat(start)
            endTime = dt.datetime.fromisoformat(end)
            timeDur = endTime-startTime

            hours = timeDur.seconds/3600

            #print(hours, event['summary'])
            tasks.append(event['summary'])
            timeDuration.append(hours)

        tasks = remove_dup(tasks)

        packageOflists = list(zip(tasks, timeDuration))


        pds_dataset= pd.DataFrame(packageOflists, columns=["Events", 'Time Duration'])

        pds_dataset["Time Duration"] =pds_dataset['Time Duration'].map(roundingProcess)

        pds_dataset.to_csv("present_csv2.csv", index=False)



        return pds_dataset




    except HttpError as error:
        print('An error occured!! :', error)


def creating_events(creds,task, from_time, end_time, description):
    #pass
    try:
        service = build('calendar', 'v3', credentials=creds)

        event_json = {
            'summary': str(task[0]),
            'description': str(description),
            'start': {
                'dateTime': from_time.isoformat(),
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'America/New_York',
            },
            'attendees': [
                {'email': 'kvudhya@gmail.com'},

            ],
            'reminders': {
                'useDefault': True,


            },
        }

        event = service.events().insert(calendarId= 'primary', body = event_json).execute()








    except HttpError as error:
        print('An error occured!! :', error)


if __name__ == "__main__":
    main()