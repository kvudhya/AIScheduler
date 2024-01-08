#import scemodel
#from scemodelexe import vectorize_layer
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from main import *
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras

from tensorflow.keras import layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


class to_do_to_schedule:
    def __init__(self):
        self.creds = main()
        #self.vec_layer = vectorize_layer

    def normalization_label(self,label):
        if label == 1:
            return 3
        if label == 15:
            return 0
        if label == 30:
            return 1
        if label == 45:
            return 2
        if label == 2:
            return 4

    label_map = {0: 15, 1: 30, 2: 45, 3: 1, 4: 2}

    def filter_train(self, time):
        return (True
                if time < 4
                else False)

    def getting_vocab(self):

        dataset = pd.read_csv("../training/present_csv.csv")

        dataset['Time Duration'] = dataset['Time Duration'].map(self.normalization_label)

        # dataset = dataset.drop(columns=["Unnamed"])
        # print(dataset.head())
        raw_train_ds = tf.data.Dataset.from_tensor_slices((dataset['Events'], dataset['Time Duration']))

        # print(raw_train_ds)
        raw_test_ds = raw_train_ds.take(10)

        raw_train_ds = raw_train_ds.skip(10)

        max_features = 10000

        vectorize_layer = layers.TextVectorization(
            max_tokens=10000,
            output_mode='int',
            output_sequence_length=250
        )

        text = raw_train_ds.map(lambda text, label: text)

        vectorize_layer.adapt(text)
        return vectorize_layer

    def prediction( self, examples):
        label_map = {0: 15, 1: 30, 2: 45, 3: 1, 4: 2}
        trained_Model = tf.keras.models.load_model('scemodeldetV2.0')

        vectorize_layer = self.getting_vocab()
        exported_model = keras.Sequential([
            vectorize_layer,
            trained_Model,
            keras.layers.Activation('softmax')
        ])
        if examples is None:
            return
        else:
            if len(examples) > 1:
                predictions = exported_model.predict(examples)
                for example in predictions:
                    score = tf.nn.softmax(example)
                    return label_map[np.argmax(score)]
            else:
                predictions = exported_model.predict(examples)
                score = tf.nn.softmax(predictions[0])
                return label_map[np.argmax(score)]

    def gettingTasks(self, month,day):
        months = {'Jan': 1,"Feb":2,'March':3,'April':4, 'May':5,  'June':6, "July":7,'Aug':8, "Sept":9, 'Oct':10,'Nov':11, 'Dec':12}

        service = build('calendar', 'v3', credentials=self.creds)
        now = dt.datetime.now().isoformat() + 'Z'
        today = dt.datetime(year=dt.datetime.now().year, month =months[month], day=day)

        start = dt.datetime.date(today)
        print(start)
        start_time = dt.datetime(start.year, start.month, start.day, hour=00, minute=00, second=00).isoformat() + 'Z'
        end_time = dt.datetime(start.year, start.month, start.day, hour=23, minute=59, second=59).isoformat() + 'Z'

        events_results = service.events().list(calendarId='primary', timeMin=start_time, timeMax=end_time,
                                               singleEvents=True, orderBy='startTime').execute()
        events = events_results.get('items', [])


        return events

    def makeanOrder(self, task, description, month, day ):
        time_duration= self.prediction(task)
        if time_duration not in [15,30,45]:
            change_duration = datetime.timedelta(hours= time_duration)
        else:
            change_duration = datetime.timedelta(minutes= time_duration)
        try:
            times = []
            events = self.gettingTasks(month,day)

            if not events:
                #today = dt.datetime(year=dt.datetime.now().year, month =months[month], day=day)
                return

            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))

                startTime = dt.datetime.fromisoformat(start)
                endTime = dt.datetime.fromisoformat(end)
                #print(event['summary'])
                times.append([startTime,endTime])

            #find difference between endTime of one event and StartTime of another event,
            # if diff>timeduration+10min then call schedule function
            #1 6 7 1 5 9
            #print(times)
            change = datetime.timedelta(minutes=10)


            for i in range(len(times)):
                print(i)
                if i == len(times)-1:
                    break
                elif len(times)>=2:
                    diff = (times[i+1][0] - times[i][1]).seconds / 3600

                    if diff >= time_duration + (2 / 6):
                        creating_events(self.creds, task,
                                        from_time=times[i][1] + change, end_time=times[i][1] + change + change_duration,
                                        description=description)


                        break

                else:
                    #print("hell yah")
                    creating_events(self.creds, task,
                                    from_time=times[i+1][1] + change, end_time=times[i+1][1] + change + change_duration,
                                    description=description)
                    break






                    # got fix this

                #10am -2pm 9pm - 11pm
                # else:
                #     for event in events:
                #         if task != str(event['summary']):
                #             creating_events(self.creds, task,
                #                             from_time=times[i][1] + change, end_time=times[i][1]+ change + change_duration,
                #                             description=description)
                #             break
                        #arraylist--> delete task from list after scheduleing.


        except HttpError as error:
            print("Error Occured!!! : ", error)



