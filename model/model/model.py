"""Welcome to Reflex! This file outlines the steps to create a basic app."""
from rxconfig import config
#import app
#from app import model
import reflex as rx
import model.style as style
from findoptiTime import to_do_to_schedule
import asyncio

docs_url = "https://reflex.dev/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"

month: list[str]= ['Jan', "Feb", 'March', 'April', 'May', 'June', "July",
                  'Aug', 'Sept', 'Nov', 'Dec']
class State(rx.State):
    """The app state."""
    task = ""
    processing:bool
    complete:bool
    chat_history: list[str]
    month: str = ""
    day:int = ""
    def predictTime(self):
        if self.month == '' or self.day == '' or self.task =='':
            self.complete = False
            return rx.window_alert("Some of field are not entered")

            #return rx.window_alert("Task is not add")

        self.complete= False
        self.processing = True

        yield
        creating_tasks = to_do_to_schedule()
        creating_tasks.makeanOrder([self.task],"None", self.month, self.day)
        #
        self.chat_history.append(self.task)
        #self.task = ""

        self.complete, self.processing= True, False

        return rx.window_alert("The task been successfully added to google calendar")

    def removing_task(self,item:tuple):
       self.chat_history= [i for i in self.chat_history if i[0] != item[0]]




def qa(task:str) -> rx.Component:
    return rx.hstack(
        rx.box(task, text_align="left", style=style.question_style),
    )
# def removingTask(item:tuple[str, str]):
#     return rx.foreach(
#         State.chat_history,
#         lambda each_task:rx.cond(
#             ~(each_task[0] == item[0]),
#             each_task,
#             ()
#
#
#         ),
#     )
def TodoList() ->rx.Component:

    return rx.vstack(
        rx.foreach(
            State.chat_history,
            lambda messages: rx.checkbox(
                qa(messages),
                color_scheme="green",
                on_change=State.removing_task(messages),
            )
        )





    )

def heading() ->rx.Component:
    return rx.heading("Schedule with Me",
                   style=style.topic_style)
def inputingTask() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.input(placeholder='Input a task',
                     style=style.input_style,
                     on_blur= State.set_task,),
            rx.button("Enter",
                      is_loading=State.processing,
                      on_click = State.predictTime,
                      style=style.button_style,)

        ),
        rx.hstack(
            rx.select(
                month,
                placeholder="Select a Month",
                on_change=State.set_month,
                color_scheme ='twitter',
                size ='md'

            ),
            rx.number_input(
                min_=1,
                max_=31,
                on_change = State.set_day,

            )

        )
    )



def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            heading(),
            rx.cond(
                State.complete,
                TodoList(),
            ),
            inputingTask(),




        )
    )



# Add state and page to the app.
app = rx.App()
app.add_page(index, title = "Schedule With Me")
app.compile()
