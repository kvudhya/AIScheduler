shadow = "rgba(0, 0, 0, 0.15) 0px 2px 8px"
chat_margin = "10%"
message_style = dict(
    padding="1em",
    border_radius="5px",
    margin_y="0.5em",
    box_shadow=shadow,
    max_width="10em",
    display="inline-block",
)

# Set specific styles for questions and answers.
question_style = message_style | dict(
    bg="#F5EFFE", margin_left=chat_margin
)
answer_style = message_style | dict(
    bg="#DEEAFD", margin_right=chat_margin
)

# Styles for the action bar.
input_style = dict(
    border_width="1px", padding="1em", box_shadow=shadow
)
button_style = dict(bg="#CEFFEE", box_shadow=shadow)


style = {
    "background-color": "#454545",
    "font_family": "Comic Sans MS",
    "font_size": "16px",
}

topic_style = {
    "color": "blue",
    "font_family": "Comic Sans MS",
    "font_size": "3em",
    "font_weight": "bold",
    "background Image ": "rgba(240, 46, 170, 0.4) 5px 5px, rgba(240, 46, 170, 0.3) 10px 10px",
    "margin-bottom": "3rem",
}

textarea_style = {
    "color": "white",
    "width": "150%",
    "height": "20em",
}


openai_input_style = {
    "color": "white",
    "margin-top": "2rem",
    "margin-bottom": "1rem",
}


submit_button_style = {
    "margin-left": "30%",
}


summary_style = {
    "color": "white",
    "margin-top": "2rem",
}