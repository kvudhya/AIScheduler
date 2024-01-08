from openai import OpenAI

import sys


client = OpenAI(api_key="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")
def chatGPTprompt():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages= [
            {"role": "system", 'content': "you are a student either in high school or college. "},
            {"role": "user", "content": "create a list of 1 event that student would put on his or her todolist: give only the events"}
        ]
    )
    return response

print(chatGPTprompt().choices[0].message.content)