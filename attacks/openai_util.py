import openai
import time

class OpenAIAPI:
    def __init__(self, temperature, system_content):
        openai.api_key = ""
        self.temperature = temperature
        self.system_content = system_content
    
    def get_embedding(self, text):
        while True:
            try:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[text]
                )
                break
            except:
                print('error')
                time.sleep(10)
        return response['data'][0]['embedding']
            

    def get_result_from_gpt4(self, query):
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response
    
    def get_result_from_gpt3(self, query):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response

    def rewrite(self, query):
        text = "Rewrite the following paragraph:\n "
        result = text + query
        while True:
            try:
                result = self.get_result_from_gpt3(result)
                break
            except:
                print('error')
                time.sleep(10)
        return result['choices'][0]['message']['content']

if __name__ == "__main__":
    openai_util = OpenAIAPI(0.2, "Your are a helpful assistant to rewrite the text.")
    text =  "Rewrite the following paragraph:\n "
    text += "Whoever gets him, they'll be getting a good one,\" David Montgomery said. INDIANAPOLIS \u2014 Hakeem Butler has been surrounded by some of the best wide receivers on the planet this month, so it's only natural he'd get a touch of star-struck when he got a chance to meet Randy Moss, a Hall of Famer and someone Butler idolized growing up and still considers an inspiration to this day.\n\"It's crazy to see someone you've watched on TV and seen on YouTube and then be standing right there and you're like, 'Man, he's just a regular dude,'\" Butler, a 6-foot 5, 227-pound receiver from Iowa State, said Thursday at the NFL Scouts' Luncheon, which kicks off the week of the NFL Scouts' Breakfast and the NFL Scouts' Luncheon at the JW Marriott Indianapolis, which takes place Sunday and Monday, Feb. 1 and 2, at Lucas Oil Stadium, 1 Lucas Oil Stadium Dr., Indianapolis, IN 46225, and which is open to all NFL Scouts and team personnel and others with a pass from the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is sponsored by the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is organized by the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is hosted by the"
    result = openai_util.get_result_from_gpt3(text)