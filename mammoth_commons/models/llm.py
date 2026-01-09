from mammoth_commons.models.model import Model
import requests


class LLM(Model):
    def __init__(self, name, url="http://localhost:11434"):
        super().__init__()
        self.name = name
        self.url = url.rstrip("/")

    def prompt(self, context, prompt):
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            "stream": False,  # disable streaming for simplicity
        }
        response = requests.post(f"{self.url}/api/chat", json=payload)
        if response.status_code != 200:
            raise Exception(f"{response.status_code}: {response.text}")

        data = response.json()
        return data["message"]["content"]
