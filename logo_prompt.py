import pdb
from io import BytesIO

import gymnasium as gym
import random

import torch
from google.genai import types
import torch.nn.functional as F
from PIL import Image


class RefineLogoPromptEnv(gym.Env):
    def __init__(self, tokenizer, text_gen_model, client, feature_extractor, vit_model, metadata):
        super(RefineLogoPromptEnv, self).__init__()

        self.tokenizer = tokenizer
        self.text_gen_model = text_gen_model
        self.client = client
        self.feature_extractor = feature_extractor
        self.vit_model = vit_model
        self.metadata = metadata

        # Observation Space, Action Space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1024,), dtype=float)
        self.actions = [
            'Reposition text, elements and descriptions',
            'Assign a color to each element in the Elements section based on the Colors section only',
            'Condense the text',
            'Change the style',
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.t = 0
        self.reset()

    def _generate_initial_state(self, metadata):
        state = 'Design a logo with {'
        if metadata[0] is not None:
            state += f'Text: {metadata[0]} '
        if metadata[1] is not None:
            state += f'Elements: {metadata[1]} '
        if metadata[2] is not None:
            state += f'Description: {metadata[2]} '
        if metadata[3] is not None:
            state += f'Colors: {metadata[3]}'
        return state + '}'

    def _state_embedding(self, state):
        input_ids = self.tokenizer(state, return_tensors='pt').input_ids
        output = self.text_gen_model.encoder(input_ids)
        hidden_state = output.last_hidden_state
        return hidden_state.mean(dim=1).detach()

    def reset(self, seed=None, options=None):
        # Sample from data randomly
        metadata = random.choice(self.metadata)
        self.native_state = self._generate_initial_state(metadata)
        self.state = self._state_embedding(self.native_state)
        self.target_logo = metadata[4]
        return self.state, {}

    def _next_state(self, state, action):
        input_text = f'{action} in the following logo prompt (You must keep the structure of the prompt, only return the prompt): "{state}"'

        response = self.client.models.generate_content(
            model='gemini-2.0-flash', contents=input_text
        )
        text = response.text
        return text
        # print(input_text)
        # input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids
        # outputs = self.text_gen_model.generate(input_ids, max_length=1024)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _logo_embedding(self, logos):
        vit_inputs = self.feature_extractor(images=logos, return_tensors='pt')
        with torch.no_grad():
            outputs = self.vit_model(**vit_inputs)
        return outputs.last_hidden_state

    def _compute_rewards(self, state):
        logos = []
        while True:
            # Generate logo and compute rewards
            response = self.client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=state,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                )
            )
            if response.generated_images is None:
                continue
            for logo_data in response.generated_images:
                logo = Image.open(BytesIO(logo_data.image.image_bytes))
                logos.append(logo)
            break

        generated_embedding = self._logo_embedding(logos)
        target_logo = Image.open(self.target_logo)
        target_embedding = self._logo_embedding(target_logo)

        return torch.mean(F.cosine_similarity(generated_embedding, target_embedding)), logos

    def step(self, a):
        action = self.actions[a]

        # State transitions
        self.native_state = self._next_state(self.native_state, action)
        print(f'{self.t}: {self.native_state}')
        self.state = self._state_embedding(self.native_state)

        reward, logos = self._compute_rewards(self.native_state)
        self.t += 1
        done = True if reward >= 0.9 else False

        return self.state, reward, done, False, logos

    def render(self):
        target = Image.open(self.target_logo)
        target.show()


