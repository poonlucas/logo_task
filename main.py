import os
import json
import pdb
import re
from google import genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, ViTFeatureExtractor, ViTModel
from logo_prompt import RefineLogoPromptEnv
from ppo import PPO


def load_data():
    data = []
    for meta_dir in os.listdir(os.path.join('data', 'metadata')):
        dir_num = meta_dir[4:]
        logo_a = os.listdir(os.path.join('data', 'logo', f'logo{dir_num}a'))
        logo_b = os.listdir(os.path.join('data', 'logo', f'logo{dir_num}b'))
        for file in os.listdir(os.path.join('data', 'metadata', meta_dir)):
            if file.endswith('.json'):
                with open(os.path.join('data', 'metadata', meta_dir, file)) as f:
                    meta_info = json.load(f)

                    text = meta_info['mark_literal_elements']
                    design_search_code = meta_info['design_search_code']
                    objects = " ".join(
                        [f'\'{obj}\'' for obj in re.sub(r'.*- ', '', design_search_code.lower()).split('\n')])
                    description = None if meta_info['description_of_mark'] is None else meta_info[
                        'description_of_mark'].lower()
                    color_claimed = meta_info['color_claimed']
                    colors = ['black', 'white']
                    if color_claimed is not None:
                        color_claimed = color_claimed.removeprefix('The color(s) ')
                        color_claimed = color_claimed.removesuffix(' is/are claimed as a feature of the mark.')
                        color_claimed = re.sub(' and ', ', ', color_claimed).lower()
                        colors = color_claimed.split(', ')
                    if f'{file[:-5]}.png' in logo_a:
                        data.append((text, objects, description, colors,
                                     os.path.join('data', 'logo', f'logo{dir_num}a', f'{file[:-5]}.png')))
                    elif f'{file[:-5]}.png' in logo_b:
                        data.append((text, objects, description, colors,
                                     os.path.join('data', 'logo', f'logo{dir_num}b', f'{file[:-5]}.png')))
    return data


if __name__ == '__main__':
    metadata = load_data()

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    text_gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    client = genai.Client(api_key='')

    env = RefineLogoPromptEnv(
        tokenizer=tokenizer,
        text_gen_model=text_gen_model,
        client=client,
        feature_extractor=feature_extractor,
        vit_model=vit_model,
        metadata=metadata
    )

    pi = PPO(env, num_steps=8, num_minibatches=4)
    pi.train(total_timesteps=128)
    pi.eval(num_episodes=2, render=True)
    pi.save()
