from fastai.vision.all import *
from fastai.vision.all import load_learner
import gradio as gr

mammal_labels = ('Aardvark', 'African Wild Dog', 'Armadillo', 'Bat', 'Blue Whale', 'Cat', 'Dolphin', 'Elephant', 'Fox', 'Hippopotamus', 'Kangaroo', 'Lion', 'Meerkat', 'Monkey', 'Mountain Gorilla', 'Mouse', 'Panda', 'Pig', 'Sheep', 'Wolf')

model = load_learner("mammal_model.pkl")

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    print(pred)
    return dict(zip(mammal_labels, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label(num_top_classes=5)  # Remove the unused parameter 'type'
examples = [
    'image_1.jpg',
    'image_2.jpg',
    'image_3.jpg',
    'image_4.jpeg',
    'image_5.jpeg'
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)