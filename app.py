import gradio as gr
from detector import detect_deepfake

demo = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Sports Deepfake Detector",
    description="Upload a sports image to test the app."
)

if __name__ == "__main__":
    demo.launch()