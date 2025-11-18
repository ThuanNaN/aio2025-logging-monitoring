import gradio as gr
from yolo_func import detect_objects


with gr.Blocks() as demo:
    gr.Markdown("Object detection using YOLO11")
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
    with gr.Row():
        detect_btn = gr.Button("Detect Objects")

    with gr.Row():
        annotated_image = gr.Image(label="Annotated Image")
    with gr.Row():
        detection_results = gr.Textbox(label="Detection Results")

    # Bind detection function
    detect_btn.click(
        fn=detect_objects, 
        inputs=input_image, 
        outputs=[annotated_image, detection_results]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
