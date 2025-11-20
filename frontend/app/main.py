import gradio as gr
from yolo_func import detect_objects
from vqa_func import answer_question


with gr.Blocks(title="AI Model Demo - YOLO & VQA") as demo:
    gr.Markdown("# AI Model Demo: Object Detection & Visual Question Answering")
    gr.Markdown("Choose between YOLO object detection or BLIP visual question answering")
    
    with gr.Tabs():
        # YOLO Object Detection Tab
        with gr.Tab("YOLO Object Detection"):
            gr.Markdown("### Upload an image to detect objects using YOLO11")
            with gr.Row():
                yolo_input_image = gr.Image(type="pil", label="Upload Image")
            
            with gr.Row():
                yolo_detect_btn = gr.Button("Detect Objects", variant="primary")
            
            with gr.Row():
                yolo_annotated_image = gr.Image(label="Annotated Image")
            
            with gr.Row():
                yolo_detection_results = gr.Textbox(
                    label="Detection Results",
                    lines=10,
                    max_lines=20
                )
            
            # Bind YOLO detection function
            yolo_detect_btn.click(
                fn=detect_objects,
                inputs=yolo_input_image,
                outputs=[yolo_annotated_image, yolo_detection_results]
            )
        
        # VQA Tab
        with gr.Tab("Visual Question Answering"):
            gr.Markdown("### Upload an image and ask a question using BLIP")
            with gr.Row():
                with gr.Column(scale=1):
                    vqa_input_image = gr.Image(type="pil", label="Upload Image")
                    vqa_question = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the image...",
                        lines=2
                    )
                    with gr.Row():
                        vqa_max_length = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=10,
                            label="Max Answer Length"
                        )
                        vqa_num_beams = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Beams"
                        )
                    vqa_answer_btn = gr.Button("Get Answer", variant="primary")
                
                with gr.Column(scale=1):
                    vqa_answer = gr.Textbox(
                        label="Answer",
                        lines=3,
                        max_lines=5
                    )
                    vqa_metadata = gr.Markdown(label="Metadata")
            
            # Bind VQA function
            vqa_answer_btn.click(
                fn=answer_question,
                inputs=[vqa_input_image, vqa_question, vqa_max_length, vqa_num_beams],
                outputs=[vqa_answer, vqa_metadata]
            )
    
    gr.Markdown("---")
    gr.Markdown("🔍 **YOLO**: Detects objects in images | 💬 **VQA**: Answers questions about images")

demo.launch(server_name="0.0.0.0", server_port=7860)
