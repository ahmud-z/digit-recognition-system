import PIL
import gradio as gr
import tensorflow as tf

model = tf.keras.models.load_model("my_model.keras")


def recognize_digit(image):

    if image is not None:
        image = 255 - image  # Inverting image

        image = image.reshape((1, 28, 28, 1)).astype("float32") / 255

        prediction = model.predict(image)

        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ""


with gr.Blocks() as demo:
    gr.HTML("<center><h1>Real-Time Digit Recognition System</h1></center>")
    gr.HTML("<center>Author: <a href='#'>Ahmudul Hossain</a></center>")
    gr.HTML("<center><p>Student ID: 213002200</p></center>")
    gr.HTML("<center><p>Dept. of CSE (GUB)</p></center>")
    

    gr.Interface(
        fn=recognize_digit,
        inputs=gr.Image(
            image_mode="L",
            sources=["upload", "webcam", "clipboard"],
            height=300,
            width=300,
        ),
        outputs=gr.Label(num_top_classes=5),
        live=True,
    )

if __name__ == "__main__":
    demo.launch()
