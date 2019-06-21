import cv2
import numpy as np
import tensorflow as tf 


def cartoonize(image_path):
    interpreter = tf.lite.Interpreter(model_path="cartoon_gan.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256),
						interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)/127.5 - 1
    image = image.reshape(input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = (np.squeeze(output)+1)*127.5
    output = np.clip(output, 0, 255)
    
    cv2.imwrite('cartoon_image.jpg', output)


if __name__ == '__main__':
	image_path = 'test_image.jpg'
	cartoonize(image_path)