```python
#code for testing the model and running predictions on new images
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

#loading previously saved model
classifier = load_model('pneumonia_model.h5')

#loading new image from directory
test_image = image.load_img('IMAGE.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#making detection
result = classifier.predict(test_image)
if result[0][0] == 1:
    print('Pneumonia Detected')
else:
    print('No Pneumonia Detected')
```
