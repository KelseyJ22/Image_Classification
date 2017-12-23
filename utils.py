def pre_process_image(image):
    # flip some images
    image = tf.image.random_flip_left_right(image)
    
    # randomly adjust hue, contrast and saturation
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    # limit pixel between [0, 1] in case of overflow
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)

    return image