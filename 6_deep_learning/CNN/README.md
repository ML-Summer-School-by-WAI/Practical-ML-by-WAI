### Convolutional Neural Network (CNN)
Please see this for Slides
https://github.com/tharhtetsan/zero_2_hero_ml/tree/main/7_Deep_Learning/CNN




### Convolutional Layers
```bash
  parameters=( F×F × C_in +1 ) × C_out
```
​

- F: Filter size (height and width, assuming square filters).
- C_in : Number of input channels.
- 𝐶_out: Number of output channels (or filters).
- 1: This accounts for the bias term for each output channel.


###  Fully Connected (Dense) Layers
```bash
  parameters=( N_in + 1 ) × N_out
```
​- N_in: Number of input neurons.
- 𝑁_out : Number of output neurons.
- 1: This accounts for the bias term for each output neuron.






### Why CNN?
Analogy: The human visual system. We don't process a whole scene at once. Our eyes scan and recognize small, local patterns (edges, corners, curves). We then combine these patterns to recognize more complex objects.

CNNs do the same thing by using two clever ideas:
    -  Local Receptive Fields: Look at small patches of the image at a time.
    -   Parameter Sharing: Use the same pattern detector (a "filter") across the entire image. If a horizontal edge detector is useful in the top-left corner, it's probably useful elsewhere too.

### What is CNN?
-   Convolution Operation: just an element-wise multiplication (in ANN we use as W.X)
-   Padding: Adding zeros around the border of the input image. Why?
    -   It allows the output feature map to have the same size as the input.
    -   It gives more attention to the pixels at the edges of the image.

- Activation Function : We normally use ReLU f(x)=max(0,x). Why?
    -   It acts like an "on/off" switch and it just replaces all negative pixel values in the feature map with zero.

- Pooling Layer : To reduce the spatial size of feature maps, it makes the feature detection more robust to small shifts and distortions in the image (this is called translation invariance in research paper).
    How it work : like summarizing a region by its most prominent feature. If a feature was strongly detected anywhere in that region, the output will reflect that.



