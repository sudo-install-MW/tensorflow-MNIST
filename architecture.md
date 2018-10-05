<p>
The parameters were calculated using the formula below
</p><p>
OUT = (W - F + 2P)/2 + 1<br>
P = (F -1)/2
</p>

```
# layer 1 = input
input dims = 28x28x1
num of parameters = 784

# layer 2 = conv2d 3x3x32
filter dims = 3x3
num of filter = 32
stride = 1
P = (3 - 1)/2
OUT = 28x28x32
num of parameter = 


# layer 2 out = 28x28x32

# layer 3 = mp 2x2
# layer 3 out = 14x14x32

# layer 4 = conv2d 3x3x64
# layer 4 out = 14x14x64

# layer 5 = mp 2x2
# layer 5 out = 7x7x64

# layer 6 = fc 512
# layer 7 = fc 10
```