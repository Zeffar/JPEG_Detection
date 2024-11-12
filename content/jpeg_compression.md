JPEG Compression Detection

![](/images/2YM_Image_1.png)

**Image Preprocessing -**

**Preparing images for training**

**??????**

**Overview**

In order to be able to create a Naive Bayes Classifier, we first need to prepare the images that will be used for training, and for that we need to compress them and extract the information that will be needed by the ?Signal Processing? part of the project.

This doc aims to explain the process.

**Function Documentation and Explanations**

**0. Imports**

`import`` ``os`

`import`` ``cv2`

`import`` ``numpy`` ``as`` ``np`

`from`` ``PIL`` ``import`` ``Image`

I will be using these imports throughout the code

1. **Compression function**

```
def compress(path, quality=100):
	img = Image.open(path)
	img = img.convert('RGB')
	img.save('temp_jpeg.jpg',quality = quality)
	return Image.open("temp_jpeg.jpg")
```
Loads the image and creates a temporary jpg file that may or may not be compressed.

**Explanation: **when using the python function img.save(), you can specify the quality parameter, which, when different from 100, compresses the image with a compression method that is determined based on the file format.

2. **Generating training data**

Generating training data
```
image_number = 700 # number of images
quality_peak_list=[]

for q in range(10,101,10):
	l=[]
	print(q)
	for i in range(0, image_number):          	# each image has 4 versions
    	file_name = f"{i + 1:04}x4w1.png"
    	path = os.path.join("./DIV2K_train_LR_wild", file_name)
   	 
    	img = compress(path, q)

    	img = img.convert('YCbCr')
    	img = zero_pad(np.array(img)[:,:,0])
    	img = img.astype(np.float64)
    	dct = DCT(img)[0,0,:]
    	hcount, bin_edges = hist(dct)
    	fft = FFT(hcount)
    	_, avg_peak_dist,_ = quality(fft)
    	l.append(avg_peak_dist)
	quality_peak_list.append(l)
```
Compresses the data set at different quality levels, then extracts the information that the classifier will need and places it in a vector.

**Explanation**: the training data set has 700 images that are compressed at different quality levels, ranging from 10 to 100(meaning they are not compressed at all), then the image is converted to an appropriate format for a jpeg. Finally, we extract the information we require from the image using the algorithms described in the ?Signal Processing? section.

3. **Testing the classifier**

```
number_of_tests = 100
testing_cutoff = 700 # images from 700 onwards are for testing

error=0

for i in range(0, number_of_tests):
    	for j in range(1, 5):
       	 
        	file_name = f"{testing_cutoff + i + 1:04}x4w1.png"
        	path = os.path.join("DIV2K_train_LR_wild", file_name)
       	 
        	qual_cat = np.random.randint(1,11)
   	 
        	img = compress(path, qual_cat*10)
        	img = img.convert('YCbCr')
        	img = zero_pad(np.array(img)[:,:,0])
        	img = img.astype(np.float64)
        	dct = DCT(img)[0,0,:]
        	hcount, bin_edges = hist(dct)
        	fft = FFT(hcount)
        	_, avg_peak_dist,_ = quality(fft)
        	print(avg_peak_dist)
        	cat=most_likely_category(avg_peak_dist)
        	print(qual_cat*10, cat*10)
        	error += np.abs(cat*10-qual_cat*10)
       	 
       	 
print(f"average error: {error / number_of_tests/4} ") # we do 4 samples for each test
```
Similar to ?Generating training data?.

**Explanation**: images from a certain range are chosen and compressed to a level that is arbitrarily selected. Then, like before, we extract the information we need using signal processing, pass that information to the classifier which gives its estimates, print them and if they are wrong we increase the error rate.
