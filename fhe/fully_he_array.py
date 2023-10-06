import base64
from Pyfhel import Pyfhel, PyCtxt
import numpy as np
import os


class FHE_ARRAY:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.HE = Pyfhel(context_params={'scheme': 'bfv', 'n': 2**14, 't_bits': 32})
        self.HE.load_public_key(current_dir + "/pub_i.key")
        self.HE.load_secret_key(current_dir + "/sec_i.key")

    def encrypt_int_array(self , arr):
        c = self.HE.encrypt(arr)
        return c  

    def decrypt_int_array(self , ciphertext):
        decrypted_value = self.HE.decryptInt(ciphertext)
        return decrypted_value
    
    def multiply(self , ciphertext1 , ciphertext2):
        c_product = ciphertext1 * ciphertext2
        return c_product
    
    def add_ciphertexts(self, ciphertext1, ciphertext2):
        c_sum = ciphertext1 + ciphertext2
        return c_sum
    
    def ciphertext_to_pyctxt(self , ciphertext):
        decoded_base64 = base64.b64decode(ciphertext.encode())
        c = PyCtxt(pyfhel = self.HE, bytestring=decoded_base64)
        return c
    
    def ciphertext_to_bytestring(self , ciphertext):
        c1 = ciphertext.to_bytes()
        c = base64.b64encode(c1)
        return c1

# simple test , will be deleted
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def image_to_vector(raw_image , pixel):
    image = raw_image.convert('RGB')
    image = image.resize((pixel , pixel))
    image_array = np.array(image)
    image_flatten = image_array.flatten()            
    return image_flatten

def vector_to_image(arr):
    r , g , b = pixel_vector_to_3vector(arr)
    size = r.size
    pixel = int(size**(1/2))
    rgb_array = np.zeros([pixel , pixel ,3])
    for i in range(0 , pixel):
        for j in range(0 ,pixel):
            rgb_array[i][j][0] = r[i * pixel + j]
            rgb_array[i][j][1] = g[i * pixel + j]
            rgb_array[i][j][2] = b[i * pixel + j]
    image = Image.fromarray(rgb_array.astype('uint8'))
    return image

def pixel_vector_to_3vector(arr):
    size = int(arr.size / 3)
    r = np.array([arr[3*i] for i in range(0 , int(size))])
    g = np.array([arr[3*i+1] for i in range(0 , int(size))])
    b = np.array([arr[3*i+2] for i in range(0 , int(size))])
    return [r,g,b]

def greyscale_vector_to_picture(arr ,  width , height):
    grayscale_vector = arr
    grayscale_image = Image.new('L', (width, height))
    grayscale_image.putdata(grayscale_vector)
    return grayscale_image

# simply test here
if __name__ == "__main__":
    pixel = 96

    # im1 = Image.open('tt1.png')
    # im2 = Image.open('tt2.png')
    # vector_of_difference = image_to_vector(im1, pixel) - image_to_vector(im1, pixel)
    # ans = 0
    # for i in vector_of_difference:
    #     ans *= i
    # print(i)
    # # print(image_to_vector(im1, pixel)[200 : 220])
    # # print(image_to_vector(im1, pixel)[200 : 220])
    

    raw_image = Image.open('..//cool_guy.png')

    plt.imshow(raw_image)
    plt.show()

    v = image_to_vector(raw_image , pixel)
    v2 = pixel_vector_to_3vector(v) 

    he = FHE_ARRAY()
    ciphertext1 = he.encrypt_int_array(v2[0])
    ciphertext2 = he.encrypt_int_array(v2[1])
    ciphertext3 = he.encrypt_int_array(v2[2])

    scale1 = np.full(pixel*pixel , 60 ,  dtype = np.int64)
    scale2 = np.full(pixel*pixel , 113 , dtype = np.int64)
    scale3 = np.full(pixel*pixel , 27 , dtype = np.int64)

    multi1 = he.encrypt_int_array(scale1)
    multi2 = he.encrypt_int_array(scale2)
    multi3 = he.encrypt_int_array(scale3)

    ciphertext1 = he.multiply(ciphertext1 , multi1)
    ciphertext2 = he.multiply(ciphertext2 , multi2)
    ciphertext3 = he.multiply(ciphertext3 , multi3)

    ciphertext = he.add_ciphertexts(he.add_ciphertexts(ciphertext1 , ciphertext2) , ciphertext3)
    # poly = ciphertext.getPoly()
    # print(poly)

    byte_representation_of_ciphertext = he.ciphertext_to_bytestring(ciphertext)
    adversary_array = np.frombuffer(byte_representation_of_ciphertext, dtype=np.uint8)
    adversary_image = vector_to_image(adversary_array[-(pixel**2)*3-11:-10])
    # adversary_image.save('tt2.png')
    plt.imshow(adversary_image)
    plt.show()

    decyphertext = (he.decrypt_int_array(ciphertext)) / 200
    answer_text = decyphertext.astype(int)
    
    image = greyscale_vector_to_picture(decyphertext[0 : pixel*pixel] , pixel , pixel)
    plt.imshow(image, cmap=cm.Greys_r)
    plt.show()