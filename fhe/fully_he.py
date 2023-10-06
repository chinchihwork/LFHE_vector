import base64
from Pyfhel import Pyfhel, PyCtxt
import numpy as np
import os

class FullyHE:
    def __init__(self, _type):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.HE = Pyfhel(context_params={'scheme':'CKKS', 'n':2**14, 'scale': 2**30, 'qi_sizes': [60, 30, 30, 30, 60]})
        # self.HE.keyGen()
        # self.HE.save_public_key(current_dir + "/pub_f.key")
        # self.HE.save_secret_key(current_dir + "/sec_f.key")
        # self.HE.rotateKeyGen()
        # self.HE.save_rotate_key(current_dir + "/rotate_f.key")
        # self.HE.relinKeyGen()
        if _type == "int":
            self.HE = Pyfhel(context_params={'scheme': 'bfv', 'n': 2**14, 't_bits': 32})
            self.HE.load_public_key(current_dir + "/pub_i.key")
            self.HE.load_secret_key(current_dir + "/sec_i.key")
            self.HE.load_relin_key(current_dir + "/relin_i.key")
            self.HE.load_rotate_key(current_dir + "/rotate_i.key")
        else:
            self.HE = Pyfhel(context_params={'scheme':'CKKS', 'n':2**14, 'scale': 2**30, 'qi_sizes': [60, 30, 30, 30, 60]})
            self.HE.load_public_key(current_dir + "/pub_f.key")
            self.HE.load_secret_key(current_dir + "/sec_f.key")
            self.HE.load_rotate_key(current_dir + "/rotate_f.key")

    def encrypt_int(self, text):
        int_value = int(float(text))
        c = self.HE.encrypt(np.array([int_value]))
        ciphertext = base64.b64encode(c.to_bytes()).decode()
        return ciphertext

    def decrypt_int(self, ciphertext):
        decoded_base64 = base64.b64decode(ciphertext.encode())
        c = PyCtxt(pyfhel=self.HE, bytestring=decoded_base64)
        decrypted_value = self.HE.decryptInt(c)[0]
        return str(decrypted_value)

    def add_ciphertexts(self, ciphertext1, ciphertext2):
        c1 = self.ciphertext_to_pyctxt(ciphertext1)
        c2 = self.ciphertext_to_pyctxt(ciphertext2)
        c_sum = c1 + c2
        add_ciphertext = base64.b64encode(c_sum.to_bytes()).decode()
        return add_ciphertext

    def subtract_ciphertexts(self, ciphertext1, ciphertext2):
        c1 = self.ciphertext_to_pyctxt(ciphertext1)
        c2 = self.ciphertext_to_pyctxt(ciphertext2)
        c_diff = c1 - c2
        subtract_ciphertext = base64.b64encode(c_diff.to_bytes()).decode()
        return subtract_ciphertext

    def multiply_ciphertexts(self, ciphertext1, ciphertext2):
        c1 = self.ciphertext_to_pyctxt(ciphertext1)
        c2 = self.ciphertext_to_pyctxt(ciphertext2)
        c_product = c1 * c2
        multiply_ciphertext = base64.b64encode(c_product.to_bytes()).decode()
        return multiply_ciphertext

    def encrypt_float(self, text):
        float_value = float(text)
        arr_x = np.array([float_value], dtype=np.float64)
        c = self.HE.encryptFrac(arr_x)
        ciphertext = base64.b64encode(c.to_bytes()).decode()
        return ciphertext

    def decrypt_float(self, ciphertext):
        decoded_base64 = base64.b64decode(ciphertext.encode())
        c = PyCtxt(pyfhel = self.HE, bytestring=decoded_base64)
        decrypted_value = self.HE.decryptFrac(c)[0]
        return str(decrypted_value)
    
    def ciphertext_to_pyctxt(self , ciphertext):
        decoded_base64 = base64.b64decode(ciphertext.encode())
        c = PyCtxt(pyfhel = self.HE, bytestring=decoded_base64)
        return c    

    def encrypt_int_array(self , arr):
        c = self.HE.encrypt(arr)
        ciphertext = base64.b64encode(c.to_bytes()).decode()
        return ciphertext
    

    def decrypt_int_array(self , ciphertext):
        decoded_base64 = base64.b64decode(ciphertext.encode())
        c = PyCtxt(pyfhel=self.HE, bytestring=decoded_base64)
        decrypted_value = self.HE.decryptInt(c)
        return decrypted_value
    
    def encrypt_float_array(self , arr):
        c = self.HE.encrypt(arr)
        ciphertext = base64.b64encode(c.to_bytes()).decode()
        return ciphertext
    

    def decrypt_float_array(self , ciphertext):
        decoded_base64 = base64.b64decode(ciphertext.encode())
        c = PyCtxt(pyfhel=self.HE, bytestring=decoded_base64)
        decrypted_value = self.HE.decryptFrac(c)
        return decrypted_value



# simple test , will be deleted
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def image_to_vector(image_path):
    image = Image.open(image_path)
    image = image.resize((24 , 24))
    image_array = np.array(image)
    image_flatten = image_array.flatten()
    return image_flatten

def pixel_vector_to_3vector(arr):
    size = int(arr.size / 3)
    r = np.array([arr[3*i] for i in range(0 , int(size))])
    g = np.array([arr[3*i+1] for i in range(1 , int(size))])
    b = np.array([arr[3*i+2] for i in range(2 , int(size))])
    return [r,g,b]

def greyscale_vector_to_picture(arr ,  width , height):
    grayscale_vector = arr
    grayscale_image = Image.new('L', (width, height))
    grayscale_image.putdata(grayscale_vector)
    return grayscale_image

if __name__ == "__main__":

    v = image_to_vector('..//test_image.jpg')
    v2 = pixel_vector_to_3vector(v)
    print(v2[0])
    # print(v2[1])
    # print(v2[2])    

    he = FullyHE('float')
    ciphertext1 = he.encrypt_int_array(v2[0])
    ciphertext2 = he.encrypt_int_array(v2[1])
    ciphertext3 = he.encrypt_int_array(v2[2])

    multi1 = np.full(60 , 24*24 , dtype = np.int64)
    multi2 = np.full(60 , 24*24 , dtype = np.int64)
    multi3 = np.full(60 , 24*24 , dtype = np.int64)

    ct4 = he.encrypt_int_array(multi1)
    ct5 = he.encrypt_int_array(np.full(117 , 24*24 , dtype = np.int64))
    ct6 = he.encrypt_int_array(np.full(23 , 24*24 , dtype = np.int64))

    ciphertext = ciphertext1 * ciphertext2
    
    # decyphertext1 = (he.decrypt_float_array(ciphertext1))
    # decyphertext2 = (he.decrypt_float_array(ciphertext2))
    # decyphertext3 = (he.decrypt_float_array(ciphertext3))

    # print(decyphertext1.astype(int))
    # print(decyphertext2.astype(int))

    decyphertext = (he.decrypt_float_array(ciphertext)) / 20
    decyphertext = decyphertext.astype(int)
    print(decyphertext)
    # image = greyscale_vector_to_picture(decyphertext , 24 , 24)
    # plt.imshow(image, cmap=cm.Greys_r) 
    # plt.show()
    