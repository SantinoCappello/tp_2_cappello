import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import sys
from scipy.ndimage import convolve

def main():
    """
    Descripción breve:
    Función principal que gestiona la entrada del usuario, procesa la imagen y muestra el resultado.

    Argumentos:
    Ninguno

    Returns:
    Ninguno
    """
    # Solicitar la ruta de la imagen al usuario
    image_path = input("Ingrese la ruta a la imagen: ")
    try:
        img = Image.open(image_path)
    except IOError:
        print("Imagen no encontrada")
        sys.exit()

    # Solicitar las nuevas dimensiones
    try:
        new_width = int(input("Ingrese el nuevo ancho de la imagen: "))
        new_height = int(input("Ingrese el nuevo alto de la imagen: "))
    except ValueError:
        print("Ancho o alto inválido")
        sys.exit()

    # Convertir la imagen a un arreglo de NumPy
    img_array = np.array(img)
    original_shape = img_array.shape

    # Lista para almacenar las imágenes intermedias
    images = [Image.fromarray(img_array.astype('uint8'))]

    # Reducir el ancho
    if new_width < img_array.shape[1]:
        for i in range(img_array.shape[1] - new_width):
            energy_map = compute_energy(img_array)
            cumulative_map = cumulative_energy_map(energy_map, axis=1)
            seam = find_seam(cumulative_map, axis=1)
            img_array = remove_seam(img_array, seam, axis=1)
            images.append(Image.fromarray(img_array.astype('uint8')))

    # Reducir el alto
    if new_height < img_array.shape[0]:
        img_array = np.transpose(img_array, (1, 0, 2))
        for i in range(img_array.shape[1] - new_height):
            energy_map = compute_energy(img_array)
            cumulative_map = cumulative_energy_map(energy_map, axis=1)
            seam = find_seam(cumulative_map, axis=1)
            img_array = remove_seam(img_array, seam, axis=1)
            images.append(Image.fromarray(np.transpose(img_array, (1, 0, 2)).astype('uint8')))
        img_array = np.transpose(img_array, (1, 0, 2))

    # Mostrar la interfaz gráfica
    show_images(images, original_shape)

def compute_energy(img_array):
    """
    Descripción breve:
    Calcula el mapa de energía de la imagen utilizando el operador de Sobel.

    Argumentos:
    - img_array: Arreglo NumPy que representa la imagen.

    Returns:
    - energy: Mapa de energía normalizado de la imagen.
    """
    img_gray = rgb2gray(img_array)

    # Definir los kernels de Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype='int32')

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype='int32')

    # Aplicar los filtros de Sobel
    energy_x = convolve(img_gray, sobel_x)
    energy_y = convolve(img_gray, sobel_y)

    # Calcular la energía
    energy = np.hypot(energy_x, energy_y)
    energy = energy / energy.max()
    return energy

def cumulative_energy_map(energy, axis):
    """
    Descripción breve:
    Calcula el mapa de energía acumulada para encontrar la ruta de mínima energía.

    Argumentos:
    - energy: Mapa de energía de la imagen.
    - axis: Dirección de la eliminación de la costura (1 para vertical, 0 para horizontal).

    Returns:
    - cumulative_map: Mapa de energía acumulada.
    """
    h, w = energy.shape
    cumulative_map = energy.copy()

    if axis == 1:  # Costura vertical
        for i in range(1, h):
            for j in range(w):
                if j == 0:
                    min_energy = min(cumulative_map[i-1, j], cumulative_map[i-1, j+1])
                elif j == w - 1:
                    min_energy = min(cumulative_map[i-1, j-1], cumulative_map[i-1, j])
                else:
                    min_energy = min(cumulative_map[i-1, j-1], cumulative_map[i-1, j], cumulative_map[i-1, j+1])
                cumulative_map[i, j] += min_energy
    else:  # Costura horizontal
        for i in range(1, w):
            for j in range(h):
                if j == 0:
                    min_energy = min(cumulative_map[j, i-1], cumulative_map[j+1, i-1])
                elif j == h - 1:
                    min_energy = min(cumulative_map[j-1, i-1], cumulative_map[j, i-1])
                else:
                    min_energy = min(cumulative_map[j-1, i-1], cumulative_map[j, i-1], cumulative_map[j+1, i-1])
                cumulative_map[j, i] += min_energy
    return cumulative_map

def find_seam(cumulative_map, axis):
    """
    Descripción breve:
    Encuentra la costura de mínima energía en el mapa de energía acumulada.

    Argumentos:
    - cumulative_map: Mapa de energía acumulada.
    - axis: Dirección de la eliminación de la costura (1 para vertical, 0 para horizontal).

    Returns:
    - seam: Lista de coordenadas que representan la costura de mínima energía.
    """
    h, w = cumulative_map.shape
    if axis == 1:
        seam = []
        j = np.argmin(cumulative_map[-1])
        seam.append((h - 1, j))
        for i in range(h - 2, -1, -1):
            if j == 0:
                idx = np.argmin(cumulative_map[i, j:j+2])
                j += idx
            elif j == w - 1:
                idx = np.argmin(cumulative_map[i, j-1:j+1])
                j += idx - 1
            else:
                idx = np.argmin(cumulative_map[i, j-1:j+2])
                j += idx - 1
            seam.append((i, j))
        seam.reverse()
    else:
        seam = []
        i = np.argmin(cumulative_map[:, -1])
        seam.append((i, w - 1))
        for j in range(w - 2, -1, -1):
            if i == 0:
                idx = np.argmin(cumulative_map[i:i+2, j])
                i += idx
            elif i == h - 1:
                idx = np.argmin(cumulative_map[i-1:i+1, j])
                i += idx - 1
            else:
                idx = np.argmin(cumulative_map[i-1:i+2, j])
                i += idx - 1
            seam.append((i, j))
        seam.reverse()
    return seam

def remove_seam(img_array, seam, axis):
    """
    Descripción breve:
    Elimina la costura especificada de la imagen.

    Argumentos:
    - img_array: Arreglo NumPy que representa la imagen.
    - seam: Lista de coordenadas que representan la costura a eliminar.
    - axis: Dirección de la eliminación de la costura (1 para vertical, 0 para horizontal).

    Returns:
    - img_array: Imagen resultante después de eliminar la costura.
    """
    h, w, _ = img_array.shape
    if axis == 1:
        mask = np.ones((h, w), dtype=bool)
        for i, j in seam:
            mask[i, j] = False
        img_array = img_array[mask].reshape((h, w - 1, 3))
    else:
        mask = np.ones((h, w), dtype=bool)
        for i, j in seam:
            mask[i, j] = False
        img_array = img_array[mask].reshape((h - 1, w, 3))
    return img_array

def rgb2gray(img_array):
    """
    Descripción breve:
    Convierte una imagen RGB a escala de grises.

    Argumentos:
    - img_array: Arreglo NumPy que representa la imagen RGB.

    Returns:
    - Imagen en escala de grises.
    """
    return np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

def show_images(images, original_shape):
    """
    Descripción breve:
    Muestra las imágenes intermedias del proceso de seam carving en una interfaz gráfica.

    Argumentos:
    - images: Lista de objetos PIL Image que representan las imágenes intermedias.
    - original_shape: Tupla con las dimensiones originales de la imagen.

    Returns:
    Ninguno
    """
    root = tk.Tk()
    root.title("Visualización de Seam Carving")

    # Convertir las imágenes a PhotoImage
    photo_images = [ImageTk.PhotoImage(img.resize((original_shape[1], original_shape[0]))) for img in images]

    label = tk.Label(root, image=photo_images[0])
    label.pack()

    slider = tk.Scale(root, from_=0, to=len(images)-1, orient=tk.HORIZONTAL, length=500,
                      label='Paso de eliminación de costura')
    slider.pack()

    def update_image(event):
        idx = slider.get()
        label.config(image=photo_images[idx])

    slider.bind("<Motion>", update_image)

    root.mainloop()

if __name__ == "__main__":
    main()
