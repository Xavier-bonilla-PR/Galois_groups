import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.language.expression import WLFunction
import re
import tkinter as tk
from PIL import Image, ImageTk
import io

def galois_info():
# Start a session with the Wolfram Engine
    with WolframLanguageSession() as session:
        # Define the Wolfram Language expression for the Galois group properties
        poly = input('polynomial: ')

        #GroupElements
        expression = f'ResourceFunction["GaloisGroupProperties"][{poly}, x, "GroupElements"]'
        result = session.evaluate(wlexpr(expression))

        #GroupOrder
        third_expression = f'ResourceFunction["GaloisGroupProperties"][{poly}, x, "GroupOrder"]'
        third_result = session.evaluate(wlexpr(third_expression))

        #polynomial degree
        second_expression = f'ResourceFunction["PolynomialDegree"][{poly}, x]'
        second_result = session.evaluate(wlexpr(second_expression))

        
    return result, second_result, third_result

def draw_circle_with_matrix(radius, markers):
    # Create a 2D matrix of the specified size
    size = radius * 2 + 1
    matrix = np.zeros((size, size), dtype=int)
    
    center = radius
    
    # Calculate the angles at which to place the markers
    angles = np.linspace(0, 2 * np.pi, len(markers), endpoint=False)
    
    marker_positions = {}
    for i, angle in enumerate(angles):
        x = int(center + radius * np.cos(angle))
        y = int(center + radius * np.sin(angle))
        matrix[y, x] = i + 2  # Using i+2 to represent 'x#'
        marker_positions[markers[i]] = (x, y)
    
    return matrix, marker_positions

def draw_arrow(ax, start, end, scale=0.4):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    ax.arrow(x1, y1, dx*scale, dy*scale, color='black', head_width=1.0, head_length=0.6)

def generate_distinct_colors(n):
    base_cmap = plt.colormaps['hsv']
    color_list = [base_cmap(i / n) for i in range(n)]
    # Ensure the colors are visually distinct
    return ['white'] + [rgb2hex(color_list[(i * 7) % n]) for i in range(n)]

def create_cycle_image(radius, markers, arrows, cycle_str):
    circle_matrix, marker_positions = draw_circle_with_matrix(radius, markers)

    colors = generate_distinct_colors(len(markers) + 1)
    cmap = ListedColormap(colors)

    # Increase figure size
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(circle_matrix, cmap=cmap, interpolation='nearest')

    for start, end in arrows:
        start_pos = marker_positions[start]
        end_pos = marker_positions[end]
        draw_arrow(ax, start_pos, end_pos)

    for marker, (x, y) in marker_positions.items():
        # Increase font size
        ax.text(x, y, marker, color='black', ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_title(f'Cycle: {cycle_str}', fontsize=18)
    
    # Save the plot to a bytes buffer with higher DPI
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def parse_cycle(cycle_str):
    def tupl_mkr(numbers):
        arrows = []
        for i in range(len(numbers)):
            start = f'x{numbers[i]}'
            end = f'x{numbers[(i+1) % len(numbers)]}'
            arrows.append((start, end))
        return arrows
    
    x = cycle_str.split('][')


    if len(x) == 1:
        numbers1 = re.findall(r'\d+', str(x))
        return tupl_mkr(numbers1)
        
    elif len(x) == 2:
        numbers0 = re.findall(r'\d+', str(x[0]))
        
        f = tupl_mkr(numbers0)
        
        numbers2 = re.findall(r'\d+', str(x[1]))
        t = tupl_mkr(numbers2)
        return f, t
    else:
        return str(cycle_str)


def marker(poly_degree):
    x = poly_degree
    y = []
    for i in range(x):
        a = 'x' + f'{i+1}'
        y.append(a)
        
    return y

def cycle_to_string(cycle):
    if isinstance(cycle, WLFunction):
        if cycle.head == wl.Cycles:
            return ''.join(cycle_to_string(c) for c in cycle.args)
        else:
            return str(cycle)
    elif isinstance(cycle, tuple):
        return ''.join(cycle_to_string(c) for c in cycle)
    else:
        return str(cycle)


class GaloisGroupApp:
    def __init__(self, root, galois_element, degree):
        self.root = root
        self.root.title("Galois Group Cycle Structure Visualization")
        
        # Create a canvas with scrollbars
        self.canvas = tk.Canvas(root)
        self.scrollbar_y = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = tk.Scrollbar(root, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.images = []
        self.photo_images = []
        self.labels = []

        self.galois_element = galois_element
        self.all_markers = marker(degree)
        self.create_base_image()
        self.create_images()
        self.display_images()

        self.frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_base_image(self):
        img_buf = create_cycle_image(10, self.all_markers, [], "Base Structure")
        img = Image.open(img_buf)
        img.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(img)
        self.images.insert(0, img)
        self.photo_images.insert(0, photo_img)

    def create_images(self):
        for cycle in self.galois_element:
            cycle_str = cycle_to_string(cycle)
            if cycle_str.strip('[]'):  # Skip empty cycles
                print(f"Processing cycle: {cycle_str}")
                cycle_arrows = parse_cycle(cycle_str)
                
                # Flatten cycle_arrows if it's a tuple
                if isinstance(cycle_arrows, tuple):
                    cycle_arrows = [arrow for sublist in cycle_arrows for arrow in sublist]
                
                img_buf = create_cycle_image(10, self.all_markers, cycle_arrows, cycle_str)
                img = Image.open(img_buf)
                img.thumbnail((400, 400))
                photo_img = ImageTk.PhotoImage(img)
                self.images.append(img)
                self.photo_images.append(photo_img)

    def display_images(self):
        for i, photo_img in enumerate(self.photo_images):
            row = i // 6  # Changed to 3 images per row
            col = i % 6
            label = tk.Label(self.frame, image=photo_img)
            label.grid(row=row, column=col, padx=10, pady=10)
            self.labels.append(label)


def main():
    galois_element, degree, group = galois_info()
    print(f'Degree: {degree}, Galois group size: {group}')
    
    root = tk.Tk()
    app = GaloisGroupApp(root, galois_element, degree)
    root.mainloop()

if __name__ == "__main__":
    main()

#x^4 - 4*x^3 - 4*x^2 + 8*x - 2

#Maximum galois group:
#x^2 - 2
#x^3 - 2
#x^4 + x + 1
#x^5 - x - 1

#If you want to adjust the size further, you can modify the figsize in create_cycle_image 
# and the thumbnail size in create_images. 
