import numpy as np
import colorsys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from collections import defaultdict
import os
import time
import plotly.io as pio

def galois_info():
# Start a session with the Wolfram Engine
    with WolframLanguageSession() as session:
        # Define the Wolfram Language expression for the Galois group properties
        poly = input('polynomial: ')

        # expression = f'ResourceFunction["GaloisGroupProperties"][{poly}, x, "GroupElements"]'
        # result = session.evaluate(wlexpr(expression))

        #cayleytable
        third_expression = f'ResourceFunction["GaloisGroupProperties"][{poly}, x, "MultiplicationTable"]'
        third_result = session.evaluate(wlexpr(third_expression))

        #polynomial degree
        # second_expression = f'ResourceFunction["PolynomialDegree"][{poly}, x]'
        # second_result = session.evaluate(wlexpr(second_expression))

        
    return third_result


def format_interactive_table(table_input, diagonal_number=25):
    # Check if input is a string or a Wolfram Language expression
    if isinstance(table_input, str):
        # If it's a string, use eval to convert it to a list
        data_list = eval(table_input.split('PackedArray(')[-1].split(']],')[0] + ']]')
    elif hasattr(table_input, 'args'):
        # If it's a Wolfram Language expression, extract the data
        data_list = table_input.args[0]
    else:
        raise ValueError("Input type not recognized")

    # Convert to numpy array
    data = np.array(data_list, dtype=np.int8)

    # Rearrange columns to put the specified number on the diagonal
    def rearrange_columns(arr, diag_num):
        n = arr.shape[0]
        new_order = np.zeros(n, dtype=int)
        for i in range(n):
            new_order[i] = np.where(arr[i] == diag_num)[0][0]
        return arr[:, new_order]

    data = rearrange_columns(data, diagonal_number)

    # Create column and row headers
    col_labels = [str(i) for i in range(1, data.shape[1] + 1)]
    row_labels = [str(i) for i in range(1, data.shape[0] + 1)]

    # Generate lighter pastel colors
    def generate_light_pastel_color(h):
        r, g, b = colorsys.hsv_to_rgb(h, 0.3, 0.99)  # Reduced saturation, increased value
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

    unique_values = np.unique(data)
    color_map = {diagonal_number: '#B3000E'}  # Keep the specific red color for the diagonal number
    other_values = [val for val in unique_values if val != diagonal_number]
    for i, val in enumerate(other_values):
        color_map[val] = generate_light_pastel_color(i / (len(other_values) or 1))

    # Create a 2D list of cell colors
    cell_colors = [[color_map[val] for val in row] for row in data]

    # Transpose the cell_colors to match the data structure expected by Plotly
    cell_colors_transposed = list(map(list, zip(*cell_colors)))

    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(values=[''] + col_labels,
                    fill_color='#40466e',
                    align='center',
                    font=dict(color='white', size=12)),
        cells=dict(values=[row_labels] + [data[:, i] for i in range(data.shape[1])],
                   fill_color=[['#40466e'] * len(row_labels)] + cell_colors_transposed,
                   align='center',
                   font=dict(color='black', size=12)))
    ])

    # Calculate dimensions to make the table more square
    cell_height = max(30, min(50, 600 / data.shape[0]))  # Adjust cell height based on number of rows
    table_height = cell_height * (data.shape[0] + 1)  # +1 for header
    table_width = table_height * (data.shape[1] + 1) / data.shape[0]  # +1 for row labels

    # Update layout for a more square appearance
    fig.update_layout(
        title=f'Interactive Light Pastel-Colored Data Table ({diagonal_number}\'s on Diagonal)',
        height=table_height,
        width=table_width,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Configure additional options for zooming and interactivity
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
    }
    
    # Show the figure
    fig.show(config=config)

def format_multiple_interactive_tables(table_input):
    # Check if input is a string or a Wolfram Language expression
    if isinstance(table_input, str):
        # If it's a string, use eval to convert it to a list
        data_list = eval(table_input.split('PackedArray(')[-1].split(']],')[0] + ']]')
    elif hasattr(table_input, 'args'):
        # If it's a Wolfram Language expression, extract the data
        data_list = table_input.args[0]
    else:
        raise ValueError("Input type not recognized")

    # Convert to numpy array
    data = np.array(data_list, dtype=np.int8)

    # Get unique values
    unique_values = np.unique(data)

    # Create color map
    color_map = {val: generate_light_pastel_color(i / len(unique_values)) for i, val in enumerate(unique_values)}

    # Configure additional options for zooming and interactivity
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
    }

    for diagonal_number in unique_values:
        rearranged_data = rearrange_columns(data, diagonal_number)
        
        # Create column and row headers
        col_labels = [str(i) for i in range(1, rearranged_data.shape[1] + 1)]
        row_labels = [str(i) for i in range(1, rearranged_data.shape[0] + 1)]

        # Create a 2D list of cell colors
        cell_colors = [[color_map[val] if val != diagonal_number else '#B3000E' for val in row] for row in rearranged_data]

        # Transpose the cell_colors to match the data structure expected by Plotly
        cell_colors_transposed = list(map(list, zip(*cell_colors)))

        # Calculate dimensions to make the table more square
        cell_height = max(30, min(50, 600 / rearranged_data.shape[0]))  # Adjust cell height based on number of rows
        table_height = cell_height * (rearranged_data.shape[0] + 1)  # +1 for header
        table_width = table_height * (rearranged_data.shape[1] + 1) / rearranged_data.shape[0]  # +1 for row labels

        # Create figure for this table
        fig = go.Figure(data=[go.Table(
            header=dict(values=[''] + col_labels,
                        fill_color='#40466e',
                        align='center',
                        font=dict(color='white', size=12),
                        height=cell_height),
            cells=dict(values=[row_labels] + [rearranged_data[:, i] for i in range(rearranged_data.shape[1])],
                       fill_color=[['#40466e'] * len(row_labels)] + cell_colors_transposed,
                       align='center',
                       font=dict(color='black', size=12),
                       height=cell_height)
        )])

        # Update layout for this figure
        fig.update_layout(
            title=f'Table with {diagonal_number}\'s on Diagonal',
            height=table_height,
            width=table_width,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        # Show the figure
        fig.show(config=config)

# Helper functions

def rearrange_columns(arr, diag_num):
    n = arr.shape[0]
    new_order = np.zeros(n, dtype=int)
    for i in range(n):
        new_order[i] = np.where(arr[i] == diag_num)[0][0]
    return arr[:, new_order]

def generate_light_pastel_color(h):
    r, g, b = colorsys.hsv_to_rgb(h, 0.3, 0.99)  # Reduced saturation, increased value
    return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

def format_col_table(table_input, diagonal_number=60):
    # Check if input is a string or a Wolfram Language expression
    if isinstance(table_input, str):
        data_list = eval(table_input.split('PackedArray(')[-1].split(']],')[0] + ']]')
    elif hasattr(table_input, 'args'):
        data_list = table_input.args[0]
    else:
        raise ValueError("Input type not recognized")

    # Convert to numpy array
    data = np.array(data_list, dtype=np.int8)

    # Rearrange columns to put the specified number on the diagonal
    def rearrange_columns(arr, diag_num):
        n = arr.shape[0]
        new_order = np.zeros(n, dtype=int)
        for i in range(n):
            new_order[i] = np.where(arr[i] == diag_num)[0][0]
        return arr[:, new_order]

    data = rearrange_columns(data, diagonal_number)

    # Create column and row headers
    col_labels = [str(i) for i in range(1, data.shape[1] + 1)]
    row_labels = [str(i) for i in range(1, data.shape[0] + 1)]

    # Generate pastel colors
    def generate_pastel_color(h):
        r, g, b = colorsys.hsv_to_rgb(h, 0.5, 0.95)
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

    # Create a color map for unique 24-cell blocks
    color_map = defaultdict(lambda: generate_pastel_color(np.random.random()))
    
    # Create a 2D list for cell colors
    cell_colors = []
    for col in range(data.shape[1]):
        column_colors = []
        for row in range(0, data.shape[0], 24):
            block = tuple(data[row:row+24, col])
            color = color_map[block]
            column_colors.extend([color] * 24)
        cell_colors.append(column_colors[:data.shape[0]])  # Trim to actual data size

    # Highlight the diagonal
    for i in range(min(data.shape)):
        cell_colors[i][i] = '#B3000E'  # Red color for diagonal

    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(values=[''] + col_labels,
                    fill_color='#40466e',
                    align='center',
                    font=dict(color='white', size=12)),
        cells=dict(values=[row_labels] + [data[:, i] for i in range(data.shape[1])],
                   fill_color=[['#40466e'] * len(row_labels)] + cell_colors,
                   align='center',
                   font=dict(color='black', size=12))
    )])

    # Calculate dimensions to make the table more square
    cell_height = max(30, min(50, 600 / data.shape[0]))  # Adjust cell height based on number of rows
    table_height = cell_height * (data.shape[0] + 1)  # +1 for header
    table_width = table_height * (data.shape[1] + 1) / data.shape[0]  # +1 for row labels

    # Update layout for a more square appearance
    fig.update_layout(
        title=f'Interactive 24-Cell Block-Colored Data Table ({diagonal_number}\'s on Diagonal)',
        height=table_height,
        width=table_width,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Configure additional options for zooming and interactivity
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
    }
    
    # Show the figure
    fig.show(config=config)

def save_multiple_interactive_tables(table_input, save_dir="table_images"):
    """
    Generate and save multiple interactive tables as PNG images.

    Args:
    table_input (str or WolframExpression): Input data for the tables.
    save_dir (str): Directory to save the PNG images. Defaults to "table_images".

    Returns:
    None
    """
    print("Starting table processing...")
    start_time = time.time()

    # Create directory for saving images
    os.makedirs(save_dir, exist_ok=True)
    print(f"Images will be saved in: {save_dir}")

    # Process input data
    try:
        if isinstance(table_input, str):
            data_list = eval(table_input.split('PackedArray(')[-1].split(']],')[0] + ']]')
        elif hasattr(table_input, 'args'):
            data_list = table_input.args[0]
        else:
            raise ValueError("Input type not recognized")
        
        data = np.array(data_list, dtype=np.int8)
    except Exception as e:
        print(f"Error processing input data: {e}")
        return

    # Get unique values
    unique_values = np.unique(data)
    total_tables = len(unique_values)
    print(f"Total number of tables to generate: {total_tables}")

    # Create color map
    color_map = {val: generate_light_pastel_color(i / len(unique_values)) 
                 for i, val in enumerate(unique_values)}

    # Configure options for zooming and interactivity
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
    }

    for index, diagonal_number in enumerate(unique_values, 1):
        table_start_time = time.time()
        print(f"\nProcessing table {index}/{total_tables} for diagonal number: {diagonal_number}")

        rearranged_data = rearrange_columns(data, diagonal_number)
        
        # Create column and row headers
        col_labels = [str(i) for i in range(1, rearranged_data.shape[1] + 1)]
        row_labels = [str(i) for i in range(1, rearranged_data.shape[0] + 1)]

        # Create a 2D list of cell colors
        cell_colors = [[color_map[val] if val != diagonal_number else '#B3000E' for val in row] 
                       for row in rearranged_data]

        # Transpose the cell_colors to match the data structure expected by Plotly
        cell_colors_transposed = list(map(list, zip(*cell_colors)))

        # Calculate dimensions to make the table more square
        cell_height = max(30, min(50, 600 / rearranged_data.shape[0]))
        table_height = cell_height * (rearranged_data.shape[0] + 1)  # +1 for header
        table_width = table_height * (rearranged_data.shape[1] + 1) / rearranged_data.shape[0]  # +1 for row labels

        # Create figure for this table
        fig = go.Figure(data=[go.Table(
            header=dict(values=[''] + col_labels,
                        fill_color='#40466e',
                        align='center',
                        font=dict(color='white', size=12),
                        height=cell_height),
            cells=dict(values=[row_labels] + [rearranged_data[:, i] for i in range(rearranged_data.shape[1])],
                       fill_color=[['#40466e'] * len(row_labels)] + cell_colors_transposed,
                       align='center',
                       font=dict(color='black', size=12),
                       height=cell_height)
        )])

        # Update layout for this figure
        fig.update_layout(
            title=f'Table with {diagonal_number}\'s on Diagonal',
            height=table_height,
            width=table_width,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        # Save the figure as PNG
        filename = os.path.join(save_dir, f'table_diagonal_{diagonal_number}.png')
        try:
            pio.write_image(fig, filename, scale=2)  # scale=2 for higher resolution
            print(f"Saved table image to: {filename}")
        except Exception as e:
            print(f"Error saving image: {e}")

        table_end_time = time.time()
        table_duration = table_end_time - table_start_time
        print(f"Table {index}/{total_tables} processed in {table_duration:.2f} seconds")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nAll tables processed and saved in {total_duration:.2f} seconds")



z = galois_info()
print(z.args[0]) 

#x^4 - 4*x^3 - 4*x^2 + 8*x - 2
#x^5 - 4*x^3 - 4*x^2 + 8*x - 2
#format_interactive_table(z)
#Has an error
format_multiple_interactive_tables(z)
#Works like a charm. Is actually more square so its better
#format_col_table(z)
#works but is hard to distinguish
#save_multiple_interactive_tables(z)
#doesnt work. Tried separetly with saving_images.py but still nothing. Future work -try with a jupyter notebook in anaconda

