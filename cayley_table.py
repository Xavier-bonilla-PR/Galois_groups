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

        # Cayley table
        third_expression = f'ResourceFunction["GaloisGroupProperties"][{poly}, x, "MultiplicationTable"]'
        third_result = session.evaluate(wlexpr(third_expression))
        
    return third_result


def format_interactive_table(table_input, diagonal_number=25):
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
    data = rearrange_columns(data, diagonal_number)

    # Create column and row headers
    col_labels = [str(i) for i in range(1, data.shape[1] + 1)]
    row_labels = [str(i) for i in range(1, data.shape[0] + 1)]

    # Generate lighter pastel colors
    unique_values = np.unique(data)
    color_map = {diagonal_number: '#B3000E'}
    other_values = [val for val in unique_values if val != diagonal_number]
    for i, val in enumerate(other_values):
        color_map[val] = generate_light_pastel_color(i / (len(other_values) or 1))

    # Create a 2D list of cell colors
    cell_colors = [[color_map[val] for val in row] for row in data]
    cell_colors_transposed = list(map(list, zip(*cell_colors)))

    # IMPROVED SIZING LOGIC
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    
    # Calculate cell size and font size based on table size
    if n_rows <= 8:
        cell_height = 60
        font_size = 12
        header_font_size = 13
    elif n_rows <= 24:
        cell_height = 35
        font_size = 10
        header_font_size = 11
    elif n_rows <= 60:
        cell_height = 20
        font_size = 8
        header_font_size = 9
    else:
        cell_height = 15
        font_size = 7  # Smaller font for very large tables
        header_font_size = 8
    
    # Calculate total dimensions
    header_height = cell_height
    table_height = cell_height * (n_rows + 1) + 100  # +100 for margins and title
    table_width = max(800, cell_height * (n_cols + 1) * 1.2)  # Slightly wider for row labels
    
    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[''] + col_labels,
            fill_color='#40466e',
            align='center',
            font=dict(color='white', size=header_font_size),
            height=header_height
        ),
        cells=dict(
            values=[row_labels] + [data[:, i] for i in range(data.shape[1])],
            fill_color=[['#40466e'] * len(row_labels)] + cell_colors_transposed,
            align='center',
            font=dict(color='black', size=font_size),
            height=cell_height
        )
    )])

    # Update layout with fixed dimensions
    fig.update_layout(
        title=dict(
            text=f'Table with {diagonal_number}\'s on Diagonal',
            font=dict(size=16)
        ),
        height=table_height,
        width=table_width,
        margin=dict(l=10, r=10, t=60, b=10),
        autosize=False  # Prevent automatic resizing
    )

    # Configure additional options for zooming and interactivity
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'cayley_table_diag_{diagonal_number}',
            'height': table_height,
            'width': table_width,
            'scale': 2
        }
    }
    
    # Show the figure
    fig.show(config=config)
    
    return fig


def format_multiple_interactive_tables(table_input):
    # Check if input is a string or a Wolfram Language expression
    if isinstance(table_input, str):
        data_list = eval(table_input.split('PackedArray(')[-1].split(']],')[0] + ']]')
    elif hasattr(table_input, 'args'):
        data_list = table_input.args[0]
    else:
        raise ValueError("Input type not recognized")

    # Convert to numpy array
    data = np.array(data_list, dtype=np.int8)

    # Get unique values
    unique_values = np.unique(data)
    
    # IMPROVED SIZING LOGIC
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    
    # Calculate cell size and font size based on table size
    if n_rows <= 8:
        cell_height = 60
        font_size = 12
        header_font_size = 13
    elif n_rows <= 24:
        cell_height = 35
        font_size = 10
        header_font_size = 11
    elif n_rows <= 60:
        cell_height = 20
        font_size = 8
        header_font_size = 9
    else:
        cell_height = 15
        font_size = 7
        header_font_size = 8
    
    # Calculate total dimensions
    header_height = cell_height
    table_height = cell_height * (n_rows + 1) + 100
    table_width = max(800, cell_height * (n_cols + 1) * 1.2)

    # Create color map
    color_map = {val: generate_light_pastel_color(i / len(unique_values)) 
                 for i, val in enumerate(unique_values)}

    # Configure additional options for zooming and interactivity
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
    }

    figures = []
    
    for diagonal_number in unique_values:
        rearranged_data = rearrange_columns(data, diagonal_number)
        
        # Create column and row headers
        col_labels = [str(i) for i in range(1, rearranged_data.shape[1] + 1)]
        row_labels = [str(i) for i in range(1, rearranged_data.shape[0] + 1)]

        # Create a 2D list of cell colors
        cell_colors = [[color_map[val] if val != diagonal_number else '#B3000E' 
                       for val in row] for row in rearranged_data]
        cell_colors_transposed = list(map(list, zip(*cell_colors)))

        # Create figure for this table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[''] + col_labels,
                fill_color='#40466e',
                align='center',
                font=dict(color='white', size=header_font_size),
                height=header_height
            ),
            cells=dict(
                values=[row_labels] + [rearranged_data[:, i] for i in range(rearranged_data.shape[1])],
                fill_color=[['#40466e'] * len(row_labels)] + cell_colors_transposed,
                align='center',
                font=dict(color='black', size=font_size),
                height=cell_height
            )
        )])

        # Update layout for this figure
        fig.update_layout(
            title=dict(
                text=f'Table with {diagonal_number}\'s on Diagonal',
                font=dict(size=16)
            ),
            height=table_height,
            width=table_width,
            margin=dict(l=10, r=10, t=60, b=10),
            autosize=False
        )

        # Update config with proper screenshot dimensions
        fig_config = config.copy()
        fig_config['toImageButtonOptions'] = {
            'format': 'png',
            'filename': f'cayley_table_diag_{diagonal_number}',
            'height': table_height,
            'width': table_width,
            'scale': 2
        }

        # Show the figure
        fig.show(config=fig_config)
        figures.append(fig)
    
    return figures


# Helper functions

def rearrange_columns(arr, diag_num):
    n = arr.shape[0]
    new_order = np.zeros(n, dtype=int)
    for i in range(n):
        new_order[i] = np.where(arr[i] == diag_num)[0][0]
    return arr[:, new_order]

def generate_light_pastel_color(h):
    r, g, b = colorsys.hsv_to_rgb(h, 0.3, 0.99)
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
        cell_colors.append(column_colors[:data.shape[0]])

    # Highlight the diagonal
    for i in range(min(data.shape)):
        cell_colors[i][i] = '#B3000E'

    # IMPROVED SIZING LOGIC
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    
    if n_rows <= 8:
        cell_height = 60
        font_size = 12
        header_font_size = 13
    elif n_rows <= 24:
        cell_height = 35
        font_size = 10
        header_font_size = 11
    elif n_rows <= 60:
        cell_height = 20
        font_size = 8
        header_font_size = 9
    else:
        cell_height = 15
        font_size = 7
        header_font_size = 8
    
    header_height = cell_height
    table_height = cell_height * (n_rows + 1) + 100
    table_width = max(800, cell_height * (n_cols + 1) * 1.2)

    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[''] + col_labels,
            fill_color='#40466e',
            align='center',
            font=dict(color='white', size=header_font_size),
            height=header_height
        ),
        cells=dict(
            values=[row_labels] + [data[:, i] for i in range(data.shape[1])],
            fill_color=[['#40466e'] * len(row_labels)] + cell_colors,
            align='center',
            font=dict(color='black', size=font_size),
            height=cell_height
        )
    )])

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive 24-Cell Block-Colored Data Table ({diagonal_number}\'s on Diagonal)',
            font=dict(size=16)
        ),
        height=table_height,
        width=table_width,
        margin=dict(l=10, r=10, t=60, b=10),
        autosize=False
    )

    # Configure additional options
    config = {
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'cayley_table_blocks_diag_{diagonal_number}',
            'height': table_height,
            'width': table_width,
            'scale': 2
        }
    }
    
    fig.show(config=config)
    return fig

def save_multiple_interactive_tables(table_input, save_dir="table_images", save_format="html"):
    """
    Generate and save multiple interactive tables as HTML or PNG images.

    Args:
    table_input (str or WolframExpression): Input data for the tables.
    save_dir (str): Directory to save the files. Defaults to "table_images".
    save_format (str): Format to save ('html', 'png', or 'both'). Defaults to 'html'.

    Returns:
    None
    """
    print("Starting table processing...")
    start_time = time.time()

    # Create directory for saving images
    os.makedirs(save_dir, exist_ok=True)
    print(f"Files will be saved in: {os.path.abspath(save_dir)}")

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

    # IMPROVED SIZING LOGIC
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    
    if n_rows <= 8:
        cell_height = 60
        font_size = 12
        header_font_size = 13
    elif n_rows <= 24:
        cell_height = 35
        font_size = 10
        header_font_size = 11
    elif n_rows <= 60:
        cell_height = 20
        font_size = 8
        header_font_size = 9
    else:
        cell_height = 15
        font_size = 7
        header_font_size = 8
    
    header_height = cell_height
    table_height = cell_height * (n_rows + 1) + 100
    table_width = max(800, cell_height * (n_cols + 1) * 1.2)

    # Create color map
    color_map = {val: generate_light_pastel_color(i / len(unique_values)) 
                 for i, val in enumerate(unique_values)}

    # If PNG format is requested, check for kaleido
    if save_format in ['png', 'both']:
        try:
            import kaleido
            print("Kaleido found - PNG export enabled")
        except ImportError:
            print("\nWARNING: kaleido not installed. PNG export will fail.")
            print("Install with: pip install kaleido --break-system-packages")
            if save_format == 'png':
                print("Switching to HTML format instead.")
                save_format = 'html'
            else:
                print("Will only save HTML files.")
                save_format = 'html'

    saved_files = []

    for index, diagonal_number in enumerate(unique_values, 1):
        table_start_time = time.time()
        print(f"\nProcessing table {index}/{total_tables} for diagonal number: {diagonal_number}")

        rearranged_data = rearrange_columns(data, diagonal_number)
        
        # Create column and row headers
        col_labels = [str(i) for i in range(1, rearranged_data.shape[1] + 1)]
        row_labels = [str(i) for i in range(1, rearranged_data.shape[0] + 1)]

        # Create a 2D list of cell colors
        cell_colors = [[color_map[val] if val != diagonal_number else '#B3000E' 
                       for val in row] for row in rearranged_data]
        cell_colors_transposed = list(map(list, zip(*cell_colors)))

        # Create figure for this table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[''] + col_labels,
                fill_color='#40466e',
                align='center',
                font=dict(color='white', size=header_font_size),
                height=header_height
            ),
            cells=dict(
                values=[row_labels] + [rearranged_data[:, i] for i in range(rearranged_data.shape[1])],
                fill_color=[['#40466e'] * len(row_labels)] + cell_colors_transposed,
                align='center',
                font=dict(color='black', size=font_size),
                height=cell_height
            )
        )])

        # Update layout for this figure
        fig.update_layout(
            title=dict(
                text=f'Table with {diagonal_number}\'s on Diagonal',
                font=dict(size=16)
            ),
            height=table_height,
            width=table_width,
            margin=dict(l=10, r=10, t=60, b=10),
            autosize=False
        )

        # Save as HTML (always works)
        if save_format in ['html', 'both']:
            html_filename = os.path.join(save_dir, f'table_diagonal_{diagonal_number}.html')
            try:
                fig.write_html(html_filename)
                print(f"✓ Saved HTML to: {html_filename}")
                saved_files.append(html_filename)
            except Exception as e:
                print(f"✗ Error saving HTML: {e}")

        # Save as PNG (requires kaleido)
        if save_format in ['png', 'both']:
            png_filename = os.path.join(save_dir, f'table_diagonal_{diagonal_number}.png')
            try:
                pio.write_image(fig, png_filename, width=table_width, height=table_height, scale=2)
                print(f"✓ Saved PNG to: {png_filename}")
                saved_files.append(png_filename)
            except Exception as e:
                print(f"✗ Error saving PNG: {e}")
                print(f"   Hint: Install kaleido with: pip install kaleido --break-system-packages")

        table_end_time = time.time()
        table_duration = table_end_time - table_start_time
        print(f"Table {index}/{total_tables} processed in {table_duration:.2f} seconds")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"All tables processed in {total_duration:.2f} seconds")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Location: {os.path.abspath(save_dir)}")
    print(f"{'='*60}")

    # Create an index HTML file
    if save_format in ['html', 'both']:
        create_index_html(save_dir, unique_values)


def create_index_html(save_dir, diagonal_numbers):
    """Create an index.html file that links to all generated tables."""
    index_path = os.path.join(save_dir, 'index.html')
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Galois Group Cayley Tables</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #40466e;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .card a {
            color: #40466e;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
        }
        .card a:hover {
            color: #B3000E;
        }
    </style>
</head>
<body>
    <h1>Galois Group Cayley Tables</h1>
    <p style="text-align: center; color: #666;">
        Click on any table to view the interactive visualization
    </p>
    <div class="grid">
"""
    
    for diag_num in diagonal_numbers:
        html_content += f"""
        <div class="card">
            <a href="table_diagonal_{diag_num}.html">
                Diagonal: {diag_num}
            </a>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ Created index file: {index_path}")
    print(f"  Open this file in your browser to access all tables")


# Example usage
if __name__ == "__main__":
    z = galois_info()
    print(z.args[0])
    
    # Uncomment the function you want to use:
    # format_interactive_table(z)
    #format_multiple_interactive_tables(z)
    # format_col_table(z)
    
    # Save tables - now with working HTML export and optional PNG
    # save_multiple_interactive_tables(z, save_format='html')
    save_multiple_interactive_tables(z, save_format='png')
    # save_multiple_interactive_tables(z, save_format='both')
    #x^4 - 4*x^3 - 4*x^2 + 8*x - 2
    #x^5 - 4*x^3 - 4*x^2 + 8*x - 2
