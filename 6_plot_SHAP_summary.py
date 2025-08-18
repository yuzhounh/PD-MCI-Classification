import os
import matplotlib.pyplot as plt
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
import cairosvg
from PIL import Image
import io

# Set font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']

def merge_shap_svgs_to_pdf(base_dir=".", output_filename="6_combined_shap_summary"):
    """
    Merge 4 SHAP summary SVG files into a 2x2 grid PDF
    """
    
    # Define 4 model names in grid order
    models = ["LR", "SVM", "RF", "XGBoost"]
    
    # Build file paths
    svg_files = []
    for model in models:
        filename = f"5_{model}_shap_summary.svg"
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            svg_files.append(filepath)
        else:
            print(f"Warning: File not found - {filepath}")
    
    if len(svg_files) != 4:
        print(f"Error: Need 4 SVG files, but only found {len(svg_files)}")
        return
    
    # Method 1: Use reportlab
    try:
        merge_with_reportlab(svg_files, models, f"{output_filename}.pdf")
    except Exception as e:
        print(f"Reportlab method failed: {e}")
        
        # Method 2: Use matplotlib as backup
        try:
            merge_with_matplotlib(svg_files, models, f"{output_filename}_matplotlib.pdf")
        except Exception as e2:
            print(f"Matplotlib method also failed: {e2}")

def merge_with_reportlab(svg_files, models, output_pdf):
    """
    Use reportlab to merge SVG files into a 2x2 grid PDF
    """
    
    # Load all SVGs as reportlab drawing objects
    drawings = []
    for svg_file in svg_files:
        drawing = svg2rlg(svg_file)
        if drawing:
            drawings.append(drawing)
    
    if len(drawings) != 4:
        print(f"Unable to load all SVG files, only successfully loaded {len(drawings)}")
        return
    
    # Set parameters
    target_height = 300  # Uniform height
    horizontal_spacing = 20  # Horizontal spacing between subplots
    vertical_spacing = 20    # Vertical spacing between subplots
    rows, cols = 2, 2
    
    # Calculate scaling ratio and scaled dimensions for each subplot
    scaled_drawings = []
    scaled_widths = []
    
    for drawing in drawings:
        # Scale while maintaining aspect ratio
        scale_factor = target_height / drawing.height
        scaled_width = drawing.width * scale_factor
        
        scaled_drawing = Drawing(scaled_width, target_height)
        scaled_drawing.scale(scale_factor, scale_factor)
        scaled_drawing.add(drawing)
        
        scaled_drawings.append(scaled_drawing)
        scaled_widths.append(scaled_width)
    
    # Calculate maximum width for each column
    col_widths = [0] * cols
    for i, width in enumerate(scaled_widths):
        col = i % cols
        col_widths[col] = max(col_widths[col], width)
    
    # Calculate total page size
    total_width = sum(col_widths) + horizontal_spacing * (cols - 1)
    total_height = rows * target_height + vertical_spacing * (rows - 1)
    
    # Create PDF
    c = canvas.Canvas(output_pdf, pagesize=(total_width, total_height))
    
    # Calculate x position for each column
    col_x_positions = []
    current_x = 0
    for col in range(cols):
        col_x_positions.append(current_x)
        current_x += col_widths[col] + horizontal_spacing
    
    # Draw each SVG
    for i, drawing in enumerate(scaled_drawings):
        row = i // cols
        col = i % cols
        
        # Calculate x position (centered in column)
        col_center_x = col_x_positions[col] + col_widths[col] / 2
        drawing_x = col_center_x - drawing.width / 2
        
        # Calculate y position (from top to bottom)
        y_position = total_height - (row + 1) * target_height - row * vertical_spacing
        
        # Draw SVG
        renderPDF.draw(drawing, c, drawing_x, y_position)
        
        # Add label (top left corner)
        font_size = 18
        c.setFont("Helvetica-Bold", font_size)
        label_x = drawing_x + 10  # 10 units from left edge
        label_y = y_position + target_height - 20  # 20 units from top edge
        
        # Generate labels based on grid position
        # labels = ["(a) LR", "(b) SVM", "(c) RF", "(d) XGBoost"]
        labels = ["(a)", "(b)", "(c)", "(d)"]
        c.drawString(label_x, label_y, labels[i])
    
    c.save()
    print(f"Saved: {output_pdf}")
    print(f"Page size: {total_width:.1f} x {total_height:.1f}")

def merge_with_matplotlib(svg_files, models, output_pdf):
    """
    Use matplotlib to merge SVG files into a 2x2 grid PDF
    """
    
    # Convert SVG to images
    images = []
    image_ratios = []
    
    for svg_file in svg_files:
        try:
            # Convert SVG to PNG
            png_data = cairosvg.svg2png(url=svg_file, output_height=400)
            img = Image.open(io.BytesIO(png_data))
            images.append(img)
            image_ratios.append(img.width / img.height)
        except Exception as e:
            print(f"Warning: Unable to process {svg_file}: {e}")
            return
    
    if len(images) != 4:
        print(f"Unable to load all images, only successfully loaded {len(images)}")
        return
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SHAP Summary Plots', fontsize=16, fontweight='bold')
    
    # Define labels
    labels = ["(a) LR", "(b) SVM", "(c) RF", "(d) XGBoost"]
    
    # Display image in each subplot
    for i, (img, ax) in enumerate(zip(images, axes.flatten())):
        ax.imshow(img)
        ax.axis('off')
        
        # Add labels
        ax.text(0.02, 0.98, labels[i], transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save PDF
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    
    print(f"Saved: {output_pdf}")

def main():
    """Main function"""
    print("Starting to merge SHAP summary SVG files...")
    print("=" * 50)
    
    # Merge 4 SHAP summary SVG files
    merge_shap_svgs_to_pdf()
    
    print("=" * 50)
    print("Merge completed!")

if __name__ == "__main__":
    main()
