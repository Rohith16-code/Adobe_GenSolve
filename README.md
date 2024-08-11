Adobe Gensolve:

Curvetopia
Welcome to Curvetopia! This project is dedicated to identifying, regularizing, and beautifying 2D curves. Our goal is to transform a set of polylines into well-defined curves with properties such as regularization, symmetry, and completeness.

Objective
The mission of Curvetopia is to take a set of polylines and output a set of curves that are well-defined and aesthetically pleasing. Instead of starting with raster images, this project simplifies the process by working with polylines, which are sequences of points representing curves.

Problem Description
In this project, a curve is defined as a connected sequence of points in 2D Euclidean space. We aim to process these polylines to achieve:

Regularization: Ensuring curves conform to expected shapes such as circles or rectangles.
Symmetry: Identifying and enhancing symmetrical properties of curves.
Completeness: Filling gaps in fragmented or occluded curves.
Simplified Approach
While the long-term goal is to handle PNG images and output cubic Bézier curves, this project begins by working directly with polylines. The input consists of paths defined by sequences of points, and the output will be sets of curves with specified properties, visualized in SVG format.

Features
Regularization: Automatically adjusts isolated curves to match predefined shapes.
Symmetry Detection: Finds and enhances symmetrical properties in curves.
Curve Completion: Fills in gaps in fragmented and occluded curves.
Visualization: Outputs curves in SVG format for easy rendering in a browser, with the option to convert them to PNG images.
Installation
To get started with Curvetopia, you'll need to have Python installed along with several dependencies. You can install them using pip:

pip install numpy matplotlib svgwrite cairosvg opencv-python scikit-image scikit-learn

Usage
Prepare Input: Ensure your input files are in CSV format with paths defined as sequences of points. You can also use PNG images for isolated curve processing.
Run the Script: Execute the script with the desired input type and file paths.

# For isolated curves
isolated_input = ['/path/to/isolated.csv']
process_curvetopia('isolated', isolated_input)

# For fragmented curves
fragmented_inputs = ['/path/to/frag0.csv', '/path/to/frag1.csv']
process_curvetopia('fragmented', fragmented_inputs)

# For connected occlusion
connected_occlusion_input = ['/path/to/occlusion1.csv']
process_curvetopia('connected_occlusion', connected_occlusion_input)

# For disconnected occlusion
disconnected_occlusion_input = ['/path/to/occlusion2.csv']
process_curvetopia('disconnected_occlusion', disconnected_occlusion_input)

# With image input (for isolated curves)
image_input = ['/path/to/image.png']
process_curvetopia('isolated', image_input)

Output
The output will consist of:

PNG Images: Rendered visualizations of the curves.
SVG Files: Scalable vector graphics files of the processed curves.
Contributing
We welcome contributions to improve Curvetopia! Please fork the repository and submit pull requests with your enhancements or bug fixes.

Acknowledgments
Thanks to the developers of the libraries used in this project: NumPy, Matplotlib, SVGWrite, CairoSVG, OpenCV, scikit-image, and scikit-learn.

