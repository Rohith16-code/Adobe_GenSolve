import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
import cv2
import os
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from scipy.optimize import minimize
from skimage import filters, measure, morphology, util
from skimage.feature import canny
from sklearn.cluster import DBSCAN
from IPython.display import display, Image, SVG

# Provided functions
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs, output_path=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    if output_path:
        plt.savefig(output_path)
        plt.close()
        display(Image(filename=output_path))
    else:
        plt.show()

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    W = max(W, 1)
    H = max(H, 1)

    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill='none', stroke=c, stroke_width=2))
    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                     parent_width=W, parent_height=H,
                     output_width=fact*W, output_height=fact*H,
                     background_color='white')

    display(SVG(filename=svg_path))
    display(Image(filename=png_path))


def is_closed(path):
    return np.allclose(path[0], path[-1])

def path_length(path):
    return np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))

def is_circle(path, tolerance=0.1):
    if not is_closed(path):
        return False
    center = np.mean(path, axis=0)
    radii = np.sqrt(np.sum((path - center)**2, axis=1))
    return np.std(radii) / np.mean(radii) < tolerance

def is_rectangle(path, tolerance=0.1):
    if not is_closed(path) or len(path) != 5:
        return False
    angles = []
    for i in range(4):
        v1 = path[i] - path[i-1]
        v2 = path[(i+1)%4] - path[i]
        angle = np.abs(np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2))))
        angles.append(angle)
    return np.all(np.abs(np.array(angles) - 90) < tolerance)

def find_symmetry_axis(path):
    center = np.mean(path, axis=0)
    angles = np.linspace(0, np.pi, 180)
    best_score = float('inf')
    best_angle = None
    for angle in angles:
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        rotated = np.dot(path - center, rot_matrix) + center
        score = np.sum(np.min(distance.cdist(path, rotated[::-1]), axis=1))
        if score < best_score:
            best_score = score
            best_angle = angle
    return best_angle

def image_to_polylines(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    edges = canny(img, sigma=2)

   
    edges = morphology.dilation(edges, morphology.disk(1))

    contours = measure.find_contours(edges, 0.8)

    polylines = []
    for contour in contours:
        contour = measure.approximate_polygon(contour, tolerance=2.0)
        polylines.append([contour])

    return polylines


def regularize_isolated(paths_XYs):
    regularized = []
    for path in paths_XYs:
        if is_circle(path[0]):
        
            center = np.mean(path[0], axis=0)
            radius = np.mean(np.sqrt(np.sum((path[0] - center)**2, axis=1)))
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)]) + center
            regularized.append([circle])
        elif is_rectangle(path[0]):
          
            corners = path[0][:4]
            regularized.append([np.vstack([corners, corners[0]])])
        else:
            regularized.append(path)
    return regularized

def regularize_fragmented(paths_XYs):
    return regularize_isolated(paths_XYs)

def complete_curve(start, end, points):
    X = points[:, 0]
    y = points[:, 1]
    poly = np.poly1d(np.polyfit(X, y, 2))
    new_X = np.linspace(start[0], end[0], 100)
    new_y = poly(new_X)
    new_points = np.column_stack([new_X, new_y])
    return new_points

def complete_connected_occlusion(paths_XYs):
    completed = []
    for path in paths_XYs:
        if is_closed(path[0]):
            completed.append(path)
        else:
            start = path[0][0]
            end = path[0][-1]
            points = path[0]
            completed_path = np.vstack([points, complete_curve(start, end, points)])
            completed.append([completed_path])
    return completed

def complete_disconnected_occlusion(paths_XYs):
    completed = []
    for i in range(0, len(paths_XYs), 2):
        if i+1 < len(paths_XYs):
            path1, path2 = paths_XYs[i][0], paths_XYs[i+1][0]
            start = path1[-1]
            end = path2[0]
            completed_path = np.vstack([path1, complete_curve(start, end, np.vstack([path1, path2])), path2])
            completed.append([completed_path])
        else:
            completed.append(paths_XYs[i])
    return completed

def process_curvetopia(input_type, input_paths):
    results = []

    if input_type == 'isolated':
        for path in input_paths:
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths_XYs = image_to_polylines(path)
            else:
                paths_XYs = read_csv(path)
            result = regularize_isolated(paths_XYs)
            results.append(result)

            output_base = os.path.splitext(path)[0]
            plot(result, f"{output_base}_output.png")
            polylines2svg(result, f"{output_base}_output.svg")

    elif input_type == 'fragmented':
        for path in input_paths:
            paths_XYs = read_csv(path)
            # print(f"Paths_XYs from {path}: {paths_XYs}")  # Debug print
            result = regularize_fragmented(paths_XYs)
            # print(f"Regularized result: {result}")  # Debug print
            results.append(result)

            output_base = os.path.splitext(path)[0]
            plot(result, f"{output_base}_output.png")
            polylines2svg(result, f"{output_base}_output.svg")

    elif input_type == 'connected_occlusion':
        for path in input_paths:
            paths_XYs = read_csv(path)
            result = complete_connected_occlusion(paths_XYs)
            results.append(result)

            output_base = os.path.splitext(path)[0]
            plot(result, f"{output_base}_output.png")
            polylines2svg(result, f"{output_base}_output.svg")

    elif input_type == 'disconnected_occlusion':
        for path in input_paths:
            paths_XYs = read_csv(path)
            result = complete_disconnected_occlusion(paths_XYs)
            results.append(result)

            output_base = os.path.splitext(path)[0]
            plot(result, f"{output_base}_output.png")
            polylines2svg(result, f"{output_base}_output.svg")

    else:
        raise ValueError("Invalid input type")

    return results

# Example usage
if __name__ == "__main__":
    isolated_input = ['/content/isolated.csv']
    process_curvetopia('isolated', isolated_input)

    fragmented_inputs = ['/content/frag0.csv', '/content/frag1.csv', '/content/frag2.csv']
    process_curvetopia('fragmented', fragmented_inputs)

    connected_occlusion_input = ['/content/occlusion1.csv']
    process_curvetopia('connected_occlusion', connected_occlusion_input)

    disconnected_occlusion_input = ['/content/occlusion2.csv']
    process_curvetopia('disconnected_occlusion', disconnected_occlusion_input)

    # Example with image input
    image_input = ['/content/image.png']
    process_curvetopia('isolated', image_input)
