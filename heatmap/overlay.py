import matplotlib.pyplot as plt
import numpy
import pandas
from scipy import ndimage
from skimage import io, transform
import dframe
from skimage.measure import block_reduce
import seaborn
numpy.set_printoptions(threshold=numpy.inf, linewidth=200)


def save_overlay(image, matrix, path=None, alpha=0.7, cmap='viridis'):
    print('shapes', matrix.shape, image.shape)
    height = image.shape[0]
    width = image.shape[1]
    matrix_resized = transform.resize(matrix, (height, width))
    # max_value = numpy.max(matrix_resized)
    # min_value = numpy.min(matrix_resized)
    # normalized_matrix = (matrix_resized - min_value) / (max_value - min_value)
    normalized_matrix = matrix_resized
    plt.imshow(image)
    plt.imshow(255 * normalized_matrix, alpha=alpha, cmap=cmap)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def not_empty(row):
    return pandas.notna(row['gaze_x']) and pandas.notna(row['gaze_y'])


def is_bounded(row, shape):
    return 0 <= row['gaze_x'] <= shape[0] and  0 <= row['gaze_y'] <= shape[1]


def is_valid(row, shape):
    return not_empty(row) and is_bounded(row, shape)


def fill_(screen, frames):
    for frame in frames:
        for _, row in frame.iterrows():
            if is_valid(row, screen.shape):
                i, j = int(row['gaze_x']), int(row['gaze_y'])
                screen[i][j] += 1
    return screen


def pool(grid, kernel_shape=(2, 2), func=numpy.sum):
    result = block_reduce(grid, kernel_shape, func)
    return numpy.flip(result, 0)

def normalize(grid):
    upper = numpy.max(grid)
    lower = numpy.min(grid)
    return numpy.round((grid - lower) / (upper - lower), 3)

if __name__ == '__main__':
    #frames = dframe.load_one('input/20200508110227', ['data_*.csv'])
    #matrix = numpy.zeros((1080, 1920), dtype=int)
    #fill_(matrix, frames)
    #pooled = pool(matrix, (192, 108))
    #numpy.save('pooled', pooled)
    pooled = numpy.load('pooled.npy')
    smoothed = ndimage.filters.gaussian_filter(pooled, sigma=2)
    #loged = numpy.nan_to_num(numpy.log(pooled), 0)
    normed = normalize(smoothed)
    print(normed)
    #print(smoothed)
    fname = 'screen.jpg'
    image = io.imread(fname)
    save_overlay(image, normed, path='heatmap.png')
