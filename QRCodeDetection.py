from matplotlib import pyplot
from matplotlib.patches import Rectangle
import math
import statistics
import pyzbar.pyzbar as pyzbar
# import cv2
import numpy as np
import matplotlib as plt
import imageIO.png


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b


# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r, g, b, w, h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()


# step 1
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    # STUDENT CODE HERE
    for i in range(image_height):
        for j in range(image_width):
            g = 0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j]
            # greyscale_pixel_array.append(round(g))
            greyscale_pixel_array[i][j] = round(g)

    return greyscale_pixel_array


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    temp_list = []
    for i in range(image_height):
        for j in range(image_width):
            temp_list.append(pixel_array[i][j])
    return min(temp_list), max(temp_list)


# quantize
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    grey_sacle = createInitializedGreyscalePixelArray(image_width, image_height)
    original_f = computeMinAndMaxValues(pixel_array, image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            if original_f[1] == original_f[0]:
                s_out = 0
            else:
                s_out = (pixel_array[i][j] - original_f[0]) * (255 - 0) / ((original_f[1] - original_f[0])) + 0
            grey_sacle[i][j] = round(s_out)

    return grey_sacle


# step 2
def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    list1 = []
    for i in range(image_height):
        list1.append([0] * image_width)

    for i in range(image_height):
        for j in range(image_width):
            if i == 0 or j == 0 or i == image_height - 1 or j == image_width - 1:
                continue
            else:
                list1[i][j] = (1 / 8) * (
                        1 * (pixel_array[i - 1][j - 1]) +
                        2 * (pixel_array[i - 1][j]) +
                        1 * (pixel_array[i - 1][j + 1]) +
                        (-1) * (pixel_array[i + 1][j - 1]) +
                        (-2) * (pixel_array[i + 1][j]) +
                        (-1) * (pixel_array[i + 1][j + 1]))

    return list1


# step 3
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    list1 = []
    for i in range(image_height):
        list1.append([0] * image_width)
    # list1 = [i for i in pixel_array]

    # for i in range(image_height):
    #     for j in range(image_width):
    #         if i==0 or j==0 or i==image_height-1 or j==image_width-1:
    #             list1[i][j] = 0

    for i in range(image_height):
        for j in range(image_width):
            if i == 0 or j == 0 or i == image_height - 1 or j == image_width - 1:
                continue
            else:
                list1[i][j] = (1 / 8) * (
                        (-1) * (pixel_array[i - 1][j - 1]) +
                        (-2) * (pixel_array[i][j - 1]) +
                        (-1) * (pixel_array[i + 1][j - 1]) +
                        (1) * (pixel_array[i - 1][j + 1]) +
                        (2) * (pixel_array[i][j + 1]) +
                        (1) * (pixel_array[i + 1][j + 1]))

    return list1


# step 4
def computeMagnitude(horizontal_list, vertical_list, image_width, image_height):
    a = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            data = math.sqrt(horizontal_list[i][j] * horizontal_list[i][j] + vertical_list[i][j] * vertical_list[i][j])
            a[i][j] = data

    return a


# step 5
def computeMean3x3RepeatBorder(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height - 1):
        for j in range(image_width - 1):
            if i == 0 and j == 0:
                value_of_4 = [pixel_array[i][j], pixel_array[i][j + 1],
                              pixel_array[i + 1][j], pixel_array[i + 1][j + 1]]
                mean_value = statistics.mean(value_of_4)
                result[i][j] = mean_value
            # elif i == image_height:
            #     value_of_6 = [pixel_array[i - 1][j - 1], pixel_array[i - 1][j], pixel_array[i - 1][j + 1],
            #                   pixel_array[i][j - 1], pixel_array[i][j], pixel_array[i][j + 1]]
            #     mean_value = statistics.mean(value_of_6)
            #     result[i][j] = mean_value
            # elif i == 0:
            #     value_of_6 = [pixel_array[i][j - 1], pixel_array[i][j], pixel_array[i][j + 1],
            #                   pixel_array[i + 1][j - 1], pixel_array[i + 1][j], pixel_array[i + 1][j + 1]]
            #     mean_value = statistics.mean(value_of_6)
            #     result[i][j] = mean_value
            # elif j == 0:
            #     value_of_6 = [pixel_array[i - 1][j], pixel_array[i - 1][j + 1],
            #                   pixel_array[i][j], pixel_array[i][j + 1],
            #                   pixel_array[i + 1][j], pixel_array[i + 1][j + 1]]
            #     mean_value = statistics.mean(value_of_6)
            #     result[i][j] = mean_value
            else:
                value_of_9 = [pixel_array[i - 1][j - 1], pixel_array[i - 1][j], pixel_array[i - 1][j + 1],
                              pixel_array[i][j - 1], pixel_array[i][j], pixel_array[i][j + 1],
                              pixel_array[i + 1][j - 1], pixel_array[i + 1][j], pixel_array[i + 1][j + 1]]
                mean_value = statistics.mean(value_of_9)
                result[i][j] = mean_value

    return result


# step 6
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    test = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] >= threshold_value:
                test[i][j] = 255
            else:
                test[i][j] = 0
    return test


# step 7
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    t2 = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] == 1 or pixel_array[i][j] == 255:
                if i != 0 and j != 0:
                    try:
                        t2[i - 1][j - 1] = 1
                    except:
                        pass
                    try:
                        t2[i - 1][j] = 1
                    except:
                        pass
                    try:
                        t2[i - 1][j + 1] = 1
                    except:
                        pass
                    try:
                        t2[i][j - 1] = 1
                    except:
                        pass
                    try:
                        t2[i][j] = 1
                    except:
                        pass
                    try:
                        t2[i][j + 1] = 1
                    except:
                        pass
                    try:
                        t2[i + 1][j - 1] = 1
                    except:
                        pass
                    try:
                        t2[i + 1][j] = 1
                    except:
                        pass
                    try:
                        t2[i + 1][j + 1] = 1
                    except:
                        pass
                elif i == 0 and j == 0:
                    t2[i][j] = 1
                    t2[i][j + 1] = 1
                    t2[i + 1][j] = 1
                    t2[i + 1][j + 1] = 1
    return t2


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    t = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height - 1):
        for j in range(image_width - 1):
            if pixel_array[i - 1][j - 1] == 1 and pixel_array[i - 1][j] == 1 and pixel_array[i - 1][j + 1] == 1 and \
                    pixel_array[i][j - 1] == 1 and pixel_array[i][j] == 1 and pixel_array[i][j + 1] == 1 and \
                    pixel_array[i + 1][j - 1] == 1 and pixel_array[i + 1][j] == 1 and pixel_array[i + 1][j + 1] == 1:
                t[i][j] = 1
            elif pixel_array[i - 1][j - 1] == 255 and pixel_array[i - 1][j] == 255 and pixel_array[i - 1][
                j + 1] == 255 and pixel_array[i][j - 1] == 255 and pixel_array[i][j] == 255 and pixel_array[i][
                j + 1] == 255 and pixel_array[i + 1][j - 1] == 255 and pixel_array[i + 1][j] == 255 and \
                    pixel_array[i + 1][j + 1] == 255:
                t[i][j] = 1
    return t


# step 8
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    q = Queue()
    current_label = 0

    result = createInitializedGreyscalePixelArray(image_width, image_height)
    ccsize = {}

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] != 0:
                count_pixel = 0
                q.enqueue((i, j))
                pixel_array[i][j] = 0
                current_label += 1

                while q.size() != 0:
                    y, x = q.dequeue()
                    count_pixel += 1
                    result[y][x] = current_label
                    if image_height > y - 1 > 0 and pixel_array[y - 1][x] != 0:  # upper
                        q.enqueue((y - 1, x))
                        pixel_array[y - 1][x] = 0
                    if 0 < y + 1 < image_height and pixel_array[y + 1][x] != 0:  # lower
                        q.enqueue((y + 1, x))
                        pixel_array[y + 1][x] = 0
                    if 0 < x - 1 < image_width and pixel_array[y][x - 1] != 0:  # left
                        q.enqueue((y, x - 1))
                        pixel_array[y][x - 1] = 0
                    if 0 < x + 1 < image_width and pixel_array[y][x + 1] != 0:  # right
                        q.enqueue((y, x + 1))
                        pixel_array[y][x + 1] = 0
                    ccsize[current_label] = count_pixel

    return result, ccsize


def main():
    # filename = "./images/covid19QRCode/poster1small.png"
    filename = "./images/covid19QRCode/challenging/bch.png"
    # filename = "./images/covid19QRCode/challenging/bloomfield.png"
    # filename = "./images/covid19QRCode/challenging/connecticut.png"
    # filename = "./images/covid19QRCode/challenging/playground.png"
    # filename = "./images/covid19QRCode/challenging/poster1smallrotated.png"
    # filename = "./images/covid19QRCode/challenging/shanghai.png"

    output_filename1 = "./images/output_image/task1.png"  # step 1 output
    output_filename2 = "./images/output_image/task2.png"  # step 2 output
    output_filename3 = "./images/output_image/task3.png"  # step 3 output
    output_filename4 = "./images/output_image/task4.png"  # step 4 output
    output_filename5 = "./images/output_image/task5.png"  # step 5 output
    output_filename6 = "./images/output_image/task6.png"  # step 6 output
    output_filename7 = "./images/output_image/task7.png"  # step 7 output
    output_filename8 = "./images/output_image/task8.png"  # step 8 output

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)

    pyplot.imshow(
        prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))

    # step 1
    grey_scale_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # step 2
    horizontal_list = computeHorizontalEdgesSobelAbsolute(grey_scale_array, image_width, image_height)
    new_horizontal_list = scaleTo0And255AndQuantize(horizontal_list, image_width, image_height)

    # step 3
    vertical_list = computeVerticalEdgesSobelAbsolute(grey_scale_array, image_width, image_height)
    new_vertical_list = scaleTo0And255AndQuantize(vertical_list, image_width, image_height)

    # step 4
    edge_magnitude = computeMagnitude(horizontal_list, vertical_list, image_width, image_height)
    new_edge_magnitude = scaleTo0And255AndQuantize(edge_magnitude, image_width, image_height)

    # step 5
    smooth_magnitude1 = computeMean3x3RepeatBorder(edge_magnitude, image_width, image_height)
    smooth_magnitude2 = computeMean3x3RepeatBorder(smooth_magnitude1, image_width, image_height)
    smooth_magnitude3 = computeMean3x3RepeatBorder(smooth_magnitude2, image_width, image_height)
    smooth_magnitude4 = computeMean3x3RepeatBorder(smooth_magnitude3, image_width, image_height)
    smooth_magnitude5 = computeMean3x3RepeatBorder(smooth_magnitude4, image_width, image_height)
    smooth_magnitude6 = computeMean3x3RepeatBorder(smooth_magnitude5, image_width, image_height)
    smooth_magnitude7 = computeMean3x3RepeatBorder(smooth_magnitude6, image_width, image_height)
    smooth_magnitude8 = computeMean3x3RepeatBorder(smooth_magnitude7, image_width, image_height)
    new_smooth_magnitude = scaleTo0And255AndQuantize(smooth_magnitude8, image_width, image_height)

    # step 6
    threshold = 70  # 100 for playground
    binary_image = computeThresholdGE(new_smooth_magnitude, threshold, image_width, image_height)
    new_binary_image = scaleTo0And255AndQuantize(binary_image, image_width, image_height)

    # step 7
    dilation1 = computeDilation8Nbh3x3FlatSE(new_binary_image, image_width, image_height)
    erosion1 = computeErosion8Nbh3x3FlatSE(dilation1, image_width, image_height)
    current_erosion = erosion1
    current_dilation = dilation1
    for i in range(2):
        temp_dilation = computeDilation8Nbh3x3FlatSE(current_dilation, image_width, image_height)
        temp_erosion = computeErosion8Nbh3x3FlatSE(current_erosion, image_width, image_height)
        current_dilation = temp_dilation
        current_erosion = temp_erosion

    # for i in range(2):
    #     pass

    new_fill_holes = scaleTo0And255AndQuantize(current_erosion, image_width, image_height)

    # step 8
    (ccimg, ccsizes) = computeConnectedComponentLabeling(new_fill_holes, image_width, image_height)
    # values = ccsizes.values()
    # values.sort()
    # max_value = values[len(values) - 1]
    # max_value = max(ccsizes.values())
    # max_key = [x for x, y in ccsizes.items() if y == max_value]
    max_key = max(ccsizes, key=ccsizes.get)
    for i in range(image_height):
        for j in range(image_width):
            if ccimg[i][j] == max_key:
                ccimg[i][j] = 255
            else:
                ccimg[i][j] = 0
    new_image_with_QR = scaleTo0And255AndQuantize(ccimg, image_width, image_height)

    # writeGreyscalePixelArraytoPNG(output_filename1, grey_scale_array, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename2, new_horizontal_list, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename3, new_vertical_list, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename4, new_edge_magnitude, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename5, new_smooth_magnitude, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename6, new_binary_image, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename7, new_fill_holes, image_width, image_height)
    # writeGreyscalePixelArraytoPNG(output_filename8, new_image_with_QR, image_width, image_height)

    # get access to the current pyplot figure
    axes = pyplot.gca()

    temp_min_i = image_height
    temp_min_j = image_width
    temp_max_i = 0
    temp_max_j = 0

    for i in range(image_height):
        for j in range(image_width):
            if new_image_with_QR[i][j] != 0:
                if temp_min_i > i:
                    temp_min_i = i
                if temp_min_j > j:
                    temp_min_j = j
                if i > temp_max_i:
                    temp_max_i = i
                if j > temp_max_j:
                    temp_max_j = j
    xy = (temp_min_j, temp_min_i)
    rectangle_width = temp_max_j - temp_min_j
    rectangle_height = temp_max_i - temp_min_i

    # create a 70x50 rectangle that starts at location 10,30, with a line width of 3
    # rect = Rectangle((10, 30), 70, 50, linewidth=3, edgecolor='g', facecolor='none')
    rect = Rectangle(xy, rectangle_width, rectangle_height, linewidth=3, edgecolor='g', facecolor='none')

    # polygon = Polygon([TL,TR,BR,BL], linewidth=3, edgecolor='g', facecolor='none')
    # paint the rectangle over the current plot
    axes.add_patch(rect)
    # axes.add_patch(polygon)

    # plot the current figure
    pyplot.show()

    code_list = []
    image_nd_array = plt.pyplot.imread(filename)
    image_nd_array = image_nd_array * 255
    # image = cv2.imread(filename)
    decode_result = pyzbar.decode(image_nd_array)
    for i in decode_result:
        i_data = i.data.decode("UTF-8")
        code_list.append(i_data)
        print("{}".format(i_data))
    # list_string = "".join(code_list)
    # print(list_string)


if __name__ == "__main__":
    main()
