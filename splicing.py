import numpy as np

def image_coor(image, w_num, h_num, size):
    '''
    Calculate image cutting position
    :param image: numpy.array, Image information matrix (W, H, C)
    :param w_num: int, Number of w cuts
    :param h_num: int, Number of h cuts
    :param size: int, Block size, square
    :return: tuple, Coordinate information
    '''
    (_,_,w,h) = image.shape
    all_w = w_num * size[0]
    all_h = h_num * size[1]

    difference_w = all_w - w
    difference_h = all_h - h

    overlap_w = int(difference_w / w_num)
    overlap_h = int(difference_h / h_num)

    start_list_w = []
    start_list_h = []
    end_list_w = []
    end_list_h = []

    for i in range(w_num):
        if i == (w_num - 1):
            start_list_w.append(w-size[0])
            end_list_w.append(w)
        else:
            start_list_w.append((size[0] - overlap_w) * i)
            end_list_w.append((size[0] - overlap_w) * i + size[0])
    for i in range(h_num):
        if i == (h_num - 1):
            start_list_h.append(h-size[1])
            end_list_h.append(h)
        else:
            start_list_h.append((size[1] - overlap_h) * i)
            end_list_h.append((size[1] - overlap_h) * i + size[1])
    return start_list_w,end_list_w,start_list_h,end_list_h

def cut_image(image,coor):
    '''
    Cut the image matrix and put it in the list
    :param image: numpy.array, Image information matrix (W, H, C)
    :param coor: tuple, Coordinate information (from function image_coor)
    :return: list, Image patch matrix list
    '''
    start_list_w = coor[0]
    end_list_w = coor[1]
    start_list_h = coor[2]
    end_list_h = coor[3]
    image_patch_list = []

    for i in range(len(start_list_w)):
        for j in range(len(start_list_h)):
            save_image = image[:,:,start_list_w[i]:end_list_w[i],start_list_h[j]:end_list_h[j]]
            image_patch_list.append(save_image)
    return image_patch_list

def splicing_image(image_patch_list,coor):
    '''
    Splicing the image patchs
    :param image_patch_list: Image patch list
    :param coor: tuple, Coordinate information (from function image_coor)
    :return: numpy.array, Splicing image
    '''
    start_list_w = coor[0]
    end_list_w = coor[1]
    start_list_h = coor[2]
    end_list_h = coor[3]

    in_start_list_w = []
    in_end_list_w = []
    in_start_list_h = []
    in_end_list_h = []
    out_start_list_w = []
    out_end_list_w = []
    out_start_list_h = []
    out_end_list_h = []

    for i in range(len(start_list_w)):
        if i == 0:
            in_start_list_w.append(start_list_w[i])
            in_start_list_h.append(start_list_h[i])

        else:
            num = int((start_list_w[i ] + end_list_w[i-1])/2)
            in_start_list_w.append(num)
            in_end_list_w.append(num)
            num = int((start_list_h[i ] + end_list_h[i-1]) / 2)
            in_start_list_h.append(num)
            in_end_list_h.append(num)
    in_end_list_w.append(end_list_w[i])
    in_end_list_h.append(end_list_h[i])

    for i in range(len(start_list_w)):
        if i == 0:
            out_start_list_w.append(start_list_w[0])
            out_start_list_h.append(start_list_h[0])

            out_end = (in_end_list_w[i]-in_start_list_w[i])
            out_end_list_w.append(out_end)
            out_end = (in_end_list_h[i] - in_start_list_h[i])
            out_end_list_h.append(out_end)
        else:
            num = int((start_list_w[i] + end_list_w[i-1])/2)
            out_start = num - start_list_w[i]
            out_end = out_start + (in_end_list_w[i]-in_start_list_w[i])
            out_start_list_w.append(out_start)
            out_end_list_w.append(out_end)
            num = int((start_list_h[i] + end_list_h[i-1]) / 2)
            out_start = num - start_list_h[i]
            out_end = out_start + (in_end_list_h[i] - in_start_list_h[i])
            out_start_list_h.append(out_start)
            out_end_list_h.append(out_end)

    sava_image = np.zeros((end_list_w[-1],end_list_h[-1],6))

    for i in range(len(start_list_w)):
        for j in range(len(start_list_h)):

            patch_num = i * len(start_list_w) + j

            sava_image[in_start_list_w[i]:in_end_list_w[i],
            in_start_list_h[j]:in_end_list_h[j],:] = \
                image_patch_list[patch_num][out_start_list_w[i]:out_end_list_w[i],
                out_start_list_h[j]:out_end_list_h[j],:]
    return sava_image