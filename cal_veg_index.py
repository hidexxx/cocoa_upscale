import numpy as np

def NDVI(s0, s2res='10m', scale=10000):
    '''
    :param s0:array read from s2 20m 9 bands data .tif, or 10m 4 bands data.tif
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    calculate NDVI with equation: NDVI = (NIR-Red)/(NIR+red); NIR -- band 8/8a, red -- band 4
    :return: a 2d-array of NDVI
    '''
    red = s0[2, :, :]  # band 4
    if s2res == '20m':
        NIR = s0[6, :, :]  # band 8
    elif s2res == '10m':
        NIR = s0[3, :, :]  # band 8
    else:
        print('wrong input, NDVI not calculated')
        NIR = 0
    ndvi = (NIR - red) / (NIR + red)
    ndvi[np.isnan(ndvi)] = 0
    ndvi_int = ndvi * scale
    return ndvi_int

def CI(s0_20m, scale=10000):
    '''
    ci = Rededge3/Rededge1 -1
    :param s0: the 9 band from s2 20m resolution. Rededge is only in 20m resolution
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    Rededge 1, 2, 3 are band 5, 6, 7
    :return: a 2d-array of CI
    '''
    rededge3 = s0_20m[5, :, :]
    rededge1 = s0_20m[3, :, :]
    CI = (rededge3 / rededge1 - 1)
    CI[np.isnan(CI)] = 0
    CI_int = CI *  scale
    return CI_int

def PSRI(s0_10m, s0_20m, scale=10000):
    '''
    PSRI = (red-blue)/Rededge2
    :param s0_20m: the 9 band from s2 20m resolution. s0_10m: the 4 band from s2 10m
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    red: band 4, blue band 2, Rededge2 band6
    :return: a 2d-array of PSRI
    '''
    red = s0_10m[2, :, :]
    blue = s0_10m[0, :, :]
    rededge2 = s0_20m[4, :, :]
    psri = ((red - blue) / rededge2)
    psri[np.isnan(psri)] = 0
    psri_int = psri * scale
    return psri_int

def GNDVI(s0, s2res='10m', scale=10000):
    '''
    :param s0:array read from s2 20m 9 bands data .tif or 10m 4 bands data.tif
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    calculate GNDVI with equation: GNDVI = (NIR-Green)/(NIR+Green); NIR -- band 8/8a, green-- band 3
    :return: a 2d-array of GNDVI
    '''
    green = s0[1, :, :]  # band 3

    if s2res == '20m':
        NIR = s0[6, :, :]  # band 8
    elif s2res == '10m':
        NIR = s0[3, :, :]  # band 8
    else:
        print('wrong input, NDVI not calculated')
        NIR = 0
    gndvi = (NIR - green) / (NIR + green)
    gndvi[np.isnan(gndvi)] = 0
    gndvi_int = gndvi * scale
    return gndvi_int

def S2REP(s0_10m, s0_20m):
    '''

    :return:
    '''
    NIR = s0_10m[3, :, :]
    red = s0_10m[2, :, :]
    rededge1 = s0_20m[3, :, :]
    rededge2 = s0_20m[4, :, :]
    s2rep = 705 + 35 * ((((NIR + red) / 2) - rededge1) / (rededge2 - rededge1))
    s2rep[np.isnan(s2rep)] = 0
    s2rep[np.isinf(s2rep)] = 0
    return s2rep

def IRECI(s0_10m, s0_20m):
    '''
    :param s0_10m:
    :param s0_20m:
    :return:
    '''
    NIR = s0_10m[3, :, :]
    red = s0_10m[2, :, :]
    rededge1 = s0_20m[3, :, :]
    rededge2 = s0_20m[4, :, :]
    ireci = (NIR - red) / (rededge1 / rededge2)
    ireci[np.isnan(ireci)] = 0
    return ireci

def print_info(s0):
    print("------")
    print(str(np.max(s0)))
    print(str(np.min(s0)))
    print(str(np.mean(s0)))

def cal_vegIndex(s0_10m, s0_20m):
    ndvi = NDVI(s0=s0_10m)
    ci = CI(s0_20m=s0_20m)
    psri = PSRI(s0_10m=s0_10m, s0_20m=s0_20m)
    gndvi = GNDVI(s0=s0_10m)
    s2rep = S2REP(s0_10m=s0_10m, s0_20m=s0_20m)
    ireci = IRECI(s0_10m=s0_10m, s0_20m=s0_20m)
    print_info(ndvi)
    print_info(ci)
    print_info(psri)
    print_info(s2rep)
    print_info(ireci)
    result_array = np.dstack((ndvi, ci, psri, gndvi, s2rep, ireci)).astype(int)
    return result_array