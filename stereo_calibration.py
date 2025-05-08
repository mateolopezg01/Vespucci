import os
import cv2
import numpy as np

# Global variables preset
total_photos = 53

# Camera resolution
photo_width = 1280
photo_height = 960

# Image resolution for processing
img_width =  640#320
img_height = 960#240
image_size = (img_width,img_height)

# Chessboard parameters
rows = 6
columns = 9
square_size = 28

# Visualization options
drawCorners = False
showSingleCamUndistortionResults = False
showStereoRectificationResults = False
writeUdistortedImages = False
imageToDisp = './scenes960/scene_1280x960_1.png'

# Calibration settings
CHECKERBOARD = (rows,columns)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.CALIB_FIX_INTRINSIC

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float32)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpointsLeft = [] # 3d point in real world space
imgpointsLeft = [] # 2d points in image plane.

objpointsRight = [] # 3d point in real world space
imgpointsRight = [] # 2d points in image plane.

if (drawCorners):
    print("You can press 'Q' to quit this script.")


# Main processing cycle
# We process all calibration images and fill up 'imgpointsLeft' and 'objpointsRight'
# arrays with found coordinates of the chessboard
photo_counter = 0
print ('Main cycle start')

while photo_counter != total_photos:
  photo_counter = photo_counter + 1
  print ('Import pair No ' + str(photo_counter))
  leftName = './pairs960/left_'+str(photo_counter).zfill(2)+'.png'
  rightName = './pairs960/right_'+str(photo_counter).zfill(2)+'.png'
  leftExists = os.path.isfile(leftName)
  rightExists = os.path.isfile(rightName)
  
  # If pair has no left or right image - exit
  if ((leftExists == False) or (rightExists == False)) and (leftExists != rightExists):
      print ("Pair No ", photo_counter, "has only one image! Left:", leftExists, " Right:", rightExists )
      continue 
  
  # If stereopair is complete - go to processing 
  if (leftExists and rightExists):
      imgL = cv2.imread(leftName,1)
      loadedY, loadedX, clrs  =  imgL.shape
      grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
      gray_small_left = cv2.resize (grayL, (img_width,img_height), interpolation = cv2.INTER_AREA)
      imgR = cv2.imread(rightName,1)
      grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
      gray_small_right = cv2.resize (grayR, (img_width,img_height), interpolation = cv2.INTER_AREA)
      
      # Find the chessboard corners
      retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
      retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
      
      # Draw images with corners found
      if (drawCorners):
          cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
          cv2.imshow('Corners LEFT', imgL)
          cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
          cv2.imshow('Corners RIGHT', imgR)
          key = cv2.waitKey(0)
          if key == ord("q"):
              exit(0)

      # Here is a fix for the OpenCV bug, which is causing this error:
      # error:(-215:Assertion failed) fabs(norm_u1) > 0 in function 'InitExtrinsics'
      # It means corners are too close to the side of the image. Let's filter them out
      
      SayMore = False; #Should we print additional debug info?
      if ((retL == True) and (retR == True)):
          minRx = cornersR[:,:,0].min()
          maxRx = cornersR[:,:,0].max()
          minRy = cornersR[:,:,1].min()
          maxRy = cornersR[:,:,1].max()

          minLx = cornersL[:,:,0].min()
          maxLx = cornersL[:,:,0].max()          
          minLy = cornersL[:,:,1].min()
          maxLy = cornersL[:,:,1].max()          
               
          border_threshold_x = loadedX/100
          border_threshold_y = loadedY/100
          if (SayMore): 
            print ("thr_X: ", border_threshold_x, "thr_Y:", border_threshold_y)
          x_thresh_bad = False
          if ((minRx<border_threshold_x) or (minLx<border_threshold_x)): # or (loadedX-maxRx < border_threshold_x) or (loadedX-maxLx < border_threshold_x)):
              x_thresh_bad = True
          y_thresh_bad = False
          if ((minRy<border_threshold_y) or (minLy<border_threshold_y)): # or (loadedY-maxRy < border_threshold_y) or (loadedY-maxLy < border_threshold_y)):
              y_thresh_bad = True
          if (y_thresh_bad==True) or (x_thresh_bad==True):
              if (SayMore):
                  print("Chessboard too close to the side!", "X thresh: ", x_thresh_bad, "Y thresh: ", y_thresh_bad)
                  print ("minRx: ", minRx, "maxRx: ", maxRx, " minLx: ", minLx, "maxLx:", maxLx)      
                  print ("minRy: ", minRy, "maxRy: ", maxRy, " minLy: ", minLy, "maxLy:", maxLy) 
              else: 
                  print("Chessboard too close to the side! Image ignored")
              retL = False
              retR = False
              continue

      # Here is our scaling trick! Hi res for calibration, low res for real work!
      # Scale corners X and Y to our working resolution
      if ((retL == True) and (retR == True)) and (img_height <= photo_height):
          scale_ratio = img_height/photo_height
          print ("Scale ratio: ", scale_ratio)
          cornersL = cornersL*scale_ratio #cornersL/2.0
          cornersR = cornersR*scale_ratio #cornersR/2.0
      elif (img_height > photo_height):
          print ("Image resolution is higher than photo resolution, upscale needed. Please check your photo and image parameters!")
          exit (0)
      
      # Refine corners and add to array for processing
      if ((retL == True) and (retR == True)):
          objpointsLeft.append(objp)
          cv2.cornerSubPix(gray_small_left,cornersL,(3,3),(-1,-1),subpix_criteria)
          imgpointsLeft.append(cornersL)
          objpointsRight.append(objp)
          cv2.cornerSubPix(gray_small_right,cornersR,(3,3),(-1,-1),subpix_criteria)
          imgpointsRight.append(cornersR)
      else:
          print ("Pair No", photo_counter, "ignored, as no chessboard found" )
          continue
       
print ('End cycle')


# This function calibrates (undistort) a single camera
def calibrate_one_camera (objpoints, imgpoints, right_or_left):
  
    # Opencv sample code uses the var 'grey' from the last opened picture
    N_OK = len(objpoints)
    DIM  = (img_width, img_height)
    
    # Single camera calibration (undistortion)
    rms, camera_matrix, distortion_coeff, rvecs, tvecs = \
         cv2.calibrateCamera(
             objpoints,
             imgpoints,
             grayL.shape[::-1],
             None,
             None
         )
    # Refine Camera calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, DIM, 1, DIM)
    # Let's rectify our results
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, newcameramtx, DIM, 5)

    # Let's rectify our results
    # map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, None, DIM, 5)
    
    # Now we'll write our results to the file for the future use
    if (os.path.isdir('./calibration_data960/{}p'.format(img_height))==False):
        os.makedirs('./calibration_data960/{}p'.format(img_height))
#    np.savez('./calibration_data/{}p/camera_calibration_{}.npz'.format(img_height, right_or_left),
#        map1=map1, map2=map2, objpoints=objpoints, imgpoints=imgpoints,
#        camera_matrix=camera_matrix, distortion_coeff=distortion_coeff)
    np.savez('./calibration_data960/{}p/camera_calibration_{}.npz'.format(img_height, right_or_left),
        map1=map1, map2=map2, objpoints=objpoints, imgpoints=imgpoints,
        camera_matrix=camera_matrix, distortion_coeff=distortion_coeff, roi=roi )
    return (True)


# Stereoscopic calibration
def calibrate_stereo_cameras(res_x=img_width, res_y=img_height):
    # We need a lot of variables to calibrate the stereo camera
    """
    Based on code from:
    https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013
    """
    processing_time01 = cv2.getTickCount()
    objectPoints = None

    rightImagePoints = None
    rightCameraMatrix = None
    rightDistortionCoefficients = None

    leftImagePoints = None
    leftCameraMatrix = None
    leftDistortionCoefficients = None

    rotationMatrix = None
    translationVector = None

    imageSize= (res_x, res_y)

    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
    OPTIMIZE_ALPHA = 0.25

    try:
        npz_file = np.load('./calibration_data960/{}p/stereo_camera_calibration.npz'.format(res_y))
    except:
        pass

    for cam_num in [0, 1]:
        right_or_left = ["_right" if cam_num==1 else "_left"][0]

        try:
            print ('./calibration_data960/{}p/camera_calibration{}.npz'.format(res_y, right_or_left))
            npz_file = np.load('./calibration_data960/{}p/camera_calibration{}.npz'.format(res_y, right_or_left))

            list_of_vars = ['map1', 'map2', 'objpoints', 'imgpoints', 'camera_matrix', 'distortion_coeff', 'roi']
            print(sorted(npz_file.files))

            if sorted(list_of_vars) == sorted(npz_file.files):
                print("Camera calibration data has been found in cache.")
                map1 = npz_file['map1']
                map2 = npz_file['map2']
                objectPoints = npz_file['objpoints']
                if right_or_left == "_right":
                    rightImagePoints = npz_file['imgpoints']
                    rightCameraMatrix = npz_file['camera_matrix']
                    rightDistortionCoefficients = npz_file['distortion_coeff']
                if right_or_left == "_left":
                    leftImagePoints = npz_file['imgpoints']
                    leftCameraMatrix = npz_file['camera_matrix']
                    leftDistortionCoefficients = npz_file['distortion_coeff']
            else:
                print("Camera data file found but data corrupted.")
        except:
            #If the file doesn't exist
            print("Camera calibration data not found in cache.")
            return False


    print("Calibrating cameras together...")

    leftImagePoints = np.asarray(leftImagePoints, dtype=np.float32)
    rightImagePoints = np.asarray(rightImagePoints, dtype=np.float32)

    # Stereo calibration
    (RMS, newLeftCameraMatrix, newLeftDistortionCoefficients, newRightCameraMatrix, 
    newRightDistortionCoefficients, rotationMatrix, translationVector, 
    essentialMatrix, fundamentalMatrix) = cv2.stereoCalibrate(
            objectPoints, leftImagePoints, rightImagePoints,
            leftCameraMatrix, leftDistortionCoefficients,
            rightCameraMatrix, rightDistortionCoefficients,
            imageSize, None, None,
            cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_FIX_ASPECT_RATIO, TERMINATION_CRITERIA)
    # Print RMS result (for calibration quality estimation)
    print ("<><><><><><><><><><><><><><><><><><><><>")
    print ("<><>   RMS is ", RMS, " <><>")
    print ("<><><><><><><><><><><><><><><><><><><><>")    
    print("Rectifying cameras...")
    R1 = np.zeros([3,3])
    R2 = np.zeros([3,3])
    P1 = np.zeros([3,4])
    P2 = np.zeros([3,4])
    Q  = np.zeros([4,4])
    
    # Rectify calibration results
    (leftRectification, rightRectification, leftProjection, rightProjection,
            dispartityToDepthMap, leftRoi, rightRoi) = cv2.stereoRectify(
                    newLeftCameraMatrix, newLeftDistortionCoefficients,
                    newRightCameraMatrix, newRightDistortionCoefficients,
                    imageSize, rotationMatrix, translationVector,
                    R1, R2, P1, P2, Q,
                    cv2.CALIB_ZERO_DISPARITY, 0 )
    
    # Saving calibration results for the future use
    print("Saving calibration...")
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            newLeftCameraMatrix, newLeftDistortionCoefficients, leftRectification,
            leftProjection, imageSize, cv2.CV_16SC2)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            newRightCameraMatrix, newRightDistortionCoefficients, rightRectification,
            rightProjection, imageSize, cv2.CV_16SC2)


    np.savez_compressed('./calibration_data960/{}p/stereo_camera_calibration.npz'.format(res_y), imageSize=imageSize,
            leftMapX=leftMapX, leftMapY=leftMapY,
            rightMapX=rightMapX, rightMapY=rightMapY, dispartityToDepthMap = dispartityToDepthMap, 
            leftRoi = leftRoi, rightRoi = rightRoi,
            leftProjection = leftProjection, rightProjection = rightProjection, 
            leftCameraMatrix = newLeftCameraMatrix, leftDistortionCoeff = newLeftDistortionCoefficients,
            rightCameraMatrix = newRightCameraMatrix, rightDistortionCoeff = newRightDistortionCoefficients)
    return True

# Now we have all we need to do stereoscopic fisheyefisheye calibration
# Let's calibrate each camera, and than calibrate them together
print ("Left camera calibration...")
result = calibrate_one_camera(objpointsLeft, imgpointsLeft, 'left')
print ("Right camera calibration...")
result = calibrate_one_camera(objpointsRight, imgpointsRight, 'right')
print ("Stereoscopic calibration...")
result = calibrate_stereo_cameras()
print ("Calibration complete!")

# The following code just shows you calibration results
# 
#

if (showSingleCamUndistortionResults):

    """
    # Takes an image in as a numpy array and undistorts it
    """
    
    #h, w = imgL.shape[:2]
    w = img_width
    h = img_height
    print("Undistorting picture with (width, height):", (w, h))
    try:
        npz_file = np.load('./calibration_data960/{}p/camera_calibration{}.npz'.format(h, '_left'))
        if 'map1' and 'map2' in npz_file.files:
            #print("Camera calibration data has been found in cache.")
            map1 = npz_file['map1']
            map2 = npz_file['map2']
            roi  = npz_file['roi']
        else:
            print("Camera data file found but data corrupted.")
            exit(0)
    except:
        print("Camera calibration data not found in cache, file " & './calibration_data960/{}p/camera_calibration{}.npz'.format(h, left))
        exit(0)

    # We didn't load a new image from file, but use last image loaded while calibration
    undistorted_left = cv2.remap(gray_small_left, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, we, he = roi
    undistorted_left = undistorted_left[y:y+he, x:x+we]
    
    #h, w = imgR.shape[:2]
    print("Undistorting picture with (width, height):", (w, h))
    try:
        npz_file = np.load('./calibration_data960/{}p/camera_calibration{}.npz'.format(h, '_right'))
        if 'map1' and 'map2' in npz_file.files:
            #print("Camera calibration data has been found in cache.")
            map1 = npz_file['map1']
            map2 = npz_file['map2']
            roi  = npz_file['roi']
        else:
            print("Camera data file found but data corrupted.")
            exit(0)
    except:
        print("Camera calibration RIGHT data not found in cache.")
        exit(0)

    undistorted_right = cv2.remap(gray_small_right, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, we, he = roi
    undistorted_right = undistorted_right[y:y+he, x:x+we]

    cv2.imshow('Left UNDISTORTED', undistorted_left)
    cv2.imshow('Right UNDISTORTED', undistorted_right)
    cv2.waitKey(0)
    if (writeUdistortedImages):
        cv2.imwrite("undistorted_left.jpg",undistorted_left)
        cv2.imwrite("undistorted_right.jpg",undistorted_right)


if (showStereoRectificationResults):
    # lets rectify pair and look at the result
    
    try:
        npzfile = np.load('./calibration_data960/{}p/stereo_camera_calibration.npz'.format(img_height))
    except:
        print("Camera calibration data not found in cache, file " & './calibration_data960/{}p/stereo_camera_calibration.npz'.format(img_height))
        exit(0)
    
    leftMapX    = npzfile['leftMapX']
    leftMapY    = npzfile['leftMapY']
    leftRoi     = npzfile['leftRoi']
    rightMapX   = npzfile['rightMapX']
    rightMapY   = npzfile['rightMapY']
    rightRoi    = npzfile['rightRoi']

    #read image to undistort
    photo_width     = 1280
    photo_height    = 960
    image_width     = img_width
    image_height    = img_height
    image_size = (image_width,image_height)

    if os.path.isfile(imageToDisp) == False:
        print ('Can not read image from file \"'+imageToDisp+'\"')
        exit(0)

    pair_img = cv2.imread(imageToDisp,0)

    # If our image has width and height we need? 
    height_check, width_check  = pair_img.shape[:2]
    print(pair_img.shape[:2])
    
    if (width_check != photo_width+photo_width) and (height_check != photo_height):
        print('No same size')
        # It's not our size. If it is scaled?
        if (width_check/photo_width == height_check/photo_height):
            # Well, it's just scaled! Let's resize it to fit our needs
            pair_img = cv2.resize (pair_img, dsize=(photo_width, photo_height), interpolation = cv2.INTER_CUBIC)
        else:
            # Image can not be scaled, as calibration was done for another image size
            print ("Wrong image size. Please choose appropriate image.")
            exit (0)

    # Read image and split it in a stereo pair
    print('Read and split image...')
    imgLTest = pair_img [0:photo_height,0:photo_width] #Y+H and X+W
    imgRTest = pair_img [0:photo_height,photo_width:] #Y+H and X+W
    print('Image size = ', imgLTest.shape)
    
    # If pair has been loaded and splitted correclty?
    width_left, height_left     = imgLTest.shape[:2]
    width_right, height_right   = imgRTest.shape[:2]
    if 0 in [width_left, height_left, width_right, height_right]:
        print("Error: Can't remap image.")

    # Rectifying left and right images
    imgL = cv2.remap(imgLTest, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRTest, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print('Remaped image size = ',imgL.shape)
    x, y, we, he = leftRoi
    imgL = imgL[y:y+he, x:x+we]
    x, y, we, he = rightRoi
    imgR = imgR[y:y+he, x:x+we]

    print(leftRoi,rightRoi)

    cv2.imshow('Left  STEREO CALIBRATED', imgL)
    cv2.imshow('Right STEREO CALIBRATED', imgR)
    cv2.imwrite("rectified_left.jpg",imgL)
    cv2.imwrite("rectified_right.jpg",imgR)
    cv2.waitKey(0)
