/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenet

import android.Manifest
import android.annotation.TargetApi
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.support.v4.app.ActivityCompat
import android.support.v4.app.DialogFragment
import android.support.v4.app.Fragment
import android.support.v4.content.ContextCompat
import android.util.Log
import android.util.Log.d
import android.util.Size
import android.util.SparseIntArray
import android.view.*
import android.widget.Button
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_posenet.*
import kotlinx.android.synthetic.main.activity_posenet.view.*
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.examples.posenet.lib.Posenet
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.*
import kotlin.concurrent.timer
import kotlin.math.absoluteValue
import kotlin.math.pow
import android.media.MediaPlayer

class PosenetActivity :
  Fragment(),
  ActivityCompat.OnRequestPermissionsResultCallback {

  /** List of body joints that should be connected.    */
  private val bodyJoints = listOf(
    Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
    Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
  )

  private var copy = Person()
  /** Threshold for confidence score. */
  private val minConfidence = 0.5

  /** Radius of circle used to draw keypoints.  */
  private val circleRadius = 8.0f
  private var mybool = false
  /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
  private var paint = Paint()

  /**하단 글자를 위한 페인트**/
  private  var charPaint = Paint()

  var mp3bool = true

  /**현재시간 변수 선언**/
  var starttime = 0
  var endtime = 0

  /**평균 위한 점수 총합**/
  var totalscore = 0.0

  /**평균 위한 횟수 총합**/
  var count = 0

  /** A shape for extracting frame data.   */
  private val PREVIEW_WIDTH = 640
  private val PREVIEW_HEIGHT = 480

  /** An object for the Posenet library.    */
  private lateinit var posenet: Posenet

  /** ID of the current [CameraDevice].   */
  private var cameraId: String? = null

  /** A [SurfaceView] for camera preview.   */
  private var surfaceView: SurfaceView? = null

  /** A [CameraCaptureSession] for camera preview.   */
  private var captureSession: CameraCaptureSession? = null

  private var person = Person()

  /** A reference to the opened [CameraDevice].    */
  private var cameraDevice: CameraDevice? = null

  /** The [android.util.Size] of camera preview.  */
  private var previewSize: Size? = null

  /** The [android.util.Size.getWidth] of camera preview. */
  private var previewWidth = 0

  /** The [android.util.Size.getHeight] of camera preview.  */
  private var previewHeight = 0

  /** A counter to keep count of total frames.  */
  private var frameCounter = 0

  /** mediaplayer**/
  var mediaplayer = MediaPlayer()

  /** An IntArray to save image data in ARGB8888 format  */
  private lateinit var rgbBytes: IntArray

  /** A ByteArray to save image data in YUV format  */
  private var yuvBytes = arrayOfNulls<ByteArray>(3)

  /** An additional thread for running tasks that shouldn't block the UI.   */
  private var backgroundThread: HandlerThread? = null

  /** A [Handler] for running tasks in the background.    */
  private var backgroundHandler: Handler? = null

  /** An [ImageReader] that handles preview frame capture.   */
  private var imageReader: ImageReader? = null

  /** [CaptureRequest.Builder] for the camera preview   */
  private var previewRequestBuilder: CaptureRequest.Builder? = null

  /** [CaptureRequest] generated by [.previewRequestBuilder   */
  private var previewRequest: CaptureRequest? = null

  /** A [Semaphore] to prevent the app from exiting before closing the camera.    */
  private val cameraOpenCloseLock = Semaphore(1)

  /** Whether the current camera device supports Flash or not.    */
  private var flashSupported = false


  /** Orientation of the camera sensor.   */
  private var sensorOrientation: Int? = null



  /** Abstract interface to someone holding a display surface.    */
  private var surfaceHolder: SurfaceHolder? = null

  /** [CameraDevice.StateCallback] is called when [CameraDevice] changes its state.   */
  private val stateCallback = object : CameraDevice.StateCallback() {

    override fun onOpened(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      this@PosenetActivity.cameraDevice = cameraDevice
      createCameraPreviewSession()
    }

    override fun onDisconnected(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      cameraDevice.close()
      this@PosenetActivity.cameraDevice = null
    }

    override fun onError(cameraDevice: CameraDevice, error: Int) {
      onDisconnected(cameraDevice)
      this@PosenetActivity.activity?.finish()
    }
  }

  /**
   * A [CameraCaptureSession.CaptureCallback] that handles events related to JPEG capture.
   */
  private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
    override fun onCaptureProgressed(
      session: CameraCaptureSession,
      request: CaptureRequest,
      partialResult: CaptureResult
    ) {
    }

    override fun onCaptureCompleted(
      session: CameraCaptureSession,
      request: CaptureRequest,
      result: TotalCaptureResult
    ) {
    }
  }

  /**
   * Shows a [Toast] on the UI thread.
   *
   * @param text The message to show
   */
  private fun showToast(text: String) {
    val activity = activity
    activity?.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_SHORT).show() }
  }


  override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                            savedInstanceState: Bundle?): View? {

    val view: View = inflater!!.inflate(R.layout.activity_posenet, container, false)


    /** mybool이 false일때 true, true일때 false로 바뀌면서 스위치 동작되게 함**/

    view.save.setOnClickListener { view ->

      copy = person

      mybool = !mybool
      if(mybool){
        //스위치가 on으로 바뀌었다면
        val tz = TimeZone.getTimeZone("Asia/Seoul")
        val gc = GregorianCalendar(tz)
        var hour= gc.get(GregorianCalendar.HOUR).toInt()
        var min = gc.get(GregorianCalendar.MINUTE).toInt()
        var sec = gc.get(GregorianCalendar.SECOND).toInt()
        starttime = hour*3600 + min*60 + sec
      }
      else{
        //스우치가 off로 바뀌었다면
        val tz = TimeZone.getTimeZone("Asia/Seoul")
        val gc = GregorianCalendar(tz)
        var hour= gc.get(GregorianCalendar.HOUR).toInt()
        var min = gc.get(GregorianCalendar.MINUTE).toInt()
        var sec = gc.get(GregorianCalendar.SECOND).toInt()
        endtime = hour*3600 + min*60 + sec
        val totaltime = endtime-starttime

        //Alert 띄우기!
        val builder = AlertDialog.Builder(ContextThemeWrapper(this.context, R.style.Theme_AppCompat_Light_Dialog))
        builder.setTitle("수고하셨습니다!")
        builder.setMessage((totaltime/3600).toString() + "시간" + ((totaltime%3600)/60).toString() + "분" + (totaltime%60).toString() + "초 동안 당신의 평균 자세 점수는\n"+ "%.2f".format(totalscore.div(count))+"점 입니다!") //평균

        builder.setPositiveButton("확인") { _, _ ->
          Log.d("alert ok","2")
        }
        builder.setNegativeButton("취소") { _, _ ->
          Log.d("alert cancel","3")
        }

        builder.show()

        //전역변수들 초기화
        totalscore=0.0
        count=0
        starttime=0
        endtime=0
      }
    }
    // Return the fragment view/layout
    return view
  }


  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    surfaceView = view.findViewById(R.id.surfaceView)
    surfaceHolder = surfaceView!!.holder
  }

  override fun onResume() {
    super.onResume()
    startBackgroundThread()
  }

  override fun onStart() {
    super.onStart()
    openCamera()
    posenet = Posenet(this.context!!)
  }

  override fun onPause() {
    closeCamera()
    stopBackgroundThread()
    super.onPause()
  }

  override fun onDestroy() {
    super.onDestroy()
    posenet.close()
  }

  private fun requestCameraPermission() {
    if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
      ConfirmationDialog().show(childFragmentManager, FRAGMENT_DIALOG)
    } else {
      requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ) {
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (allPermissionsGranted(grantResults)) {
        ErrorDialog.newInstance(getString(R.string.request_permission))
          .show(childFragmentManager, FRAGMENT_DIALOG)
      }
    } else {
      super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
  }

  private fun allPermissionsGranted(grantResults: IntArray) = grantResults.all {
    it == PackageManager.PERMISSION_GRANTED
  }

  /**
   * Sets up member variables related to camera.
   */
  private fun setUpCameraOutputs() {

    val activity = activity
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      for (cameraId in manager.cameraIdList) {
        val characteristics = manager.getCameraCharacteristics(cameraId)

        // We don't use a front facing camera in this sample.
        val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
        if (cameraDirection != null &&
          cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
        ) {
          continue
        }

        previewSize = Size(PREVIEW_WIDTH, PREVIEW_HEIGHT)

        imageReader = ImageReader.newInstance(
          PREVIEW_WIDTH, PREVIEW_HEIGHT,
          ImageFormat.YUV_420_888, /*maxImages*/ 2
        )

        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

        previewHeight = previewSize!!.height
        previewWidth = previewSize!!.width

        // Initialize the storage bitmaps once when the resolution is known.
        rgbBytes = IntArray(previewWidth * previewHeight)

        // Check if the flash is supported.
        flashSupported =
          characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true

        this.cameraId = cameraId

        // We've found a viable camera and finished setting up member variables,
        // so we don't need to iterate through other available cameras.
        return
      }
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: NullPointerException) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      ErrorDialog.newInstance(getString(R.string.camera_error))
        .show(childFragmentManager, FRAGMENT_DIALOG)
    }
  }

  /**
   * Opens the camera specified by [PosenetActivity.cameraId].
   */
  private fun openCamera() {
    val permissionCamera = ContextCompat.checkSelfPermission(activity!!, Manifest.permission.CAMERA)
    if (permissionCamera != PackageManager.PERMISSION_GRANTED) {
      requestCameraPermission()
    }
    setUpCameraOutputs()
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      // Wait for camera to open - 2.5 seconds is sufficient
      if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw RuntimeException("Time out waiting to lock camera opening.")
      }
      manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera opening.", e)
    }
  }

  /**
   * Closes the current [CameraDevice].
   */
  private fun closeCamera() {
    if (captureSession == null) {
      return
    }

    try {
      cameraOpenCloseLock.acquire()
      captureSession!!.close()
      captureSession = null
      cameraDevice!!.close()
      cameraDevice = null
      imageReader!!.close()
      imageReader = null
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera closing.", e)
    } finally {
      cameraOpenCloseLock.release()
    }
  }

  /**
   * Starts a background thread and its [Handler].
   */
  private fun startBackgroundThread() {
    backgroundThread = HandlerThread("imageAvailableListener").also { it.start() }
    backgroundHandler = Handler(backgroundThread!!.looper)
  }

  /**
   * Stops the background thread and its [Handler].
   */
  private fun stopBackgroundThread() {
    backgroundThread?.quitSafely()
    try {
      backgroundThread?.join()
      backgroundThread = null
      backgroundHandler = null
    } catch (e: InterruptedException) {
      Log.e(TAG, e.toString())
    }
  }

  /** Fill the yuvBytes with data from image planes.   */
  private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
    // Row stride is the total number of bytes occupied in memory by a row of an image.
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (i in planes.indices) {
      val buffer = planes[i].buffer
      if (yuvBytes[i] == null) {
        yuvBytes[i] = ByteArray(buffer.capacity())
      }
      buffer.get(yuvBytes[i]!!)
    }
  }

  /** A [OnImageAvailableListener] to receive frames as they are available.  */
  private var imageAvailableListener = object : OnImageAvailableListener {
    override fun onImageAvailable(imageReader: ImageReader) {
      // We need wait until we have some size from onPreviewSizeChosen
      if (previewWidth == 0 || previewHeight == 0) {
        return
      }

      val image = imageReader.acquireLatestImage() ?: return
      fillBytes(image.planes, yuvBytes)

      ImageUtils.convertYUV420ToARGB8888(
        yuvBytes[0]!!,
        yuvBytes[1]!!,
        yuvBytes[2]!!,
        previewWidth,
        previewHeight,
        /*yRowStride=*/ image.planes[0].rowStride,
        /*uvRowStride=*/ image.planes[1].rowStride,
        /*uvPixelStride=*/ image.planes[1].pixelStride,
        rgbBytes
      )

      // Create bitmap from int array
      val imageBitmap = Bitmap.createBitmap(
        rgbBytes, previewWidth, previewHeight,
        Bitmap.Config.ARGB_8888
      )

      // Create rotated version for portrait display
      val rotateMatrix = Matrix()
      rotateMatrix.postRotate(90.0f)

      val rotatedBitmap = Bitmap.createBitmap(
        imageBitmap, 0, 0, previewWidth, previewHeight,
        rotateMatrix, true
      )
      image.close()

      // Process an image for analysis in every 3 frames.
      frameCounter = (frameCounter + 1) % 3
      if (frameCounter == 0) {
        processImage(rotatedBitmap)
      }
    }
  }

  /** Crop Bitmap to maintain aspect ratio of model input.   */
  private fun cropBitmap(bitmap: Bitmap): Bitmap {
    val bitmapRatio = bitmap.height.toFloat() / bitmap.width
    val modelInputRatio = MODEL_HEIGHT.toFloat() / MODEL_WIDTH
    var croppedBitmap = bitmap

    // Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
    val maxDifference = 1e-5

    // Checks if the bitmap has similar aspect ratio as the required model input.
    when {
      abs(modelInputRatio - bitmapRatio) < maxDifference -> return croppedBitmap
      modelInputRatio < bitmapRatio -> {
        // New image is taller so we are height constrained.
        val cropHeight = bitmap.height - (bitmap.width.toFloat() / modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          0,
          (cropHeight / 2).toInt(),
          bitmap.width,
          (bitmap.height - cropHeight).toInt()
        )
      }
      else -> {
        val cropWidth = bitmap.width - (bitmap.height.toFloat() * modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          (cropWidth / 2).toInt(),
          0,
          (bitmap.width - cropWidth).toInt(),
          bitmap.height
        )
      }
    }
    return croppedBitmap
  }

  /** Set the paint color and size.    */
  private fun setPaintRED() {
    paint.color = Color.RED
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }

  private fun setPaintGREEN() {
    paint.color = Color.GREEN
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }

  private fun setcharPaintWHITE(){
    charPaint.color = Color.WHITE
    charPaint.textSize = 65.0f
    charPaint.strokeWidth = 8.0f
    }

  private fun setPaintWHITE() {
    paint.color = Color.WHITE
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }

  private fun setPaintBLUE() {
    paint.color = Color.BLUE
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }

  private fun setPaintGRAY() {
    paint.color = Color.GRAY
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }

  private fun setPaintYELLOW() {
    paint.color = Color.YELLOW
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }

  /** Draw bitmap on Canvas.   */
  private fun draw(canvas: Canvas, person: Person, bitmap: Bitmap) {
    canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
    // Draw `bitmap` and `person` in square canvas.
    val screenWidth: Int
    val screenHeight: Int
    val left: Int
    val right: Int
    val top: Int
    val bottom: Int
    val score : Double
    /** score 계산 **/
    fun calculateScore(person:Person, copy:Person): Int {

      var diff = 0

      if (0 == person.keyPoints.size || 0 == copy.keyPoints.size) {return 0}

      for (i in person.keyPoints.indices) {


        if (person.keyPoints[i].bodyPart.equals(BodyPart.NOSE) || person.keyPoints[i].bodyPart.equals(BodyPart.LEFT_EYE) ||
          person.keyPoints[i].bodyPart.equals(BodyPart.RIGHT_EYE) || person.keyPoints[i].bodyPart.equals(BodyPart.LEFT_EAR) || person.keyPoints[i].bodyPart.equals(BodyPart.RIGHT_EAR)
                ) {

          val eachDiff =
            (person.keyPoints[i].position.y - copy.keyPoints[i].position.y).toDouble()
          Log.d("diffOfFace", eachDiff.toString())
          if (eachDiff < 30) { diff = diff + abs(eachDiff).toInt()} else{diff = diff + (abs(eachDiff)-20).pow(2).toInt()}


        } else if (person.keyPoints[i].bodyPart.equals(BodyPart.LEFT_WRIST) || person.keyPoints[i].bodyPart.equals(BodyPart.RIGHT_WRIST)){

          val eachDiff = (person.keyPoints[i].position.y - copy.keyPoints[i].position.y).toDouble()
          diff = diff + eachDiff.toInt()

        } else if (person.keyPoints[i].bodyPart.equals(BodyPart.LEFT_HIP) || person.keyPoints[i].bodyPart.equals(BodyPart.RIGHT_HIP)){

          val eachDiff =
            (person.keyPoints[i].position.x - copy.keyPoints[i].position.x).toDouble().pow(2) +
                    (person.keyPoints[i].position.y - copy.keyPoints[i].position.y).toDouble().pow(2)
          diff = diff + eachDiff.pow(0.3).toInt()

        } else if (person.keyPoints[i].bodyPart.equals(BodyPart.LEFT_SHOULDER) || person.keyPoints[i].bodyPart.equals(BodyPart.RIGHT_SHOULDER)){

          val eachDiff = (person.keyPoints[i].position.x - copy.keyPoints[i].position.x).toDouble().pow(1.8) +
                  (person.keyPoints[i].position.y - copy.keyPoints[i].position.y).toDouble().pow(2.3)
          diff = diff + eachDiff.pow(0.7).toInt()

        } else{


          val eachDiff =
            (person.keyPoints[i].position.x - copy.keyPoints[i].position.x).toDouble().pow(2) +
                    (person.keyPoints[i].position.y - copy.keyPoints[i].position.y).toDouble().pow(2)
          diff = diff + eachDiff.pow(0.5).toInt()
        }
      }


      return diff/(person.keyPoints.size)
    }
    val difference = calculateScore(person, copy).toDouble()
    score = (1800.div(difference.pow(1.8)+1800))*100

    totalscore += score
    count++

    if (canvas.height > canvas.width) {
      screenWidth = canvas.width
      screenHeight = canvas.width
      left = 0
      top = (canvas.height - canvas.width) / 2
    } else {
      screenWidth = canvas.height
      screenHeight = canvas.height
      left = (canvas.width - canvas.height) / 2
      top = 0
    }
    right = left + screenWidth
    bottom = top + screenHeight


    /**변하는 자세를 그릴 컬러 지정**/
    setPaintGRAY()

    /**실시간 변하는 자세 그리기**/
    canvas.drawBitmap(
      bitmap,
      Rect(0, 0, bitmap.width, bitmap.height),
      Rect(left, top, right, bottom),
      paint
    )

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    // Draw key points over the image.
    for (keyPoint in person.keyPoints) {
      if (keyPoint.score > minConfidence) {
        val position = keyPoint.position
        val adjustedX: Float = position.x.toFloat() * widthRatio + left
        val adjustedY: Float = position.y.toFloat() * heightRatio + top
        canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
      }
    }

    for (line in bodyJoints) {
      if (
        (person.keyPoints[line.first.ordinal].score > minConfidence) and
        (person.keyPoints[line.second.ordinal].score > minConfidence)
      ) {
        canvas.drawLine(
          person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
          person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
          person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
          person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
          paint
        )
      }
    }


    /**고정 할 자세 그릴 컬러 지정**/
    if(mybool) {
      if (score > 85.00) {
        setPaintGREEN()
        mediaplayer?.pause()
        mp3bool = true
      } else if (score > 75.00) {
        setPaintYELLOW()
        mediaplayer?.pause()
        mp3bool = true
      } else {
        setPaintRED()
        Log.d("mp3bool", mp3bool.toString())




        if (mp3bool) {
          mediaplayer = MediaPlayer.create(context,R.raw.alarm)
          mediaplayer?.start()
          Log.d("mp3bool", mp3bool.toString())
          mp3bool = false
        }
      }
    }
    else{
      setPaintWHITE()
      mediaplayer?.pause()
      mp3bool = true
    }

    /**고정 할 자세 그리기**/
    if(mybool) {
       for (keyPoint in copy.keyPoints) {
         if (keyPoint.score > minConfidence) {
           val position = keyPoint.position
           val adjustedX: Float = position.x.toFloat() * widthRatio + left
           val adjustedY: Float = position.y.toFloat() * heightRatio + top
           canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
         }
       }
        for (line in bodyJoints) {
         if (
           (copy.keyPoints[line.first.ordinal].score > minConfidence) and
           (copy.keyPoints[line.second.ordinal].score > minConfidence)
         ) {
           canvas.drawLine(
             copy.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
             copy.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
             copy.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
             copy.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
              paint
           )
         }
       }
     }

    /** 하단 글자색 하얀색으로 지정**/
    setcharPaintWHITE()

    /**하단 글자 띄우기**/
    if(mybool) {
      canvas.drawText(
        "Score: %.2f".format(score),
        (15.0f * widthRatio),
        (30.0f * heightRatio + bottom),
        charPaint
      )
    }
    else {
      canvas.drawText(
        "%s".format("바른자세로 앉은 뒤 스위치를 누르세요."),
        (15.0f * widthRatio),
        (30.0f * heightRatio + bottom),
        charPaint
      )
    }

    /**시간 계산**/
    var totaltime = 0
    if(mybool){
      val tz = TimeZone.getTimeZone("Asia/Seoul")
      val gc = GregorianCalendar(tz)
      var hour= gc.get(GregorianCalendar.HOUR).toInt()
      var min = gc.get(GregorianCalendar.MINUTE).toInt()
      var sec = gc.get(GregorianCalendar.SECOND).toInt()
      endtime = hour*3600 + min*60 + sec
      totaltime = endtime - starttime
    }

    canvas.drawText(
      "Time: %d min %d sec".format(totaltime/60,totaltime%60),
      (15.0f * widthRatio),
      (50.0f * heightRatio + bottom),
      charPaint
    )

    // Draw!
    surfaceHolder!!.unlockCanvasAndPost(canvas)
  }



  /** Process image using Posenet library.   */
  private fun processImage(bitmap: Bitmap) {
    // Crop bitmap.
    val croppedBitmap = cropBitmap(bitmap)

    // Created scaled version of bitmap for model input.
    val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true)

    // Perform inference.
    person = posenet.estimateSinglePose(scaledBitmap)


    val canvas: Canvas = surfaceHolder!!.lockCanvas()

    draw(canvas, person, scaledBitmap)
  }

  /**
   * Creates a new [CameraCaptureSession] for camera preview.
   */
  private fun createCameraPreviewSession() {
    try {

      // We capture images from preview in YUV format.
      imageReader = ImageReader.newInstance(
        previewSize!!.width, previewSize!!.height, ImageFormat.YUV_420_888, 2
      )
      imageReader!!.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

      // This is the surface we need to record images for processing.
      val recordingSurface = imageReader!!.surface

      // We set up a CaptureRequest.Builder with the output Surface.
      previewRequestBuilder = cameraDevice!!.createCaptureRequest(
        CameraDevice.TEMPLATE_PREVIEW
      )
      previewRequestBuilder!!.addTarget(recordingSurface)

            // Here, we create a CameraCaptureSession for camera preview.
            cameraDevice!!.createCaptureSession(
              listOf(recordingSurface),
              object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
                  // The camera is already closed
            if (cameraDevice == null) return

            // When the session is ready, we start displaying the preview.
            captureSession = cameraCaptureSession
            try {
              // Auto focus should be continuous for camera preview.
              previewRequestBuilder!!.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
              )
              // Flash is automatically enabled when necessary.
              setAutoFlash(previewRequestBuilder!!)

              // Finally, we start displaying the camera preview.
              previewRequest = previewRequestBuilder!!.build()
              captureSession!!.setRepeatingRequest(
                previewRequest!!,
                captureCallback, backgroundHandler
              )
            } catch (e: CameraAccessException) {
              Log.e(TAG, e.toString())
            }
          }

          override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
            showToast("Failed")
          }
        },
        null
      )
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    }
  }

  private fun setAutoFlash(requestBuilder: CaptureRequest.Builder) {
    if (flashSupported) {
      requestBuilder.set(
        CaptureRequest.CONTROL_AE_MODE,
        CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
      )
    }
  }

  /**
   * Shows an error message dialog.
   */
  class ErrorDialog : DialogFragment() {

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
      AlertDialog.Builder(activity)
        .setMessage(arguments!!.getString(ARG_MESSAGE))
        .setPositiveButton(android.R.string.ok) { _, _ -> activity!!.finish() }
        .create()

    companion object {

      @JvmStatic
      private val ARG_MESSAGE = "message"

      @JvmStatic
      fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
        arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
      }
    }
  }

  companion object {
    /**
     * Conversion from screen rotation to JPEG orientation.
     */
    private val ORIENTATIONS = SparseIntArray()
    private val FRAGMENT_DIALOG = "dialog"

    init {
      ORIENTATIONS.append(Surface.ROTATION_0, 90)
      ORIENTATIONS.append(Surface.ROTATION_90, 0)
      ORIENTATIONS.append(Surface.ROTATION_180, 270)
      ORIENTATIONS.append(Surface.ROTATION_270, 180)
    }

    /**
     * Tag for the [Log].
     */
    private const val TAG = "PosenetActivity"
  }
}
