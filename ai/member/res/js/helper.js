async function openCamera(video, config) {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // 设置分辨率和帧率的约束条件
    const constraints = {
      video: {
        width: { ideal: 1080 }, // 期望的宽度
        height: { ideal: 1080 }, // 期望的高度
        frameRate: { ideal: 30 }, // 期望的帧率
        facingMode: "environment",
        ...config
      }
    };

    // 请求获取摄像头流
    const stream = await navigator.mediaDevices.getUserMedia(constraints)

    video.srcObject = stream;
    video.play();
    return new Promise(resolve => {
      // Add event listener to make sure the webcam has been fully initialized.
      video.onloadedmetadata = () => {
        resolve();
      };
    });
  } else {
    throw "浏览器不支持 getUserMedia 方法"
  }
}